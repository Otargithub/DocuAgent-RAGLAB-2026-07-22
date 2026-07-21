"""
Métricas de recuperación, compartidas por los cinco bancos.

Modelo de relevancia: HÍBRIDO (exactitud + significado)
-------------------------------------------------------
La cita de test (`source_context`) es (idealmente) una copia literal del
documento; `corpus.py` valida que así sea antes de que llegue aquí. Eso da dos
formas complementarias de juzgar si un fragmento recuperado es relevante:

  1. EXACTITUD (léxico). ¿Contiene, palabra por palabra, una parte sustancial de
     la cita? Se mide por solapamiento de n-gramas de palabras (5-gramas). Es
     verdad de terreno dura, pero da falsos negativos ante paráfrasis.

  2. SIGNIFICADO (semántico). ¿Se PARECE en sentido a la cita? Se mide por
     similitud del coseno usando un embedding ÁRBITRO FIJO (siempre el mismo,
     definido en CONFIG.embedding_juez), distinto del modelo que se evalúa.

     IMPORTANTE — recalibración del suelo. Los embeddings tipo e5 dan un coseno
     base alto (~0.7) incluso a texto no relacionado. Sin corregirlo, el término
     semántico es casi constante entre fragmentos: no discrimina e infla el
     recall con un suelo fijo. Por eso reescalamos con un SUELO calibrado
     EMPÍRICAMENTE por árbitro (evaluacion.obtener_base_semantica: mediana del
     coseno entre pares de textos no relacionados):

         sem = clip( (cos - base) / (1 - base), 0, 1 )

     base=0.0 reproduce el comportamiento antiguo (solo recorte a [0,1]).

Se combinan en una sola relevancia por fragmento:

    relevancia(c) = α · léxico(c) + (1 − α) · semántico(c)         con α ∈ [0, 1]

donde α = CONFIG.peso_exactitud es una PERILLA REAL (ya no una constante muerta):
se puede mover desde config o por entorno (RAG_PESO_EXACTITUD) para barridos.
  - α = 1 (y sin similitudes) reproduce EXACTAMENTE la métrica solo-léxica.
  - El árbitro es FIJO, así que comparar dos modelos de embeddings los corrige el
    mismo juez. Esto es condición NECESARIA pero no suficiente para la justicia:
    exige además que el árbitro esté FUERA de la parrilla evaluada (y a ser
    posible fuera de su familia representacional). Ese guardarraíl vive en
    evaluacion.py / embeddings_banco.py, no aquí.

Qué mide EXACTAMENTE cada número (nombres honestos; todas en [0, 1]):
  - relevancia : relevancia del MEJOR fragmento recuperado. (Antes se llamaba
                "precision", nombre PRESTADO y enganoso: la precision clasica de
                IR es la fraccion de recuperados relevantes, otra cosa. Este
                numero es la nota del mejor fragmento — un proxy de calidad del
                contexto. Su agregado por banco es 'relevancia_media'.)
  - mrr       : 1 / posición del primer fragmento con relevancia ≥ umbral.
  - recuperacion_pasaje (antes 'recall'): recuperación del PASAJE,
                asimétrica A PROPÓSITO —
                    α · (cobertura léxica de la UNIÓN de fragmentos)   [reensamblado]
                  + (1−α) · (mejor equivalencia semántica de UN fragmento).
                Las dos ramas son dos modos legítimos de "recuperar el pasaje"
                (reensamblarlo literalmente troceado, o traerlo entero en un
                fragmento equivalente), no la misma cantidad medida dos veces.

Por qué se ancla al pasaje de test y no a la pregunta: medir similitud contra la
pregunta en crudo premiaría a fragmentos que "hablan del tema" sin contener la
respuesta. El pasaje de test es lo que de verdad responde.

Separación de responsabilidades: este módulo solo hace la MEZCLA. Quien llama
(evaluacion.py, rerankers.py, fase_b.py) calcula las similitudes semánticas con
el árbitro y su caché, y se las pasa ya hechas.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Parámetros por defecto: α y umbral se leen de CONFIG (perillas reales), con
# respaldo si CONFIG no está disponible (tests puramente aritméticos).
# La base semántica NO se lee aquí: la CALIBRA empíricamente evaluacion.py por
# árbitro (obtener_base_semantica) y llega como argumento. El default 0.0 de
# las funciones equivale a "sin recalibrar" (comportamiento neutro y seguro).
# --------------------------------------------------------------------------- #
try:  # import guardado: no obligamos a tener entorno para testear la aritmética
    from .config import CONFIG
    _PESO_EXACTITUD_DEF = float(CONFIG.peso_exactitud)
    _UMBRAL_DEF = float(CONFIG.umbral_relevancia)
except Exception:  # noqa: BLE001
    _PESO_EXACTITUD_DEF = 0.5
    _UMBRAL_DEF = 0.5

N_GRAMA = 5                          # tamaño del n-grama de palabras para el léxico
UMBRAL_RELEVANCIA = _UMBRAL_DEF      # relevancia mínima para que un fragmento cuente (mrr)
PESO_EXACTITUD = _PESO_EXACTITUD_DEF # α: cuánto pesa la exactitud frente al significado

# Centinela de dispersión "sin señal": cuando la media de la métrica es 0 no hay
# estabilidad que premiar; el CV clásico daría 0/0 -> 0 (¡el MEJOR valor de un
# objetivo 'min'!) y una config totalmente fallida ganaría los desempates. Este
# centinela la manda al final. Lo usan retrievers/rerankers/splitters.
CV_SIN_SENAL = 9999.0

_TOKEN = re.compile(r"[0-9a-záéíóúüñ]+")


# --------------------------------------------------------------------------- #
# Exactitud: solapamiento léxico de n-gramas
# --------------------------------------------------------------------------- #
def _tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin puntuación, deshaciendo guiones de fin de línea del PDF."""
    texto = re.sub(r"-\s*\n\s*", "", texto)
    return _TOKEN.findall(texto.casefold())


def _ngramas(tokens: list[str], n: int = N_GRAMA) -> set[tuple]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def gramas_de(texto: str, n: int = N_GRAMA) -> set[tuple]:
    """n-gramas de un texto como conjunto (para precomputar la cita)."""
    return _ngramas(_tokenizar(texto), n)


def cobertura(grams_fragmento: set[tuple], grams_cita: set[tuple]) -> float:
    """Fracción de los n-gramas de la cita presentes en el fragmento ([0, 1])."""
    if not grams_cita:
        return 0.0
    return len(grams_fragmento & grams_cita) / len(grams_cita)


# --------------------------------------------------------------------------- #
# Significado: similitud semántica con un árbitro fijo (recalibrada)
# --------------------------------------------------------------------------- #
def recalibrar_similitud(cos: float, base: float = 0.0) -> float:
    """
    Lleva un coseno crudo a [0, 1] quitando el SUELO del árbitro.

        sem = clip( (cos - base) / (1 - base), 0, 1 )

    `base` es el coseno típico que el árbitro asigna a texto NO relacionado
    (los e5 rondan 0.7; los OpenAI, mucho menos). Se calibra EMPÍRICAMENTE por
    árbitro en evaluacion.obtener_base_semantica; el default 0.0 equivale al
    simple recorte a [0,1] (sin recalibrar). Se satura a [0, 0.99] para no
    dividir por ~0.
    """
    base = min(0.99, max(0.0, float(base)))
    return float(np.clip((cos - base) / (1.0 - base), 0.0, 1.0))


def vector_normalizado(texto: str, embed_juez: Any, cache: dict | None = None) -> np.ndarray:
    """
    Embedding L2-normalizado de un texto con el modelo árbitro, con caché por texto.

    El árbitro es siempre el mismo modelo (lo construye quien llama), así que dos
    experimentos distintos obtienen la misma similitud para el mismo fragmento.
    """
    if cache is not None and texto in cache:
        return cache[texto]
    v = np.asarray(embed_juez.embed_documents([texto])[0], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-10)
    if cache is not None:
        cache[texto] = v
    return v


def similitudes_semanticas(textos: list[str], emb_cita: np.ndarray,
                          embed_juez: Any, cache: dict | None = None,
                          base: float = 0.0) -> list[float]:
    """
    Similitud del coseno de cada texto con la cita (vector ya normalizado),
    RECALIBRADA a [0, 1] con `recalibrar_similitud`. Lista paralela a `textos`.
    `base` debe ser el suelo CALIBRADO del árbitro (evaluacion.
    obtener_base_semantica); 0.0 = sin recalibrar.
    """
    sims = []
    for t in textos:
        v = vector_normalizado(t, embed_juez, cache)
        sims.append(recalibrar_similitud(float(v @ emb_cita), base))
    return sims


# --------------------------------------------------------------------------- #
# Mezcla: la nota de recuperación
# --------------------------------------------------------------------------- #
def evaluar_recuperacion(docs_recuperados: list, source_context,
                        sims: list[float] | None = None,
                        peso_exactitud: float = PESO_EXACTITUD,
                        umbral: float = UMBRAL_RELEVANCIA, n: int = N_GRAMA) -> dict:
    """
    Combina exactitud léxica y (opcionalmente) similitud semántica.

    Parámetros
    ----------
    docs_recuperados : documentos recuperados EN ORDEN (objetos con .page_content).
    source_context   : cita de test (texto) o sus n-gramas ya precomputados (set).
    sims             : similitudes semánticas por fragmento (YA recalibradas por
                       quien llama con el árbitro). Si es None, la nota es
                       SOLO-LÉXICA (equivale a peso_exactitud = 1.0).
    peso_exactitud   : α ∈ [0, 1]. Cuánto pesa la exactitud frente al significado.
    umbral           : relevancia mínima para contar como acierto (para el MRR).

    Devuelve {"relevancia", "mrr", "recuperacion_pasaje"} en [0, 1]. Ver el docstring del
    módulo para el significado EXACTO de cada uno (los nombres son proxies).
    """
    grams_cita = source_context if isinstance(source_context, (set, frozenset)) \
        else gramas_de(source_context or "", n)

    if not docs_recuperados or not grams_cita:
        return {"relevancia": 0.0, "mrr": 0.0, "recuperacion_pasaje": 0.0}

    # GUARDA: sims debe ser paralela a los docs. Sin esto, zip() truncaría en
    # silencio y la nota se calcularía sobre un subconjunto arbitrario.
    if sims is not None and len(sims) != len(docs_recuperados):
        raise ValueError(
            f"sims tiene {len(sims)} valores para {len(docs_recuperados)} "
            f"documentos; deben ser listas paralelas.")

    # N-GRAMA EFECTIVO: si la cita tiene menos de `n` tokens, sus "n-gramas" son
    # una única tupla más corta. Los fragmentos deben trocearse con ESA longitud;
    # si no, la intersección sería siempre vacía y una cita corta contenida
    # literalmente en el fragmento puntuaría 0 (falso negativo estructural).
    n_ef = n
    if grams_cita:
        lg_min = min(len(t) for t in grams_cita)
        if 0 < lg_min < n_ef:
            n_ef = lg_min

    # Sin similitudes no se puede hacer semántica: caemos a solo-léxico.
    alpha = 1.0 if sims is None else float(np.clip(peso_exactitud, 0.0, 1.0))

    lexico, grams_union = [], set()
    for doc in docs_recuperados:
        g = gramas_de(getattr(doc, "page_content", "") or "", n_ef)
        lexico.append(cobertura(g, grams_cita))
        grams_union |= g

    semantico = sims if sims is not None else [0.0] * len(docs_recuperados)

    # Relevancia combinada por fragmento.
    relevancias = [alpha * lx + (1 - alpha) * sm for lx, sm in zip(lexico, semantico)]

    # relevancia := la del MEJOR fragmento recuperado.
    relevancia = max(relevancias)

    mrr = 0.0
    for posicion, r in enumerate(relevancias, start=1):
        if r >= umbral:
            mrr = 1.0 / posicion
            break

    # recuperacion_pasaje := recuperación del pasaje (antes 'recall', renombrado:
    # no es el recall clásico de IR ni depende de ningún k). Rama léxica =
    # cobertura de la UNIÓN
    # (reensamblado troceado); rama semántica = mejor fragmento equivalente.
    # Asimetría intencionada: son dos modos de recuperar el mismo pasaje.
    cobertura_union = cobertura(grams_union, grams_cita)
    mejor_sem = max(semantico) if semantico else 0.0
    recall = alpha * cobertura_union + (1 - alpha) * mejor_sem

    return {"relevancia": round(float(relevancia), 4),
            "mrr": round(float(mrr), 4),
            "recuperacion_pasaje": round(float(recall), 4)}


# --------------------------------------------------------------------------- #
# Utilidades que ya usaban otros módulos (se conservan)
# --------------------------------------------------------------------------- #
def coseno_lote(vector_objetivo: np.ndarray, vectores: np.ndarray) -> np.ndarray:
    """Similitud del coseno entre un vector objetivo y un lote de vectores."""
    objetivo_norm = vector_objetivo / (np.linalg.norm(vector_objetivo) + 1e-10)
    vectores_norm = vectores / (np.linalg.norm(vectores, axis=1, keepdims=True) + 1e-10)
    return vectores_norm @ objetivo_norm


def coeficiente_variacion(textos: list[str]) -> float:
    """Desviación típica / media de las longitudes. Bajo = fragmentos homogéneos."""
    longitudes = [len(t) for t in textos]
    if not longitudes:
        return 0.0
    media = float(np.mean(longitudes))
    return float(np.std(longitudes) / media) if media else 0.0


def normalizar(valor: float, minimo: float, maximo: float, invertir: bool = False) -> float:
    """
    Lleva un valor a [0, 1]. `invertir=True` si 'menos es mejor'.

    Caso degenerado (max == min): todos los valores son iguales, así que no hay
    información para distinguirlos; se devuelve 0.5 NEUTRO. (Antes devolvía 1.0,
    que premiaba arbitrariamente el empate como si fuera el mejor valor.)
    """
    if maximo == minimo:
        return 0.5
    n = (valor - minimo) / (maximo - minimo)
    return 1.0 - n if invertir else n
