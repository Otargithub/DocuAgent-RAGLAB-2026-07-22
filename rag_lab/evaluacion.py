"""
Evaluación sobre las preguntas de test.

Los cinco bancos recuperan documentos para cada pregunta de test y comprueban,
con la nota híbrida (exactitud léxica + similitud semántica, ver metricas.py), si
lo recuperado responde a lo pedido. Ese bucle vive aquí una sola vez.

El árbitro semántico
--------------------
La parte semántica de la nota usa un embedding ÁRBITRO FIJO, independiente del
modelo que cada banco esté evaluando. Eso es lo que mantiene justa la comparación
entre modelos. Aquí se construye una sola vez (perezosamente) y se puede:
  - cambiar su modelo con `EMBEDDING_JUEZ`;
  - inyectar uno de prueba con `fijar_embedding_juez(...)` (lo usan los tests);
  - desactivar la semántica con `usar_semantica=False` (nota solo-léxica).

El árbitro debería ser LOCAL (HuggingFace) para no añadir coste de API: la nota se
calcula muchas veces. Por defecto un multilingüe pequeño (es/en).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .config import CONFIG
from .metricas import (
    PESO_EXACTITUD,
    evaluar_recuperacion,
    gramas_de,
    similitudes_semanticas,
    vector_normalizado,
)

# Modelo árbitro: local y multilingüe (es/en) por defecto, pero configurable en
# CONFIG.embedding_juez para poder sacarlo de la familia de los candidatos.
EMBEDDING_JUEZ = dict(CONFIG.embedding_juez)

_JUEZ: Any | None = None              # instancia única del árbitro
_CACHE_JUEZ: dict[str, np.ndarray] = {}  # caché de vectores por texto


# --------------------------------------------------------------------------- #
# Guardarraíl de independencia: el árbitro NO debe estar en la parrilla evaluada
# --------------------------------------------------------------------------- #
def _familia(model: str) -> str:
    """
    Raíz de familia de un identificador de modelo, para detectar parentesco
    representacional (no solo identidad exacta). Ej.: 'intfloat/multilingual-e5
    -small' -> 'e5'; 'BAAI/bge-m3' -> 'bge'. Heurística deliberadamente simple.
    """
    m = (model or "").split("/")[-1].casefold()
    for raiz in ("e5", "bge", "gte", "voyage", "text-embedding", "nomic", "jina"):
        if raiz in m:
            return raiz
    return m


def relacion_con_arbitro(config: dict) -> str:
    """
    Clasifica un modelo candidato frente al árbitro actual:
      "identico"  -> mismo modelo exacto (autoevaluación: hay que EXCLUIRLO).
      "familia"   -> misma familia representacional (sesgo de afinidad: AVISAR).
      "ok"        -> independiente.
    """
    cand = str((config or {}).get("model", ""))
    arb = str(EMBEDDING_JUEZ.get("model", ""))
    if not cand or not arb:
        return "ok"
    if cand.casefold() == arb.casefold():
        return "identico"
    if _familia(cand) == _familia(arb):
        return "familia"
    return "ok"


def fijar_embedding_juez(embedder: Any) -> None:
    """Inyecta un árbitro concreto (para tests o para forzar un modelo)."""
    global _JUEZ
    _JUEZ = embedder
    _CACHE_JUEZ.clear()
    _BASES_CALIBRADAS.clear()   # el suelo es POR árbitro: recalibrar con el nuevo


def limpiar_cache_juez() -> None:
    """Vacía la caché de vectores del árbitro (conviene hacerlo entre documentos)."""
    _CACHE_JUEZ.clear()


def obtener_embedding_juez() -> Any:
    """Devuelve el árbitro, construyéndolo perezosamente la primera vez."""
    global _JUEZ
    if _JUEZ is None:
        from .fabricas import construir_embedding  # import perezoso (evita deps en tests)
        _JUEZ = construir_embedding(EMBEDDING_JUEZ)
    return _JUEZ


# --------------------------------------------------------------------------- #
# Calibración EMPÍRICA del suelo semántico del árbitro
# --------------------------------------------------------------------------- #
# Cada familia de embeddings asigna un coseno base distinto a texto NO
# relacionado (los e5 rondan 0.7-0.8; los OpenAI, cerca de 0). Ese suelo es una
# propiedad DEL ÁRBITRO, no de los datos: se estima una vez por modelo y se
# cachea. Método: sondas fijas de temas deliberadamente dispares (es/en, mismos
# idiomas que el corpus objetivo del laboratorio), coseno de TODOS los pares
# (ninguno relacionado con ningún otro) y MEDIANA como estimador robusto.
#
# La mediana (y no el máximo) es a propósito: el suelo debe absorber el coseno
# TÍPICO del ruido, no el peor caso; usar el máximo comería señal débil real.
# Determinista: sondas fijas => misma base para el mismo árbitro, siempre.
_SONDAS_CALIBRACION: tuple[str, ...] = (
    "La fotosíntesis convierte la luz solar en energía química dentro de los cloroplastos.",
    "El tipo de interés del banco central subió veinticinco puntos básicos este trimestre.",
    "The goalkeeper saved two penalties during the championship final on Sunday.",
    "La receta lleva harina de trigo, levadura fresca y una pizca de sal marina.",
    "Quantum entanglement links the states of two particles across large distances.",
    "El juzgado admitió a trámite la demanda por incumplimiento de contrato mercantil.",
    "The volcanic eruption deposited a thick layer of ash over the northern valley.",
    "La sinfonía abre con un tema lento en los violonchelos y contrabajos.",
    "Compilers translate high-level source code into machine instructions.",
    "El velero cruzó el estrecho con viento de popa y mar ligeramente rizada.",
    "Ancient trade routes carried silk, spices and silver across the continent.",
    "La vacuna requiere dos dosis separadas por un intervalo de cuatro semanas.",
)

_BASES_CALIBRADAS: dict[str, float] = {}   # clave: identidad del árbitro


def calibrar_base_semantica(juez: Any) -> float:
    """
    Estima el suelo del coseno de `juez`: mediana de las similitudes entre todos
    los pares de sondas no relacionadas. Devuelve un valor en [0, 0.95].
    """
    vecs = np.stack([vector_normalizado(t, juez, None) for t in _SONDAS_CALIBRACION])
    sim = vecs @ vecs.T
    n = len(_SONDAS_CALIBRACION)
    pares = sim[np.triu_indices(n, k=1)]          # 66 pares, todos "no relacionados"
    base = float(np.clip(np.median(pares), 0.0, 0.95))
    return round(base, 4)


def _clave_juez(juez: Any) -> str:
    """Identidad estable del árbitro para la caché de bases calibradas."""
    for attr in ("model_name", "model"):
        v = getattr(juez, attr, None)
        if isinstance(v, str) and v:
            return v
    return f"{type(juez).__name__}@{id(juez)}"


def obtener_base_semantica(juez: Any | None = None) -> float:
    """
    Suelo semántico a usar en la nota:
      - Si CONFIG.base_semantica es un número, MANDA (override manual).
      - Si es None (por defecto), se calibra EMPÍRICAMENTE una vez por árbitro
        y se cachea en el proceso.
    """
    fijo = getattr(CONFIG, "base_semantica", None)
    if fijo is not None:
        return float(fijo)
    juez = juez if juez is not None else obtener_embedding_juez()
    clave = _clave_juez(juez)
    if clave not in _BASES_CALIBRADAS:
        base = calibrar_base_semantica(juez)
        _BASES_CALIBRADAS[clave] = base
        print(f"[evaluacion] Suelo semántico calibrado para el árbitro "
              f"'{clave}': base={base} (mediana de {len(_SONDAS_CALIBRACION)} "
              f"sondas no relacionadas, "
              f"{len(_SONDAS_CALIBRACION)*(len(_SONDAS_CALIBRACION)-1)//2} pares).")
    return _BASES_CALIBRADAS[clave]


def embeber_con_reintento(modelo_embedding: Any, textos: list[str],
                        max_intentos: int = 5) -> list:
    """Embebe reintentando ante límites de ritmo (utilidad general)."""
    for intento in range(max_intentos):
        try:
            return modelo_embedding.embed_documents(textos)
        except Exception as error:  # noqa: BLE001
            mensaje = str(error).lower()
            if "rate" in mensaje or "429" in mensaje:
                espera = 2 ** intento
                print(f"[evaluacion] Límite de ritmo; espero {espera}s "
                      f"(intento {intento + 1}/{max_intentos})")
                time.sleep(espera)
            else:
                raise
    raise RuntimeError("Se agotaron los reintentos al embeber.")


def evaluar_sobre_preguntas(recuperar: Callable[[str], list],
                          preguntas_test: list[dict],
                          modelo_embedding: Any = None,
                          peso_exactitud: float = PESO_EXACTITUD,
                          usar_semantica: bool = True) -> dict:
    """
    Evalúa una función de recuperación contra todas las preguntas de test.

    Parámetros
    ----------
    recuperar        : función consulta(str) -> lista de documentos recuperados.
    preguntas_test : preguntas de test (cada una con 'question' y 'source_context').
    modelo_embedding : ignorado para puntuar (se mantiene por compatibilidad con
                       las llamadas de los bancos; la semántica usa el ÁRBITRO).
    peso_exactitud   : α ∈ [0, 1]; cuánto pesa la exactitud frente al significado.
    usar_semantica   : si False, nota solo-léxica (no construye el árbitro).

    Devuelve medias de relevancia, MRR y recall, más las listas por pregunta.
    """
    grams_objetivo = [gramas_de(gq.get("source_context", "")) for gq in preguntas_test]

    juez = obtener_embedding_juez() if usar_semantica else None
    base = obtener_base_semantica(juez) if juez else 0.0
    # La cita de test se embebe una sola vez por pregunta (vector normalizado).
    emb_objetivo = ([vector_normalizado(gq.get("source_context", ""), juez, _CACHE_JUEZ)
                     for gq in preguntas_test] if juez else None)

    relevancias, mrrs, recalls = [], [], []
    for i, gq in enumerate(preguntas_test):
        docs = recuperar(gq.get("question", ""))
        sims = None
        if juez and docs:
            textos = [getattr(d, "page_content", "") or "" for d in docs]
            sims = similitudes_semanticas(textos, emb_objetivo[i], juez, _CACHE_JUEZ,
                                          base=base)
        m = evaluar_recuperacion(docs, grams_objetivo[i], sims=sims,
                                 peso_exactitud=peso_exactitud)
        relevancias.append(m["relevancia"])
        mrrs.append(m["mrr"])
        recalls.append(m["recuperacion_pasaje"])

    return {
        "relevancia_media": round(float(np.mean(relevancias)), 4) if relevancias else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "recuperacion_pasaje": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "_relevancias": relevancias,
        "_mrrs": mrrs,
        "_recuperaciones": recalls,
    }
