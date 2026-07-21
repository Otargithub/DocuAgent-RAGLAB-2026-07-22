"""
Preparación de la sesión de trabajo.

Una "sesión" es todo lo que los cuatro bancos deben COMPARTIR para que sus
resultados sean comparables:

  - el documento cargado (para no releer el PDF en cada banco),
  - el tipo de documento (clasificado una sola vez),
  - sus metadatos estructurales,
  - y las preguntas de test (generadas una sola vez).

Si cada banco generase sus propias preguntas, estaríamos comparando peras con
manzanas. Por eso esto se hace UNA vez, aquí, y se pasa a todos por igual.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from .fabricas import construir_llm_clasificador
from .metricas import gramas_de, cobertura
from .perfil import perfil_de_documento, descriptor
from .config import CONFIG

# Fracción mínima de los 5-gramas de la cita que deben aparecer en el documento
# para aceptarla como "literal" (tolera el ruido de extracción del PDF: espacios,
# guiones de línea, saltos). Por debajo, la cita se considera parafraseada o
# inventada por el LLM y se descarta: es verdad de terreno no verificable.
COBERTURA_MINIMA_CITA = 0.5


def _normalizar_para_busqueda(texto: str) -> str:
    """Minúsculas, sin guiones de fin de línea del PDF y con espacios colapsados."""
    texto = re.sub(r"-\s*\n\s*", "", texto)
    return re.sub(r"\s+", " ", texto.casefold()).strip()


def validar_citas(preguntas: list[dict], documentos: list[Any],
                  cobertura_min: float = COBERTURA_MINIMA_CITA,
                  minimo: int = 3) -> list[dict]:
    """
    Filtra las preguntas cuya `source_context` NO aparece de verdad en el
    documento. El LLM tiene instrucciones de copiar literalmente, pero no siempre
    obedece; sin esta comprobación, la métrica léxica mide contra una cita que
    puede no existir en el corpus (verdad de terreno fantasma).

    Criterio en dos niveles, tolerante al ruido de extracción del PDF:
      1) coincidencia literal tras normalizar (subcadena exacta), o
      2) que al menos `cobertura_min` de los 5-gramas de la cita estén en el doc.

    Guardarraíl: si el filtro dejaría MENOS de 3 preguntas (o ninguna), es más
    probable un desajuste sistemático de extracción que citas inventadas una a
    una; en ese caso se conservan todas y se AVISA en vez de vaciar el test.
    """
    if not preguntas:
        return preguntas
    texto_norm = _normalizar_para_busqueda(
        "\n".join(getattr(d, "page_content", "") or "" for d in documentos))
    grams_doc = gramas_de(texto_norm)

    validas, descartadas = [], []
    for q in preguntas:
        cita = (q.get("source_context") or "").strip()
        if not cita:
            descartadas.append((q, "cita vacía"))
            continue
        if _normalizar_para_busqueda(cita) in texto_norm:
            validas.append(q)
            continue
        cob = cobertura(grams_doc, gramas_de(cita))
        if cob >= cobertura_min:
            validas.append(q)
        else:
            descartadas.append((q, f"cobertura {cob:.2f} < {cobertura_min}"))

    if descartadas:
        for q, motivo in descartadas:
            preg = (q.get("question") or "")[:70]
            print(f"[sesion] Cita NO verificable ({motivo}); descarto pregunta: "
                  f"{preg!r}")

    # Guardarraíl SOLO si de verdad hubo descartes: sin descartes no hay nada
    # que sospechar aunque el lote sea pequeño (antes, un lote de 1 pregunta
    # 100% verificada disparaba una falsa alarma de "medición poco fiable").
    if descartadas and len(validas) < minimo:
        print(f"[sesion] AVISO: solo {len(validas)}/{len(preguntas)} citas se "
              f"verificaron. Probable desajuste de extracción del PDF (no citas "
              f"inventadas): CONSERVO todas y marco la medición como poco fiable.")
        return preguntas

    if descartadas:
        print(f"[sesion] Citas verificadas: {len(validas)}/{len(preguntas)} "
              f"({len(descartadas)} descartadas).")
    return validas


# --------------------------------------------------------------------------- #
# Verificación de preguntas: convertir el examen sintético en uno de confianza
# --------------------------------------------------------------------------- #
# La confianza en una pregunta de test son CUATRO propiedades distintas, cada
# una con su verificación:
#   Capa 1 (hecho, gratis)      la cita existe en el documento  -> validar_citas
#   Capa 3 (regex, gratis)      la pregunta es autocontenida    -> _es_autocontenida
#   Capa 4 (árbitro, céntimos)  la pregunta discrimina su cita  -> _discrimina
#   Capa 2 (juez LLM, cara)     la cita responde a la pregunta  -> round-trip QA
#
# Capa 2 usa un verificador CRUZADO: el juez (Claude) verifica lo que generó el
# clasificador (gpt-4o-mini). Modelos distintos = errores no correlacionados.
# El nivel se controla con CONFIG.verificacion_preguntas (citas|barata|completa).
# Política de fallo: la verificación NUNCA rompe la sesión; si una capa no puede
# ejecutarse (sin red, sin credenciales), se omite con aviso y consta en el
# informe. Guardarraíl compartido con validar_citas: nunca dejar menos de 3
# preguntas; antes de eso, conservar y marcar la medición como poco fiable.

# Deícticos que hacen una pregunta IRRECUPERABLE por diseño: solo tienen sentido
# con el documento delante, así que ningún retriever puede acertarlas.
_DEICTICOS = re.compile(
    r"\b(?:este|esta|dicho|dicha|el\s+presente|la\s+presente)\s+"
    r"(?:documento|texto|art[ií]culo|cap[ií]tulo|secci[oó]n|p[aá]rrafo|"
    r"fragmento|extracto|pasaje|informe|obra)\b"
    r"|seg[uú]n\s+(?:el\s+texto|lo\s+anterior|el\s+fragmento|la\s+cita)"
    r"|\b(?:anteriormente|arriba|abajo)\s+(?:mencionad|citad|descrit)"
    r"|\bla\s+(?:tabla|figura|p[aá]gina|secci[oó]n)\s+\d+"
    r"|\bel\s+autor\s+del\s+texto\b",
    re.IGNORECASE)


def _es_autocontenida(pregunta: str) -> bool:
    """Capa 3: sin referencias deícticas al documento ('este texto', 'tabla 3')."""
    return not _DEICTICOS.search(pregunta or "")


class _VeredictoQA(BaseModel):
    indice: int
    respondible: bool      # ¿la pregunta puede responderse SOLO con la cita?
    coincide: bool         # ¿la expected_answer es correcta según la cita?


class _ListaVeredictosQA(BaseModel):
    veredictos: list[_VeredictoQA]


def _verificador_qa_por_defecto(items: list[dict]) -> list[dict]:
    """
    Capa 2 por defecto: round-trip con el LLM JUEZ (cruzado respecto al
    generador). Una sola llamada estructurada por tanda. Devuelve una lista de
    {indice, respondible, coincide}.
    """
    from .fabricas import construir_llm_juez  # import perezoso
    llm = construir_llm_juez().with_structured_output(_ListaVeredictosQA)
    bloques = []
    for it in items:
        bloques.append(
            f"### Ítem {it['indice']}\n"
            f"CITA (única fuente permitida):\n{it['source_context']}\n\n"
            f"PREGUNTA: {it['question']}\n"
            f"RESPUESTA ESPERADA: {it['expected_answer']}")
    sistema = (
        "Eres el verificador de un examen para sistemas RAG. Para CADA ítem, "
        "usando EXCLUSIVAMENTE la cita (nada de conocimiento externo), responde "
        "dos cosas: `respondible` = la pregunta puede contestarse solo con la "
        "cita; `coincide` = la respuesta esperada es correcta según la cita. "
        "Sé estricto: si hay que suponer información que la cita no contiene, "
        "`respondible` es false. Devuelve un veredicto por ítem, con su índice.")
    res: _ListaVeredictosQA = llm.invoke(
        [("system", sistema), ("human", "\n\n".join(bloques))])
    return [v.model_dump() for v in res.veredictos]


def _dedupe_preguntas(preguntas: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Capa 0 de la verificacion: PREGUNTAS GEMELAS fuera. El generador produce a
    veces dos parafrasis de la misma pregunta sobre el mismo pasaje (visto en
    Annie: '¿Que recuerda el narrador sobre su madre al ver su apariencia en el
    sillon?' y '¿Que le recuerda al narrador la imagen de su madre en el
    sillon?', ambas congeladas). Ninguna capa las cazaba: verifican cada
    pregunta contra el documento, no entre si. Efectos de los gemelos: los
    pasajes favoritos del generador pesan doble en TODAS las metricas y el
    mismo error cuenta dos veces.

    Criterio: solapamiento Jaccard de palabras entre preguntas >= umbral, o
    citas practicamente identicas (>= 0.8). Se conserva la primera.
    """
    def _palabras_de(t: str) -> set:
        return set(_normalizar_para_busqueda(t or "").split())

    unicas, descartadas = [], []
    for q in preguntas:
        pal_q = _palabras_de(q.get("question", ""))
        pal_c = _palabras_de(q.get("source_context", ""))
        gemela = False
        for u in unicas:
            pal_u = _palabras_de(u.get("question", ""))
            j_preg = (len(pal_q & pal_u) / len(pal_q | pal_u)) if pal_q | pal_u else 0.0
            pal_cu = _palabras_de(u.get("source_context", ""))
            j_cita = (len(pal_c & pal_cu) / len(pal_c | pal_cu)) if pal_c | pal_cu else 0.0
            # Gemela = misma pregunta reformulada: preguntas PARECIDAS sobre la
            # MISMA cita, o preguntas casi identicas. La cita compartida a
            # secas NO basta: dos preguntas distintas sobre el mismo pasaje
            # ("¿quien descubrio X?" / "¿en que año?") son legitimas.
            if (j_preg >= 0.5 and j_cita >= 0.6) or j_preg >= 0.85:
                gemela = True
                break
        if gemela:
            descartadas.append(q.get("question", ""))
        else:
            unicas.append(q)
    return unicas, descartadas


def verificar_preguntas(preguntas: list[dict], documentos: list[Any], *,
                        juez: Any = None, base: float | None = None,
                        verificador_qa: Any = None,
                        nivel: str | None = None,
                        minimo: int = 3) -> dict:
    """
    Verifica un lote de preguntas YA validadas por citas (capa 1) con las capas
    3 (autocontención), 4 (discriminación semántica) y 2 (round-trip QA), según
    `nivel` (por defecto CONFIG.verificacion_preguntas).

    Devuelve {"validas": [...], "informe": [...], "tasa": float}. El informe
    tiene una entrada por pregunta con el resultado de cada capa y el motivo del
    descarte si lo hay. Guardarraíl: si quedaran menos de `minimo`, se conservan
    todas (marcadas) y se avisa: un examen diezmado señala que hay que subir el
    modelo generador, no medir contra 2 preguntas.
    """
    nivel = (nivel or getattr(CONFIG, "verificacion_preguntas", "completa")).casefold()
    preguntas, gemelas = _dedupe_preguntas(preguntas)
    for g in gemelas:
        print(f"[sesion] Pregunta descartada (gemela de otra del examen): "
              f"'{g[:70]}'")
    if not preguntas or nivel == "citas":
        return {"validas": preguntas,
                "informe": [{"question": q.get("question", ""), "veredicto": "ok",
                             "capas": {"solo_citas": True}} for q in preguntas],
                "tasa": 1.0 if preguntas else 0.0}

    informe = [{"question": q.get("question", ""), "veredicto": "ok",
                "capas": {}, "motivo": None} for q in preguntas]

    # ---- Capa 3: autocontención (gratis) --------------------------------- #
    for q, inf in zip(preguntas, informe):
        ok = _es_autocontenida(q.get("question", ""))
        inf["capas"]["autocontenida"] = ok
        if not ok and inf["veredicto"] == "ok":
            inf["veredicto"], inf["motivo"] = "descartada", "referencia deíctica al documento"

    # ---- Capa 4: discriminación con el árbitro semántico ------------------ #
    # La cita debe ser MÁS afín a la pregunta que cualquier página ajena: si la
    # pregunta 'suena' igual en todo el corpus, acertarla no mide recuperación.
    try:
        if juez is None or base is None:
            from .evaluacion import obtener_base_semantica, obtener_embedding_juez
            juez = juez if juez is not None else obtener_embedding_juez()
            base = base if base is not None else obtener_base_semantica(juez)
        from .metricas import recalibrar_similitud, vector_normalizado
        cache: dict = {}
        textos_pag = [(getattr(d, "page_content", "") or "") for d in documentos]
        norms_pag = [_normalizar_para_busqueda(t) for t in textos_pag]
        v_pags = [vector_normalizado(t[:2000], juez, cache) for t in textos_pag]
        for q, inf in zip(preguntas, informe):
            cita = (q.get("source_context") or "")
            cita_norm = _normalizar_para_busqueda(cita)
            grams_cita = gramas_de(cita)
            v_q = vector_normalizado(q.get("question", ""), juez, cache)
            v_c = vector_normalizado(cita, juez, cache)
            sim_cita = recalibrar_similitud(float(v_q @ v_c), base)
            # Páginas AJENAS: las que no contienen la cita (literal o >=50% grams).
            ajenas = [recalibrar_similitud(float(v_q @ v_p), base)
                      for v_p, t_norm, t_crudo in zip(v_pags, norms_pag, textos_pag)
                      if cita_norm not in t_norm
                      and cobertura(gramas_de(t_crudo), grams_cita) < 0.5]
            mejor_ajena = max(ajenas) if ajenas else 0.0
            discrimina = sim_cita >= mejor_ajena
            inf["capas"]["discrimina"] = {"ok": discrimina,
                                          "sim_cita": round(sim_cita, 3),
                                          "mejor_pagina_ajena": round(mejor_ajena, 3)}
            if not discrimina and inf["veredicto"] == "ok":
                inf["veredicto"] = "descartada"
                inf["motivo"] = (f"no discrimina: la pregunta es más afín a otra "
                                 f"página ({mejor_ajena:.2f}) que a su cita "
                                 f"({sim_cita:.2f})")
    except Exception as e:  # noqa: BLE001
        print(f"[sesion] Capa de discriminación omitida ({e}).")
        for inf in informe:
            inf["capas"]["discrimina"] = None

    # ---- Capa 2: round-trip QA con el juez (solo nivel 'completa') -------- #
    if nivel == "completa":
        vivos = [i for i, inf in enumerate(informe) if inf["veredicto"] == "ok"]
        try:
            fn = verificador_qa or _verificador_qa_por_defecto
            items = [{"indice": i, **{k: preguntas[i].get(k, "") for k in
                      ("question", "expected_answer", "source_context")}}
                     for i in vivos]
            if items:
                veredictos = {v["indice"]: v for v in fn(items)}
                for i in vivos:
                    v = veredictos.get(i)
                    if v is None:
                        informe[i]["capas"]["qa"] = None   # sin veredicto: no castigar
                        continue
                    informe[i]["capas"]["qa"] = {"respondible": v["respondible"],
                                                 "coincide": v["coincide"]}
                    if not (v["respondible"] and v["coincide"]):
                        informe[i]["veredicto"] = "descartada"
                        informe[i]["motivo"] = (
                            "la cita no responde a la pregunta" if not v["respondible"]
                            else "la respuesta esperada no se sostiene en la cita")
        except Exception as e:  # noqa: BLE001
            print(f"[sesion] Capa round-trip QA omitida ({e}).")
            for i in vivos:
                informe[i]["capas"]["qa"] = None

    # ---- Cierre: guardarraíl y resumen ------------------------------------ #
    validas = [q for q, inf in zip(preguntas, informe) if inf["veredicto"] == "ok"]
    for inf in informe:
        if inf["veredicto"] == "descartada":
            print(f"[sesion] Pregunta descartada ({inf['motivo']}): "
                  f"{inf['question'][:70]!r}")
    tasa = round(len(validas) / len(preguntas), 3)

    if len(validas) < minimo:
        print(f"[sesion] AVISO: la verificación dejaría {len(validas)}/"
              f"{len(preguntas)} preguntas (<{minimo}). CONSERVO todas y marco el "
              f"examen como poco fiable: con tasa {tasa:.0%}, la palanca correcta "
              f"es subir CONFIG.modelo_clasificador o regenerar, no medir con "
              f"{len(validas)} preguntas.")
        return {"validas": preguntas, "informe": informe, "tasa": tasa,
                "guardarrail": True}

    print(f"[sesion] Verificación de preguntas ({nivel}): {len(validas)}/"
          f"{len(preguntas)} superan todas las capas (tasa {tasa:.0%}).")
    return {"validas": validas, "informe": informe, "tasa": tasa}


def nuevo_run_id() -> str:
    """
    Identificador único de sesión: marca de tiempo legible + sufijo aleatorio.

    El timestamp a segundos colisionaba si se procesaban dos documentos en el
    mismo segundo (en bucles batch/Colab), y como el run_id es clave de
    diccionario en el meta-supervisor, el segundo pisaba al primero. El sufijo
    uuid lo hace único sin perder legibilidad cronológica.
    """
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]


def hash_pdf(pdf_path: str) -> str | None:
    """
    SHA-256 del CONTENIDO del fichero, para identificar un documento de forma
    robusta: independiente de la ruta y de la máquina (mismo PDF en el Mac y en
    Colab/Drive da el mismo hash). Es la clave con la que la memoria sabe si un
    documento ya se estudió. Devuelve None si el fichero no se puede leer
    (entonces no hay deduplicación por contenido y se cae al nombre del fichero).
    """
    try:
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Modelos Pydantic: dan forma a lo que el LLM debe devolver
# --------------------------------------------------------------------------- #
class PreguntaTest(BaseModel):
    question: str
    expected_answer: str
    source_context: str  # cita literal del texto, para poder medir la recuperación


class ListaPreguntasTest(BaseModel):
    questions: list[PreguntaTest]


# --------------------------------------------------------------------------- #
# La sesión como objeto: agrupa todo lo que se comparte
# --------------------------------------------------------------------------- #
@dataclass
class Sesion:
    """Contenedor de todo el contexto compartido entre bancos."""

    run_id: str
    pdf_path: str
    doc_type: str
    metadatos: dict
    perfil: dict = None  # perfil deterministico (perfil.py); se completa con LDA/UMAP
    restricciones: dict = None  # restricciones del plan por banco (fijar/familias/etiquetas)
    preguntas_test: list[dict] = None
    documentos: list[Any] = field(default_factory=list)
    # Hash de contenido del PDF: identifica el documento para la deduplicación
    # (no reentrenar con algo ya estudiado). Vacío si no se pudo calcular.
    doc_hash: str = ""
    # Informe de verificar_preguntas sobre el examen de la sesión (capas, tasa).
    verificacion: dict = field(default_factory=dict)

    # Configuraciones que un banco hereda del anterior. Empiezan vacías y el
    # meta-supervisor las va rellenando a medida que cada banco decide.
    config_splitter: dict = field(default_factory=dict)
    config_embedding: dict = field(default_factory=dict)
    config_retriever: dict = field(default_factory=dict)


def _muestrear_texto(documentos: list[Any], n_paginas: int = 5,
                    chars_por_pagina: int = 800) -> str:
    """
    Toma una muestra representativa del documento.

    En vez de leer las primeras páginas (que suelen ser portada e índice),
    repartimos la muestra a lo largo de todo el documento.
    """
    total = len(documentos)
    if total == 0:
        return ""
    paso = max(1, total // n_paginas)
    indices = list(range(0, total, paso))[:n_paginas]
    return "\n\n".join(documentos[i].page_content[:chars_por_pagina] for i in indices)


def preparar_sesion(pdf_path: str,
                    preguntas_predefinidas: list[dict] | None = None) -> Sesion:
    """
    Carga el PDF, lo CARACTERIZA (perfil deterministico, sin LLM) y genera las
    preguntas de test (unica tarea del LLM aqui).

    `preguntas_predefinidas`: examen CONGELADO de una corrida anterior del mismo
    documento (mismo doc_hash). Si llega, se usa tal cual —sin regenerar ni
    reverificar— para que corridas repetidas midan contra el MISMO examen:
    comparar contra exámenes distintos mete ruido entre corridas.

    El numero de preguntas de test es una constante de medicion de `CONFIG`
    (`n_preguntas_test`), no una decision del plan.
    """
    # La caché de vectores del árbitro es global y se indexa por TEXTO: entre
    # documentos no se reutiliza nada (los textos cambian) pero sí crece sin
    # límite durante un batch (fuga de memoria real en tandas largas). Documento
    # nuevo => caché limpia. (La calibración del suelo es POR ÁRBITRO y se
    # conserva: esa sí se reutiliza entre documentos.)
    from .evaluacion import limpiar_cache_juez
    limpiar_cache_juez()

    n_preguntas = max(3, min(20, int(CONFIG.n_preguntas_test)))
    run_id = nuevo_run_id()
    print(f"[sesion] Iniciando sesión {run_id}")

    documentos = PyPDFLoader(pdf_path).load()
    print(f"[sesion] PDF cargado: {len(documentos)} páginas")

    # Perfil deterministico (linguistica computacional): sustituye a la
    # clasificacion por LLM. LDA/UMAP se rellenan despues, en el meta.
    perfil = perfil_de_documento(documentos)
    print(f"[sesion] Perfil: {descriptor(perfil)} "
          f"(densidad={perfil['densidad_textual']}, tablas={perfil['prop_tablas']}, "
          f"lex={perfil['repetitividad_lexica']})")

    if preguntas_predefinidas:
        print(f"[sesion] Examen congelado reutilizado: "
              f"{len(preguntas_predefinidas)} preguntas (no se regenera).")
        return Sesion(
            run_id=run_id,
            pdf_path=pdf_path,
            doc_type=descriptor(perfil),
            metadatos=perfil,
            perfil=perfil,
            preguntas_test=list(preguntas_predefinidas),
            documentos=documentos,
            doc_hash=hash_pdf(pdf_path) or "",
            verificacion={"congelado": True},
        )

    muestra = _muestrear_texto(documentos)

    # El LLM SOLO genera preguntas (with_structured_output evita parseo fragil).
    llm = construir_llm_clasificador().with_structured_output(ListaPreguntasTest)
    instruccion = (
        "Eres un experto en control de calidad de sistemas RAG. Analiza el texto "
        f"y genera exactamente {n_preguntas} preguntas de test complejas.\n\n"
        "Reglas para las preguntas:\n"
        "1. 'source_context' debe ser una cita LITERAL del texto, de al menos 50 palabras.\n"
        "2. No parafrasees: copia el fragmento exacto tal como aparece.\n"
        "3. La pregunta debe poder responderse usando solo ese fragmento.\n"
        "4. 'expected_answer' puede estar en español."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", instruccion),
        ("human", "Texto del documento:\n\n{texto}"),
    ])
    lista: ListaPreguntasTest = (prompt | llm).invoke({"texto": muestra})

    preguntas_test = [q.model_dump() for q in lista.questions]
    print(f"[sesion] {len(preguntas_test)} preguntas de test generadas")
    preguntas_test = validar_citas(preguntas_test, documentos)          # capa 1
    verif = verificar_preguntas(preguntas_test, documentos)             # capas 3/4/2
    preguntas_test = verif["validas"]

    return Sesion(
        run_id=run_id,
        pdf_path=pdf_path,
        doc_type=descriptor(perfil),   # etiqueta legible SOLO para logs/seeds
        metadatos=perfil,
        perfil=perfil,
        preguntas_test=preguntas_test,
        documentos=documentos,
        doc_hash=hash_pdf(pdf_path) or "",
        verificacion={"tasa": verif["tasa"], "informe": verif["informe"],
                      "guardarrail": verif.get("guardarrail", False)},
    )


def generar_preguntas_test(sesion: "Sesion", n: int,
                           evitar: list[str] | None = None) -> list[dict]:
    """
    Genera `n` preguntas de test ADICIONALES para un documento ya cargado.

    Es la palanca de preguntas (Fase A.2): cuando un banco queda en disputa por IC
    ancho (`margen_no_concluyente`), el meta-supervisor pide mas preguntas para
    estrechar el IC. Muestrea mas ancho que la clasificacion y pide al LLM
    preguntas NUEVAS (no reformulaciones de las existentes). Guardado: si algo
    falla, devuelve [] y el meta degrada a parar (no rompe la corrida).
    """
    if not getattr(sesion, "documentos", None) or n <= 0:
        return []
    muestra = _muestrear_texto(sesion.documentos, n_paginas=8, chars_por_pagina=900)
    evitar = evitar or [q.get("question", "") for q in sesion.preguntas_test]
    evitar_txt = "\n".join(f"- {q}" for q in evitar[:30]) or "(ninguna)"
    instruccion = (
        "Eres un experto en control de calidad de sistemas RAG. Genera exactamente "
        f"{n} preguntas de test complejas y NUEVAS sobre el texto.\n\n"
        "Reglas:\n"
        "1. 'source_context' debe ser una cita LITERAL del texto, de al menos 50 palabras.\n"
        "2. No parafrasees: copia el fragmento exacto tal como aparece.\n"
        "3. La pregunta debe poder responderse usando solo ese fragmento.\n"
        "4. 'expected_answer' puede estar en español.\n"
        "5. NO repitas ni reformules ninguna de estas preguntas ya existentes:\n"
        f"{evitar_txt}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", instruccion),
        ("human", "Texto del documento:\n\n{texto}"),
    ])
    try:
        llm = construir_llm_clasificador().with_structured_output(ListaPreguntasTest)
        res: ListaPreguntasTest = (prompt | llm).invoke({"texto": muestra})
        nuevas = [q.model_dump() for q in res.questions][:n]
        nuevas = validar_citas(nuevas, sesion.documentos, minimo=0)     # capa 1
        # Mismas capas que el examen inicial; minimo=0 porque aquí un lote vacío
        # es aceptable (el examen base ya existe; esto solo lo amplía).
        nuevas = verificar_preguntas(nuevas, sesion.documentos,
                                     minimo=0)["validas"]
        print(f"[sesion] Generadas {len(nuevas)} preguntas de test adicionales "
              f"(verificadas: citas + autocontención + discriminación + QA).")
        return nuevas
    except Exception as e:  # noqa: BLE001
        print(f"[sesion] No se pudieron generar preguntas extra: {e}")
        return []
