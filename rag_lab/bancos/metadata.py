"""
Banco de metadata: Unified Embeddings (fusion de contenido y metadatos).

La idea que evalúa este banco: en lugar de indexar solo el contenido de cada
fragmento, fusionar su embedding de contenido con el embedding de sus metadatos
estructurales (tipo de documento, página, título de sección, posición...):

    e_unificado = alpha * e_contenido + (1 - alpha) * e_metadatos

La variable que probamos es doble: el peso `alpha` (cuánto pesa el contenido
frente a los metadatos) y qué campos de metadatos incluimos. Con alpha = 1.0
tenemos el baseline (contenido puro, sin metadatos); valores menores mezclan más
contexto estructural.

Reutiliza todo lo del laboratorio (métricas, fábricas, evaluación). Lo único
propio de este banco es el motor de fusión de embeddings y la calidad del
espacio vectorial; eso vive aquí porque no lo usa ningún otro banco.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import faiss as faiss_lib
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..corpus import Sesion
from ..evaluacion import evaluar_sobre_preguntas
from ..fabricas import construir_embedding
from .cache_documento import huella_troceado, por_documento
from .base import BancoBase
from .embeddings_banco import _construir_splitter

K_EVAL = 5
MAX_FRAGMENTOS = 200

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

CONFIGS_METADATOS = [
    {"name": "baseline", "fields": [],
     "desc": "Sin metadatos: vector de contenido puro."},
    {"name": "structural", "fields": ["doc_type", "page_number"],
     "desc": "Metadatos estructurales globales."},
    {"name": "positional", "fields": ["chunk_index", "position_ratio"],
     "desc": "Posición del fragmento dentro del documento."},
    {"name": "section", "fields": ["section_title"],
     "desc": "Título de sección (señal semántica fuerte)."},
    {"name": "full", "fields": ["doc_type", "page_number", "section_title",
                                "chunk_index", "position_ratio"],
     "desc": "Todos los campos: máximo contexto."},
]


def enriquecer_con_metadatos(chunks: list[Any], doc_type: str) -> list[Any]:
    """
    Añade a cada fragmento los metadatos estructurales que luego podremos fundir:
    tipo de documento, página, índice, posición relativa y título de sección.
    """
    total = len(chunks)
    enriquecidos = []
    for i, chunk in enumerate(chunks):
        nuevo = Document(page_content=chunk.page_content,
                         metadata=copy.deepcopy(chunk.metadata))
        primera_linea = chunk.page_content.split("\n")[0].strip()
        titulo = primera_linea[:80] if 10 < len(primera_linea) < 100 else ""
        nuevo.metadata.update({
            "doc_type": doc_type,
            "page_number": chunk.metadata.get("page", i // max(1, total // 10)),
            "chunk_index": i,
            "total_chunks": total,
            "position_ratio": round(i / max(total - 1, 1), 3),
            "section_title": titulo,
        })
        enriquecidos.append(nuevo)
    return enriquecidos


def _texto_metadatos(chunk: Any, campos: list[str]) -> str:
    """Convierte los metadatos activos de un fragmento en texto embebible."""
    m = chunk.metadata
    plantillas = {
        "doc_type": lambda: f"Document type: {m.get('doc_type', '')}",
        "page_number": lambda: f"Page {m.get('page_number', '')}",
        "section_title": lambda: (f"Section: {m.get('section_title', '')}"
                                  if m.get("section_title") else ""),
        "chunk_index": lambda: f"Chunk {m.get('chunk_index', '')} of {m.get('total_chunks', '')}",
        "position_ratio": lambda: f"Position in document: {m.get('position_ratio', '')}",
    }
    partes = [plantillas[c]() for c in campos if c in plantillas and plantillas[c]()]
    return ". ".join(partes) + "." if partes else ""


def calcular_vectores_unificados(chunks, modelo, alpha, campos) -> np.ndarray:
    """Calcula los vectores fusionados y normalizados."""
    e_contenido = np.array(modelo.embed_documents([c.page_content for c in chunks]),
                           dtype=np.float32)
    if alpha >= 1.0 or not campos:
        normas = np.linalg.norm(e_contenido, axis=1, keepdims=True)
        return e_contenido / (normas + 1e-10)

    textos_meta = [_texto_metadatos(c, campos) for c in chunks]
    seguros = [t if t.strip() else "no metadata" for t in textos_meta]
    e_meta = np.array(modelo.embed_documents(seguros), dtype=np.float32)
    for i, t in enumerate(textos_meta):
        if not t.strip():
            e_meta[i] = 0.0

    fusion = alpha * e_contenido + (1 - alpha) * e_meta
    normas = np.linalg.norm(fusion, axis=1, keepdims=True)
    return fusion / (normas + 1e-10)


def _vectores_unificados_cacheados(chunks, modelo, alpha, campos,
                                   contexto: dict) -> np.ndarray:
    """Como calcular_vectores_unificados, pero sin re-embeber: el contenido viene
    del contexto y los metadatos se memoizan por variante de campos."""
    e_contenido = contexto["e_contenido"]
    if alpha >= 1.0 or not campos:
        normas = np.linalg.norm(e_contenido, axis=1, keepdims=True)
        return e_contenido / (normas + 1e-10)
    clave = tuple(campos)
    cache = contexto["e_meta_cache"]
    if clave not in cache:
        textos_meta = [_texto_metadatos(c, campos) for c in chunks]
        seguros = [t if t.strip() else "no metadata" for t in textos_meta]
        e_meta = np.array(modelo.embed_documents(seguros), dtype=np.float32)
        for i, t in enumerate(textos_meta):
            if not t.strip():
                e_meta[i] = 0.0
        cache[clave] = e_meta
    fusion = alpha * e_contenido + (1 - alpha) * cache[clave]
    normas = np.linalg.norm(fusion, axis=1, keepdims=True)
    return fusion / (normas + 1e-10)


def _faiss_desde_vectores(chunks, vectores, modelo) -> FAISS:
    """Construye un índice FAISS a partir de vectores ya calculados."""
    indice = faiss_lib.IndexFlatL2(vectores.shape[1])
    indice.add(vectores.astype("float32"))
    docstore = InMemoryDocstore({str(i): chunks[i] for i in range(len(chunks))})
    return FAISS(embedding_function=modelo, index=indice, docstore=docstore,
                 index_to_docstore_id={i: str(i) for i in range(len(chunks))})


def calidad_espacio(vectores: np.ndarray, chunks: list[Any]) -> dict:
    """Mide la calidad del espacio vectorial resultante."""
    if len(vectores) < 4:
        return {"intra_cohesion": 0.0, "inter_separation": 0.0, "rango_similitud": 0.0}

    n = min(50, len(vectores))
    # Muestreo DETERMINISTA: era el único punto no sembrado del proyecto; con
    # np.random.choice sin semilla la métrica cambiaba entre ejecuciones sobre
    # los mismos datos (irreproducible y ruido espurio entre runs).
    rng = np.random.default_rng(len(vectores) * 7919 + 12345)
    idx = rng.choice(len(vectores), n, replace=False)
    muestra = vectores[idx]
    paginas = [chunks[i].metadata.get("page_number", 0) for i in idx]

    normas = np.linalg.norm(muestra, axis=1, keepdims=True)
    norm = muestra / (normas + 1e-10)
    sim = norm @ norm.T

    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if paginas[i] == paginas[j] else inter).append(float(sim[i, j]))

    todas = sim[np.triu_indices(n, k=1)].tolist()
    gap = (round(float(np.percentile(todas, 90) - np.percentile(todas, 10)), 4)
           if todas else 0.0)
    return {
        "intra_cohesion": round(float(np.mean(intra)), 4) if intra else 0.0,
        "inter_separation": round(float(np.mean(inter)), 4) if inter else 0.0,
        "rango_similitud": gap,
    }


class BancoMetadata(BancoBase):
    nombre = "metadata"
    clave_decision = "config_recomendada"
    descripcion = (
        "estrategias Unified Embeddings (fusión de contenido y metadatos). "
        "Principios: una mejora de MRR > 0.05 justifica el coste de embeber dos "
        "veces; si la mejora es <= 0.02, conviene el baseline (alpha=1.0). En "
        "documentos técnicos el título de sección suele ser el campo más útil; en "
        "documentos legales, el tipo y la página; un rango_similitud alto indica que "
        "los metadatos mejoran la separación del espacio"
    )
    esquema_json = (
        "{\n"
        '  "config_recomendada": "<baseline|structural|positional|section|full>",\n'
        '  "alpha_recomendado": <float entre 0.5 y 1.0>,\n'
        '  "fields_recomendados": ["<campos>"],\n'
        '  "mejora_mrr_vs_baseline": <float>,\n'
        '  "confianza": "<alta|media|baja>",\n'
        '  "justificacion": "<2-3 frases basadas en los datos>",\n'
        '  "alertas": ["<aviso si hay algo preocupante>"]\n'
        "}"
    )
    # Primaria: MRR. Secundario: coste de embeber dos veces (embed_time).
    # rango_similitud (antes 'relevant_gap' — nombre enganoso: no mide
    # relevancia, mide la APERTURA p90-p10 del rango de similitudes del
    # espacio) NO decide hasta que su valor como criterio se justifique
    # empiricamente (decision de Xavier): se calcula y se comunica como
    # contexto; hay experimento offline pendiente gap<->mrr.
    metrica_primaria = ("mrr", "max")
    objetivos_secundarios = [("embed_time", "min")]
    metricas_contexto = ("rango_similitud", "relevancia_media", "recuperacion_pasaje")

    def preparar_contexto(self, sesion: Sesion) -> dict:
        """Trocea con el splitter heredado, enriquece y prepara el modelo (una vez)."""
        splitter = _construir_splitter(sesion.config_splitter)
        chunks = splitter.split_documents(sesion.documentos)[:MAX_FRAGMENTOS]
        chunks = enriquecer_con_metadatos(chunks, sesion.doc_type)

        # Base del enriquecimiento, por orden de preferencia:
        #  1) el embedding YA FIJADO en la sesion (ganador coronado) — mide el
        #     metadata sobre el pipeline REAL, no sobre uno ajeno;
        #  2) en exploracion pura (sin ganador aun), un default de laboratorio
        #     LOCAL y gratis, no el OpenAI de pago que quiza no despliegues.
        # (Observacion de Xavier: enriquecer con un embedding distinto al del
        #  pipeline responde a otra pregunta y puede no transferir.)
        from .embeddings_banco import embedding_base_exploratorio
        config_emb = sesion.config_embedding or embedding_base_exploratorio()
        sobre_ganador = bool(sesion.config_embedding)
        modelo = construir_embedding(config_emb)
        # CACHE (eficacia): e_contenido no depende del experimento (mismos chunks,
        # mismo modelo) => se embebe UNA vez aqui, no 26 veces (una por config).
        # e_meta solo depende de `fields` (3-4 variantes) => memoizada por campos.
        # Cada experimento queda en mezcla numpy (gratis) + montar FAISS (ms).
        # POR DOCUMENTO, no por run: la instancia del banco es nueva en cada
        # run y este embed de todos los chunks (lo caro del banco) se pagaba
        # una vez POR RUN. Los chunks y el modelo no cambian entre runs.
        clave = (huella_troceado(chunks), config_emb.get("model"))
        e_contenido = por_documento(
            "metadata_contenido", sesion.run_id, clave,
            lambda: np.array(modelo.embed_documents(
                [c.page_content for c in chunks]), dtype=np.float32))
        # Aviso de DISCREPANCIA (peticion de Xavier): declarar SIEMPRE si el
        # embedding base coincide con el ganador conocido del documento — medir
        # el metadata sobre un embedding distinto al del pipeline responde a
        # otra pregunta y puede no transferir.
        ganador = None
        try:
            from ..corpus import hash_pdf
            from ..experimento_dirigido import _embedding_ganador
            from ..memory import Memoria
            dh = getattr(sesion, "doc_hash", None)
            if dh:
                ganador = _embedding_ganador(Memoria(), dh)
        except Exception:  # noqa: BLE001
            ganador = None
        base_model = config_emb.get("model")
        if sobre_ganador:
            etiqueta = "GANADOR del pipeline"
        elif ganador and ganador.get("model") and ganador["model"] != base_model:
            etiqueta = f"⚠ NO coincide con el ganador del documento ({ganador['model']})"
        elif ganador and ganador.get("model") == base_model:
            etiqueta = "coincide con el ganador del documento"
        else:
            etiqueta = "exploratorio (sin ganador coronado aun)"
        print(f"[metadata] {len(chunks)} fragmentos enriquecidos | "
              f"embedding base: {base_model} [{etiqueta}] (contenido cacheado)")
        return {"chunks": chunks, "modelo": modelo, "e_contenido": e_contenido,
                # dict COMPARTIDO entre runs vía la caché por documento: las
                # variantes de e_meta (3-4 juegos de campos) también son
                # idénticas entre runs.
                "e_meta_cache": por_documento("metadata_meta", sesion.run_id,
                                              clave, dict)}

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        experimentos = []
        for alpha in ALPHAS:
            for cfg in CONFIGS_METADATOS:
                if alpha >= 1.0 and cfg["name"] != "baseline":
                    continue
                experimentos.append({
                    "etiqueta": f"a{alpha:.1f}_{cfg['name']}",
                    "alpha": alpha,
                    "config": cfg["name"],
                    "fields": cfg["fields"],
                })
        return experimentos

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        chunks = contexto["chunks"]
        modelo = contexto["modelo"]
        alpha = experimento["alpha"]
        campos = experimento["fields"]

        t0 = time.time()
        vectores = _vectores_unificados_cacheados(chunks, modelo, alpha, campos, contexto)
        vectorstore = _faiss_desde_vectores(chunks, vectores, modelo)
        retriever = vectorstore.as_retriever(search_kwargs={"k": K_EVAL})
        tiempo = round(time.time() - t0, 3)

        metricas = evaluar_sobre_preguntas(
            lambda consulta: retriever.invoke(consulta),
            sesion.preguntas_test,
            modelo,
        )
        metricas.update(calidad_espacio(vectores, chunks))
        metricas["embed_time"] = tiempo
        return metricas
