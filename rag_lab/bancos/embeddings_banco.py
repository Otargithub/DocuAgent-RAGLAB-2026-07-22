"""
Banco de embeddings: ¿qué modelo de embeddings representa mejor este documento?

Aquí la variable es el modelo de embeddings. Para que la comparación sea justa,
todos los modelos trocean el documento con el MISMO splitter (el que heredamos
del banco de splitters) y se evalúan con el mismo retriever neutro.

El catálogo de modelos vive en EMBEDDING_REGISTRY: para probar uno nuevo, basta
con añadir una entrada ahí.
"""

from __future__ import annotations

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from .cache_documento import por_documento, huella_troceado
from ..config import CONFIG
from ..corpus import Sesion
from ..evaluacion import evaluar_sobre_preguntas, relacion_con_arbitro
from ..fabricas import construir_embedding
from .base import BancoBase

EMBEDDING_REGISTRY: dict[str, dict[str, Any]] = {
    "text-embedding-3-small": {
        "provider": "openai", "model": "text-embedding-3-small",
        "max_tokens": 8191, "cost_per_1k_tokens": 0.00002,
        "notes": "Baseline OpenAI. Buen balance coste/calidad.",
    },
    "text-embedding-3-large": {
        "provider": "openai", "model": "text-embedding-3-large",
        "max_tokens": 8191, "cost_per_1k_tokens": 0.00013,
        "notes": "Máxima calidad OpenAI, bastante más caro.",
    },
    "bge-small-en-v1.5": {
        "provider": "huggingface", "model": "BAAI/bge-small-en-v1.5",
        "max_tokens": 512, "cost_per_1k_tokens": 0.0,
        "notes": "Local, fuerte en inglés técnico.",
    },
    "bge-m3": {
        "provider": "huggingface", "model": "BAAI/bge-m3",
        "max_tokens": 8192, "cost_per_1k_tokens": 0.0,
        "notes": "Local y multilingüe. Más lento pero potente ES-EN.",
    },
    "voyage-3-lite": {
        "provider": "voyage", "model": "voyage-3-lite",
        "max_tokens": 32000, "cost_per_1k_tokens": 0.000016,
        "notes": "Voyage rápido y barato.",
    },
    "qwen3-0.6b": {
        "provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B",
        "max_tokens": 32000, "cost_per_1k_tokens": 0.0,
        "notes": "Local, multilingüe (100+ idiomas), familia #1 de MTEB "
                 "multilingüe 2026. Rinde aún más con prompt de instrucción "
                 "en queries (prompt_name='query' en sentence-transformers).",
    },
    "granite-311m-r2": {
        "provider": "huggingface",
        "model": "ibm-granite/granite-embedding-311m-multilingual-r2",
        "max_tokens": 32768, "cost_per_1k_tokens": 0.0,
        "notes": "Local, ModernBERT, 200+ idiomas, 768 dims. 65.2 en MTEB "
                 "Multilingual Retrieval (#2 abierto <500M). Apache 2.0.",
    },
    "granite-97m-r2": {
        "provider": "huggingface",
        "model": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "max_tokens": 32768, "cost_per_1k_tokens": 0.0,
        "notes": "Local y MUY rápido (97M, 384 dims). Mejor abierto sub-100M "
                 "en MTEB Multilingual Retrieval (60.3). Apache 2.0.",
    },
}

_SPLITTERS = {
    "recursive": RecursiveCharacterTextSplitter,
    "token": TokenTextSplitter,
    "character": CharacterTextSplitter,
}


# Embedding base para bancos que necesitan uno FIJO en exploracion (metadata,
# retrievers, rerankers). Local y gratis: medir sobre un modelo de pago que
# quiza no despliegues sesga el veredicto y cuesta dinero. granite-311m es el
# local solido por defecto; si el documento ya tiene embedding ganador en
# memoria, los bancos lo prefieren a este (ver preparar_contexto).
def embedding_base_exploratorio() -> dict:
    e = EMBEDDING_REGISTRY.get("granite-311m-r2")
    return {"model": e["model"], "provider": e["provider"]} if e else \
        {"model": "text-embedding-3-small", "provider": "openai"}


def _construir_splitter(config: dict):
    """
    Reconstruye un splitter desde su config (claves del banco o heredadas).

    DELEGA en el builder del banco de splitters para garantizar UNA sola
    semantica en todo el sistema (tamaños en CARACTERES; soporte de page y
    semantic). Sin esta delegacion, un finalista 'token_1000' (1000 caracteres,
    ~250 tokens) se reconstruia aqui como TokenTextSplitter(1000 TOKENS, ~4000
    caracteres): la Fase B y cualquier pipeline reconstruido median un artefacto
    DISTINTO del que gano en el banco; y 'page'/'semantic' caian al default
    recursive sin aviso.
    """
    from .splitters import _construir_splitter_banco  # import local: evita ciclos
    nombre = config.get("name") or config.get("splitter", "recursive")
    modelo = None
    if nombre == "semantic":
        modelo = construir_embedding(CONFIG.embedding_neutro)
    return _construir_splitter_banco(
        nombre,
        int(config.get("chunk_size", 1000) or 0),
        int(config.get("chunk_overlap", config.get("overlap", 100)) or 0),
        modelo)


class BancoEmbeddings(BancoBase):
    nombre = "embedding"
    clave_decision = "embedding_recomendado"
    descripcion = "modelos de embeddings"
    esquema_json = (
        "{\n"
        '  "embedding_recomendado": "<clave del registro>",\n'
        '  "provider": "<openai|huggingface|voyage>",\n'
        '  "confianza": "<alta|media|baja>",\n'
        '  "justificacion": "<2-3 frases; menciona la relación calidad/coste>",\n'
        '  "alertas": ["<aviso si hay algo preocupante>"]\n'
        "}"
    )
    # Primaria: MRR (calidad de representación = lo relevante sale arriba).
    # Secundarios: coste y latencia.
    metrica_primaria = ("mrr", "max")
    # Calidad antes que coste: el coste ya tiene su via (alternativa economica)
    objetivos_secundarios = [("recuperacion_pasaje", "max"), ("relevancia_media", "max"),
                             ("coste_1k", "min"), ("latency", "min")]
    metricas_contexto = ("relevancia_media", "recuperacion_pasaje")

    def preparar_contexto(self, sesion: Sesion) -> dict:
        """Troceamos UNA vez con el splitter heredado; lo reutilizan todos los modelos."""
        splitter = _construir_splitter(sesion.config_splitter)
        chunks = splitter.split_documents(sesion.documentos)
        print(f"[embedding] Corpus troceado en {len(chunks)} fragmentos "
              f"(splitter base: {sesion.config_splitter.get('name', 'recursive')})")
        return {"chunks": chunks}

    def _tendencias(self, resultados: list[dict],
                    ganador: dict | None = None) -> list[str]:
        met = self.metrica_primaria[0]
        vals = sorted((float(r[met]) for r in resultados
                       if r.get(met) is not None), reverse=True)
        if len(vals) < 2:
            return []
        lineas = [f"separacion: mejor={vals[0]:.3f} peor={vals[-1]:.3f} "
                  f"rango={vals[0] - vals[-1]:.3f} | margen 1º-2º="
                  f"{vals[0] - vals[1]:.3f}"]
        if vals[0] < 0.35:
            lineas.append(f"NIVEL BAJO GENERALIZADO (mejor {met}={vals[0]:.3f}): "
                          f"ningun embedding rinde sobre el pipeline base — "
                          f"sospecha del TROCEO base desalineado con el documento "
                          f"(este veredicto de banco es poco fiable; la Fase B "
                          f"con el splitter ganador tendra la palabra)")
        return lineas

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        """
        Parrilla de candidatos, aplicando el guardarraíl de independencia del
        árbitro: un modelo IDÉNTICO al árbitro se autoevaluaría (se EXCLUYE); uno
        de su misma familia representacional arrastra sesgo de afinidad (se AVISA
        pero se conserva, porque descartar familias enteras podría vaciar la
        parrilla). Ver evaluacion.relacion_con_arbitro.
        """
        experimentos = []
        for clave, cfg in EMBEDDING_REGISTRY.items():
            rel = relacion_con_arbitro(cfg)
            if rel == "identico":
                print(f"[embedding] EXCLUIDO '{clave}': es el propio árbitro "
                      f"semántico; evaluarlo con su propio espacio sesgaría la "
                      f"comparación. (Cambia CONFIG.embedding_juez para incluirlo.)")
                continue
            if rel == "familia":
                print(f"[embedding] AVISO '{clave}': misma familia que el árbitro; "
                      f"la nota semántica puede favorecerlo por afinidad. Interpreta "
                      f"su ventaja con cautela o usa un árbitro de otra familia.")
            experimentos.append({"etiqueta": clave, "clave": clave, **cfg})
        return experimentos

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        import time

        chunks = contexto["chunks"]
        config = {"model": experimento["model"], "provider": experimento["provider"]}

        modelo = construir_embedding(config)

        def _fabricar():
            # La latencia (metrica secundaria = tiempo de embebido) se mide UNA
            # vez, al construir; los runs siguientes reportan ese mismo valor:
            # es la misma operacion sobre los mismos chunks, y repetir la
            # medicion solo anadiria ruido entre runs.
            t0 = time.time()
            vs = FAISS.from_documents(chunks, modelo)
            return vs, round(time.time() - t0, 4)

        vectorstore, latencia = por_documento(
            "embedding", sesion.run_id,
            (huella_troceado(chunks), experimento["model"]), _fabricar)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        metricas = evaluar_sobre_preguntas(
            lambda consulta: retriever.invoke(consulta),
            sesion.preguntas_test,
            modelo,
        )
        metricas["coste_1k"] = experimento.get("cost_per_1k_tokens", 0.0)
        metricas["latency"] = latencia
        return metricas
