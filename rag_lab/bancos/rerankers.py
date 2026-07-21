"""
Banco de rerankers: una vez recuperados los candidatos, qué reordenador los
ordena mejor para el LLM final.

La variable es el reranker. Todo lo anterior (splitter, embedding, retriever) se
hereda y se fija. Optimización: como la configuración de recuperación es la misma
para todos los rerankers, recuperamos los candidatos UNA vez en preparar_contexto
y cada reranker solo reordena esa lista.

Métrica propia: rerank_gain, la mejora de MRR frente a no reordenar nada.

Puntuación híbrida: igual que el resto del laboratorio, la nota combina exactitud
léxica y similitud semántica con un ÁRBITRO FIJO (no el embedding heredado), de
modo que comparar rerankers sea justo. Los n-gramas y los vectores árbitro de los
pasajes de test se precomputan una sola vez; los embeddings de los candidatos se
cachean (los candidatos son los mismos para todos los rerankers).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS

from ..config import CONFIG
from ..corpus import Sesion
from ..evaluacion import obtener_base_semantica, obtener_embedding_juez
from ..fabricas import construir_embedding
from .cache_documento import huella_troceado, por_documento
from ..metricas import (
    CV_SIN_SENAL,
    PESO_EXACTITUD,
    evaluar_recuperacion,
    gramas_de,
    similitudes_semanticas,
    vector_normalizado,
)
from ..recuperacion import construir_recuperador
from .base import BancoBase
from .embeddings_banco import _construir_splitter

K_CANDIDATOS = 20
K_FINAL = 5

RERANKER_REGISTRY: dict[str, dict[str, Any]] = {
    "no_reranker": {
        "provider": "none", "model": "none", "cost_per_1k_queries": 0.0,
        "notes": "Baseline: el retriever puro, sin reordenar. Referencia del rerank_gain.",
    },
    "ms-marco-MiniLM-L-6-v2": {
        "provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cost_per_1k_queries": 0.0, "notes": "Cross-encoder ligero, buena latencia.",
    },
    "ms-marco-MiniLM-L-12-v2": {
        "provider": "huggingface", "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "cost_per_1k_queries": 0.0, "notes": "Cross-encoder mediano, más calidad.",
    },
    "bge-reranker-base": {
        "provider": "huggingface", "model": "BAAI/bge-reranker-base",
        "cost_per_1k_queries": 0.0, "notes": "Fuerte en inglés técnico y científico.",
    },
    "voyage-rerank-2": {
        "provider": "voyage", "model": "rerank-2", "cost_per_1k_queries": 0.05,
        "notes": "Voyage Rerank 2, multilingüe.",
    },
    "cohere-rerank-3-5": {
        "provider": "cohere", "model": "rerank-v3.5", "cost_per_1k_queries": 2.0,
        "notes": "Cohere Rerank 3.5, estado del arte y el más caro.",
    },
}


_CROSS_CACHE: dict = {}
_CLIENTE_CACHE: dict = {}


def _cross_encoder(model: str):
    """Cross-encoder HF cacheado: se carga UNA vez por proceso, no por pregunta."""
    if model not in _CROSS_CACHE:
        from sentence_transformers import CrossEncoder
        _CROSS_CACHE[model] = CrossEncoder(model)
    return _CROSS_CACHE[model]


def _cliente_api(provider: str):
    """Clientes Voyage/Cohere cacheados por proceso."""
    if provider not in _CLIENTE_CACHE:
        if provider == "voyage":
            import voyageai
            _CLIENTE_CACHE[provider] = voyageai.Client(api_key=CONFIG.voyage_api_key)
        elif provider == "cohere":
            import cohere
            _CLIENTE_CACHE[provider] = cohere.Client(CONFIG.cohere_api_key)
    return _CLIENTE_CACHE[provider]


def aplicar_reranker(provider: str, model: str, consulta: str,
                    docs: list, k_final: int) -> tuple[list, float]:
    """
    Reordena `docs` y devuelve los k_final mejores y la latencia.

    La construccion del modelo/cliente queda FUERA del cronometro (y cacheada):
    la latencia mide solo la inferencia/llamada, que es lo comparable entre
    rerankers locales y de API. Antes, cada pregunta cargaba el cross-encoder
    desde disco y esa carga inflaba la latencia de los modelos locales.
    """
    if provider == "none":
        return docs[:k_final], 0.0
    if provider == "huggingface":
        cross = _cross_encoder(model)                    # carga cacheada, sin cronometrar
        t0 = time.perf_counter()
        pares = [(consulta, d.page_content) for d in docs]
        puntuaciones = cross.predict(pares)
        ranking = sorted(zip(puntuaciones, docs), key=lambda x: -x[0])
        return [d for _, d in ranking[:k_final]], round(time.perf_counter() - t0, 4)
    if provider in ("voyage", "cohere"):
        cliente = _cliente_api(provider)                 # cliente cacheado
        textos = [d.page_content for d in docs]
        t0 = time.perf_counter()
        if provider == "voyage":
            resp = cliente.rerank(query=consulta, documents=textos, model=model,
                                  top_k=k_final)
        else:
            resp = cliente.rerank(model=model, query=consulta, documents=textos,
                                  top_n=k_final)
        return ([docs[r.index] for r in resp.results],
                round(time.perf_counter() - t0, 4))
    return docs[:k_final], 0.0


class BancoRerankers(BancoBase):
    nombre = "reranker"
    clave_decision = "reranker_recomendado"
    descripcion = "rerankers (reordenadores de candidatos)"
    esquema_json = (
        "{\n"
        '  "reranker_recomendado": "<clave del registro>",\n'
        '  "provider": "<none|huggingface|voyage|cohere>",\n'
        '  "k_final": <int>,\n'
        '  "confianza": "<alta|media|baja>",\n'
        '  "justificacion": "<2-3 frases; valora rerank_gain frente a coste>",\n'
        '  "alertas": ["<aviso si hay algo preocupante>"]\n'
        "}"
    )
    # Primaria: MRR. Secundarios: coste y latencia (donde se pagan los de API).
    # rerank_gain (delta de MRR vs no reordenar) se MUESTRA como contexto pero no
    # decide, para no contar dos veces la misma señal que la primaria.
    metrica_primaria = ("mrr", "max")
    # rerank_gain NO es objetivo de decision: gain = mrr - mrr(no_reranker),
    # una constante compartida por run -> ranking y duelos identicos a los del
    # mrr (decision de Xavier: se calcula y se comunica, pero no decide).
    objetivos_secundarios = [("relevancia_media", "max"),
                             ("coste_1k", "min"), ("latency", "min")]
    metricas_contexto = ("rerank_gain", "relevancia_media", "recuperacion_pasaje")

    def preparar_contexto(self, sesion: Sesion) -> dict:
        """Índice + candidatos (una vez) + n-gramas y vectores árbitro de las citas."""
        splitter = _construir_splitter(sesion.config_splitter)
        chunks = splitter.split_documents(sesion.documentos)

        from .embeddings_banco import embedding_base_exploratorio
        config_emb = sesion.config_embedding or embedding_base_exploratorio()
        modelo = construir_embedding(config_emb)
        # Indice base cacheado POR DOCUMENTO: los chunks y el modelo base no
        # cambian entre runs; sin esto, cada run (instancia nueva del banco)
        # re-embebía el corpus entero solo para recuperar candidatos.
        vectorstore = por_documento(
            "reranker", sesion.run_id,
            (huella_troceado(chunks), config_emb.get("model")),
            lambda: FAISS.from_documents(chunks, modelo))

        cfg_ret = sesion.config_retriever or {"retriever_name": "similarity", "params": {}}
        recuperador = construir_recuperador(
            cfg_ret.get("retriever_name", "similarity"),
            cfg_ret.get("params", {}), vectorstore, chunks, K_CANDIDATOS)

        candidatos = [recuperador.invoke(gq.get("question", ""))
                      for gq in sesion.preguntas_test]

        # Árbitro semántico (fijo) + precomputo de las citas de test.
        juez = obtener_embedding_juez()
        cache: dict = {}
        grams_objetivo = [gramas_de(gq.get("source_context", ""))
                          for gq in sesion.preguntas_test]
        emb_objetivo = [vector_normalizado(gq.get("source_context", ""), juez, cache)
                        for gq in sesion.preguntas_test]

        print(f"[reranker] Candidatos recuperados con "
              f"'{cfg_ret.get('retriever_name', 'similarity')}' "
              f"para {len(candidatos)} preguntas")
        contexto = {"candidatos": candidatos, "grams_objetivo": grams_objetivo,
                    "emb_objetivo": emb_objetivo, "juez": juez, "cache": cache,
                    # Suelo del árbitro calibrado UNA vez y compartido por los
                    # 6 experimentos: misma recalibración para todos (justicia).
                    "base_sem": obtener_base_semantica(juez)}
        # Nota del BASELINE (sin reordenar) precalculada: es identica para los 6
        # experimentos; antes se recalculaba en cada uno (6x el mismo trabajo).
        contexto["notas_base"] = [self._nota(candidatos[i][:K_FINAL], i, contexto)
                                  for i in range(len(candidatos))]
        return contexto

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        return [{"etiqueta": clave, "clave": clave, **cfg}
                for clave, cfg in RERANKER_REGISTRY.items()]

    def _nota(self, docs: list, i: int, contexto: dict) -> dict:
        """Nota híbrida de una lista de documentos para la pregunta i."""
        textos = [getattr(d, "page_content", "") or "" for d in docs]
        sims = similitudes_semanticas(textos, contexto["emb_objetivo"][i],
                                      contexto["juez"], contexto["cache"],
                                      base=contexto["base_sem"]) if docs else None
        return evaluar_recuperacion(docs, contexto["grams_objetivo"][i], sims=sims,
                                    peso_exactitud=PESO_EXACTITUD)

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        candidatos = contexto["candidatos"]
        provider, model = experimento["provider"], experimento["model"]

        relevancias, mrrs, recalls, gains, latencias = [], [], [], [], []
        for i, gq in enumerate(sesion.preguntas_test):
            docs = candidatos[i]
            reordenados, lat = aplicar_reranker(provider, model,
                                                gq.get("question", ""), docs, K_FINAL)
            m = self._nota(reordenados, i, contexto)
            base = contexto["notas_base"][i]

            relevancias.append(m["relevancia"])
            mrrs.append(m["mrr"])
            recalls.append(m["recuperacion_pasaje"])
            gains.append(m["mrr"] - base["mrr"])
            latencias.append(lat)

        # CV de dispersión: con media 0 no hay señal que sea "estable"; el CV
        # clásico daría 0/0->0 (el MEJOR valor de un objetivo min) y una config
        # totalmente fallida ganaría desempates. Centinela = peor dispersión.
        media_rel = float(np.mean(relevancias)) if relevancias else 0.0
        cv = (round(float(np.std(relevancias)) / media_rel, 4)
              if media_rel > 1e-9 else CV_SIN_SENAL)
        return {
            "relevancia_media": round(media_rel, 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "recuperacion_pasaje": round(float(np.mean(recalls)), 4),
            "rerank_gain": round(float(np.mean(gains)), 4),
            "cv_variance": cv,
            "latency": round(float(np.mean(latencias)), 4),
            "coste_1k": experimento.get("cost_per_1k_queries", 0.0),
            # Listas por pregunta: las necesita _confianza_estadistica (base.py)
            # para el IC bootstrap y la comparacion pareada lider vs 2o. Sin ellas,
            # el banco de rerankers quedaba sin confianza estadistica (disponible=False).
            "_relevancias": relevancias,
            "_mrrs": mrrs,
            "_recuperaciones": recalls,
        }
