"""
Fábricas de modelos: un único sitio donde se construyen LLMs y embeddings.

Antes, cada banco tenía su propia copia de `_build_embedding_model`, y empezaron
a divergir. Aquí las centralizamos. Para añadir un proveedor nuevo, se toca solo
este fichero.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import CONFIG


def _dispositivo_hf() -> str:
    """Elige el mejor backend disponible para modelos locales: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — si no hay torch, cpu y listo
        pass
    return "cpu"


def construir_llm_juez(temperatura: float = 0.0) -> ChatAnthropic:
    """El LLM que razona sobre resultados (supervisores y meta-supervisor)."""
    return ChatAnthropic(model=CONFIG.modelo_juez, temperature=temperatura)


def construir_llm_clasificador(temperatura: float = 0.0) -> ChatOpenAI:
    """El LLM barato que clasifica el documento y genera preguntas de test."""
    return ChatOpenAI(model=CONFIG.modelo_clasificador, temperature=temperatura)


_EMBEDDINGS_HF_CACHE: dict = {}


def construir_embedding(config: dict) -> Any:
    """
    Crea un modelo de embeddings a partir de su configuración.

    `config` es un diccionario como {"model": "...", "provider": "..."}.
    Importamos los proveedores no-OpenAI de forma perezosa (dentro del if) para
    no obligar a instalar todas las librerías si solo se va a usar OpenAI.
    """
    proveedor = config.get("provider", "openai")
    modelo = config.get("model", "text-embedding-3-small")

    if proveedor == "openai":
        # timeout: sin el, una conexion estancada cuelga la corrida entera en
        # silencio (visto en Colab); con el, falla en ~60s, el experimento se
        # registra como fallo y el banco continua. max_retries acota la espera.
        return OpenAIEmbeddings(model=modelo, request_timeout=60, max_retries=2)

    if proveedor == "huggingface":
        # CACHE por proceso (gemelo del de cross-encoders): sin el, cada run de
        # cada banco recargaba el modelo local desde disco (~13s bge-m3 x run).
        # El encoder es sin estado: compartirlo entre bancos/runs es seguro.
        clave_cache = (modelo, _dispositivo_hf())
        if clave_cache not in _EMBEDDINGS_HF_CACHE:
            from langchain_huggingface import HuggingFaceEmbeddings
            _EMBEDDINGS_HF_CACHE[clave_cache] = HuggingFaceEmbeddings(
                model_name=modelo,
                model_kwargs={"device": _dispositivo_hf()},
                encode_kwargs={"normalize_embeddings": True},
            )
        return _EMBEDDINGS_HF_CACHE[clave_cache]

    if proveedor == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings
        return VoyageAIEmbeddings(voyage_api_key=CONFIG.voyage_api_key, model=modelo)

    # Si llega un proveedor que no conocemos, no fallamos en silencio: avisamos
    # y caemos al embedding neutro para que el experimento pueda continuar.
    print(f"[fabricas] Proveedor '{proveedor}' desconocido; uso el embedding neutro.")
    return OpenAIEmbeddings(model=CONFIG.embedding_neutro["model"])
