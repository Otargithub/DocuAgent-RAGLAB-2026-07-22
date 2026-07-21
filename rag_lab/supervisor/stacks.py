"""
Traducción de veredictos y certezas a configuraciones heredables.

Funciones PURAS (sin estado, sin memoria, sin LLM) extraídas de meta.py en la
fase 1 del refactor (2026-07-16): el ensamblado del stack es aritmética de
diccionarios y merece vivir donde se pueda testear a solas. meta.py conserva
delegados finos para no mover las costuras del grafo ni del golden.
"""
from __future__ import annotations

from ..bancos import CLAVE_DECISION
from ..bancos.embeddings_banco import EMBEDDING_REGISTRY


def config_splitter(veredicto: dict) -> dict:
    return {"name": veredicto.get("splitter_recomendado", "recursive"),
            "chunk_size": veredicto.get("chunk_size_recomendado", 1000),
            "chunk_overlap": veredicto.get("chunk_overlap_recomendado", 200)}


def config_embedding(veredicto: dict) -> dict:
    clave = veredicto.get("embedding_recomendado", "text-embedding-3-small")
    entrada = EMBEDDING_REGISTRY.get(clave)
    if entrada:
        return {"model": entrada["model"], "provider": entrada["provider"]}
    return {"model": clave, "provider": veredicto.get("provider", "openai")}


def config_retriever(veredicto: dict) -> dict:
    return {"retriever_name": veredicto.get("retriever_recomendado", "similarity"),
            "params": veredicto.get("params", {})}


def config_desde_veredicto(banco: str, veredicto: dict) -> dict:
    """Config evaluada de un banco (respaldo si un finalista no trae config)."""
    if banco == "splitter":
        return config_splitter(veredicto)
    if banco == "embedding":
        return config_embedding(veredicto)
    if banco == "retriever":
        return config_retriever(veredicto)
    return {CLAVE_DECISION[banco]: veredicto.get(CLAVE_DECISION[banco])}


def stack_desde_certezas(certezas: dict, veredictos: dict) -> dict:
    """
    Ensamblado DETERMINISTA: por banco, el LIDER evaluado (etiqueta + config
    real del experimento). Sin eleccion del LLM: si el banco esta resuelto no
    hay nada que elegir, y si esta en empate quien dirime es la Fase B
    end-to-end, no una opinion. Nada de parametros inventados.
    """
    stack = {}
    for banco, cer in certezas.items():
        finalistas = cer.get("finalistas") or []
        lider = next((f for f in finalistas if f.get("es_lider")),
                     finalistas[0] if finalistas else None)
        config = (lider or {}).get("config")
        if config is None:  # respaldo: banco sin finalistas con config
            config = config_desde_veredicto(banco, veredictos.get(banco, {}))
        stack[banco] = {"etiqueta": (lider or {}).get("etiqueta"),
                        "config": config,
                        "resuelto": cer.get("resuelto"),
                        "en_disputa": cer.get("es_empate")}
    return stack
