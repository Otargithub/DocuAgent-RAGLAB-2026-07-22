"""
rag_lab: laboratorio modular de pruebas para sistemas RAG.

Idea general
------------
Un meta-supervisor dirige cuatro bancos de pruebas (splitters, embeddings,
retrievers, rerankers) para encontrar la mejor combinación de componentes para
un documento dado. Todo se orquesta con LangGraph, todo lo aprendido se guarda
en una memoria unificada, y dos análisis complementarios (LDA de temas y UMAP
del espacio de embeddings) alimentan las decisiones.

Uso rápido
----------
    from rag_lab import MetaSupervisor

    meta = MetaSupervisor()
    # Modo autónomo:
    stack = meta.ejecutar("documento.pdf")
    # Por instrucción:
    stack = meta.ejecutar("documento.pdf", "prueba solo embeddings y rerankers, prioriza coste")
"""

from .config import CONFIG
from .memory import Memoria
from .supervisor import MetaSupervisor, Plan

__all__ = ["CONFIG", "Memoria", "MetaSupervisor", "Plan"]
