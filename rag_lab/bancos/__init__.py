"""Los cinco bancos de pruebas. Todos heredan de BancoBase."""

from .base import BancoBase
from .splitters import BancoSplitters
from .embeddings_banco import BancoEmbeddings
from .metadata import BancoMetadata
from .retrievers import BancoRetrievers
from .rerankers import BancoRerankers

__all__ = [
    "BancoBase", "BancoSplitters", "BancoEmbeddings",
    "BancoMetadata", "BancoRetrievers", "BancoRerankers",
]

# Orden del pipeline, nombres y clave de decision de cada banco. Viven aqui
# (casa natural) para que supervisor/plan.py, meta.py y los notebooks importen
# de un unico sitio sin ciclos.
PIPELINE = [
    ("splitter", BancoSplitters),
    ("embedding", BancoEmbeddings),
    ("metadata", BancoMetadata),
    ("retriever", BancoRetrievers),
    ("reranker", BancoRerankers),
]
NOMBRES_BANCOS = [n for n, _ in PIPELINE]
CLAVE_DECISION = {
    "splitter": "splitter_recomendado", "embedding": "embedding_recomendado",
    "metadata": "config_recomendada",
    "retriever": "retriever_recomendado", "reranker": "reranker_recomendado",
}
__all__ += ["PIPELINE", "NOMBRES_BANCOS", "CLAVE_DECISION"]
