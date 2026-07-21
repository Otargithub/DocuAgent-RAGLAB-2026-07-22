"""Análisis complementarios del documento: LDA (temas) y UMAP (espacio semántico)."""

from .lda import analizar_temas
from .umap_proyeccion import proyectar_embeddings

__all__ = ["analizar_temas", "proyectar_embeddings"]
