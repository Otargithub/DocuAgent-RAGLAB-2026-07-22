"""
Proyección del espacio de embeddings del documento (UMAP, con respaldo PCA).

Qué hace y por qué es útil aquí
-------------------------------
Reduce los embeddings (cientos/miles de dimensiones) a 2D conservando la
estructura de vecindad, para dos cosas:

  1. Visualizar el "mapa semántico" del documento (coordenadas 2D).
  2. Medir cómo de ESTRUCTURADO está el espacio. Si los fragmentos forman grupos
     limpios, el embedding capta estructura y recuperar será más fácil; si todo es
     una nube indistinta, ese embedding no discrimina bien este documento.

El meta-supervisor aprende de aquí: una mala separación con un embedding es un
argumento para preferir otro.

Cambios P1
----------
  - Respaldo PCA: si umap-learn no está instalado, antes el análisis desaparecía
    (devolvía disponible:False). Ahora cae a PCA (scikit-learn, siempre presente),
    así la señal no se pierde en una máquina sin umap. El método usado se informa.
  - La SILUETA se calcula sobre los embeddings ORIGINALES, no sobre la proyección
    2D. Así la calidad del espacio no depende del método de reducción (la 2D solo
    sirve para dibujar) y es comparable entre máquinas con y sin umap.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _reducir_2d(vectores: np.ndarray) -> tuple[np.ndarray, str]:
    """Proyecta a 2D con UMAP si está disponible; si no, con PCA."""
    try:
        import umap  # umap-learn
        coords = umap.UMAP(n_components=2, random_state=0).fit_transform(vectores)
        return coords, "umap"
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(vectores)
        return coords, "pca"


def proyectar_embeddings(chunks: list[Any], modelo_embedding: Any,
                       max_fragmentos: int = 400, n_clusters: int = 5) -> dict:
    """
    Embebe los fragmentos, mide la estructura del espacio y los proyecta a 2D.

    Parámetros
    ----------
    chunks           : fragmentos a proyectar (objetos con .page_content).
    modelo_embedding : modelo de embeddings ya construido (de fabricas.py).
    max_fragmentos   : tope para no disparar el coste; si hay más, se muestrea.
    n_clusters       : número de grupos para la métrica de calidad (silueta).

    Devuelve separación (silueta), su etiqueta, el método de proyección y las
    coordenadas 2D; o {"disponible": False, ...} si no hay material suficiente.
    """
    textos = [c.page_content for c in chunks if getattr(c, "page_content", "").strip()]
    if len(textos) < n_clusters + 1:
        return {"disponible": False, "motivo": "Pocos fragmentos para proyectar."}

    # Muestreo determinista uniforme si hay demasiados fragmentos (no sesgar al
    # principio del documento).
    if len(textos) > max_fragmentos:
        paso = len(textos) / max_fragmentos
        textos = [textos[int(i * paso)] for i in range(max_fragmentos)]

    vectores = np.asarray(modelo_embedding.embed_documents(textos), dtype=np.float32)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Calidad del espacio: agrupamos los embeddings ORIGINALES y medimos la silueta.
    etiquetas = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(vectores)
    try:
        separacion = float(silhouette_score(vectores, etiquetas))
    except ValueError:
        separacion = 0.0

    # 2D solo para dibujar.
    coords, metodo = _reducir_2d(vectores)

    return {
        "disponible": True,
        "metodo": metodo,
        "n_fragmentos": len(textos),
        "separacion_silueta": round(separacion, 3),
        "calidad_espacio": _interpretar_silueta(separacion),
        "coordenadas": [[round(float(x), 3), round(float(y), 3)] for x, y in coords],
    }


def _interpretar_silueta(valor: float) -> str:
    """Traduce la silueta a una etiqueta legible para el supervisor."""
    if valor >= 0.5:
        return "alta"
    if valor >= 0.25:
        return "media"
    return "baja"
