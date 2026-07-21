"""
UMAP COMPARATIVO POR EMBEDDING — herramienta HUMANA para `analizar.ipynb`.

Responde a la pregunta "¿como separa cada embedding ESTE documento?" proyectando
el mismo corpus con varios modelos y comparando siluetas lado a lado.

IMPORTANTE (principio de diseño ya acordado):
  - Esto es para TUS ojos: los scatter 2D no se pasan a ningun LLM ni alimentan
    el perfil del documento. La silueta del PERFIL se mide siempre con el
    embedding base neutro para que los documentos sean comparables entre si;
    aqui, en cambio, variamos el embedding a proposito para diagnostico.
  - Un embedding con mejor silueta NO es automaticamente el mejor recuperador:
    la palabra final la tienen el banco de embeddings y la Fase B. Esto es un
    mapa, no un veredicto.

Uso tipico en el notebook:

    from rag_lab.analisis.umap_comparativo import comparar_embeddings_umap, graficar
    res = comparar_embeddings_umap(sesion.documentos)          # todos los del registro
    res = comparar_embeddings_umap(sesion.documentos,
                                   claves=["bge-m3", "voyage-3-lite"])  # subconjunto
    graficar(res)                                               # rejilla de scatters
"""
from typing import Any

from ..bancos.embeddings_banco import EMBEDDING_REGISTRY, _construir_splitter
from ..fabricas import construir_embedding
from .umap_proyeccion import proyectar_embeddings


def comparar_embeddings_umap(documentos: list[Any],
                             claves: list[str] | None = None,
                             splitter_cfg: dict | None = None,
                             max_fragmentos: int = 300) -> dict:
    """
    Proyecta el MISMO troceado con cada embedding y devuelve, por clave:
    {silueta, calidad, metodo, n_fragmentos, coordenadas}. Los proveedores no
    instalados o sin API key se saltan con su motivo (degradacion limpia).
    El troceado es fijo (por defecto, el del pipeline base) para que la unica
    variable sea el embedding: comparacion ceteris paribus tambien aqui.
    """
    splitter_cfg = splitter_cfg or {"name": "recursive", "chunk_size": 1000,
                                    "chunk_overlap": 200}
    chunks = _construir_splitter(splitter_cfg).split_documents(documentos)
    if len(chunks) > max_fragmentos:
        paso = len(chunks) / max_fragmentos
        chunks = [chunks[int(i * paso)] for i in range(max_fragmentos)]

    seleccion = claves or list(EMBEDDING_REGISTRY.keys())
    resultados: dict[str, dict] = {}
    for clave in seleccion:
        cfg = EMBEDDING_REGISTRY.get(clave)
        if not cfg:
            resultados[clave] = {"disponible": False, "motivo": "no esta en el registro"}
            continue
        try:
            modelo = construir_embedding({"model": cfg.get("model", clave),
                                          "provider": cfg.get("provider", "openai")})
            proy = proyectar_embeddings(chunks, modelo, max_fragmentos=max_fragmentos)
        except Exception as e:  # noqa: BLE001
            resultados[clave] = {"disponible": False, "motivo": str(e)}
            print(f"[umap_comp] '{clave}' saltado: {e}")
            continue
        resultados[clave] = proy
        if proy.get("disponible"):
            print(f"[umap_comp] {clave:24} silueta={proy.get('separacion_silueta')} "
                  f"({proy.get('calidad_espacio')}, via {proy.get('metodo', '?')})")
    return resultados


def graficar(resultados: dict, titulo: str = "Separacion semantica por embedding"):
    """
    Rejilla de scatters 2D, uno por embedding, con su silueta en el titulo.
    Devuelve la figura de matplotlib (para guardarla si quieres).
    """
    import matplotlib.pyplot as plt

    validos = {k: v for k, v in resultados.items()
               if v.get("disponible") and v.get("coordenadas")}
    if not validos:
        print("[umap_comp] nada que dibujar (¿todos los proveedores fallaron?)")
        return None
    n = len(validos)
    cols = min(3, n)
    filas = -(-n // cols)
    fig, ejes = plt.subplots(filas, cols, figsize=(5 * cols, 4.2 * filas),
                             squeeze=False)
    # Orden por silueta descendente: el que mejor separa, primero.
    orden = sorted(validos.items(),
                   key=lambda kv: -(kv[1].get("separacion_silueta") or -1))
    for i, (clave, r) in enumerate(orden):
        ax = ejes[i // cols][i % cols]
        xs, ys = zip(*r["coordenadas"])
        ax.scatter(xs, ys, s=8, alpha=0.6)
        ax.set_title(f"{clave}\nsilueta={r.get('separacion_silueta')} "
                     f"({r.get('calidad_espacio')})", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    for j in range(n, filas * cols):
        ejes[j // cols][j % cols].axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()
    return fig
