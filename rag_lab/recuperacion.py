"""
Estrategias de recuperacion.

Tanto el banco de retrievers como el de rerankers necesitan construir un
recuperador a partir de una configuracion. Las cuatro estrategias son:

  - similarity : vecinos mas cercanos por similitud del coseno.
  - mmr        : Maximal Marginal Relevance; equilibra relevancia y diversidad.
  - bm25       : recuperacion lexica clasica (coincidencia de palabras).
  - ensemble   : combinacion ponderada de bm25 y similitud (implementacion propia
                 para no depender de la ubicacion cambiante de EnsembleRetriever
                 entre versiones de LangChain).
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_community.retrievers import BM25Retriever

from .metricas import _tokenizar


def _construir_bm25(chunks: list, k: int) -> BM25Retriever:
    """
    BM25 con preprocesado adecuado para español (y para cualquier idioma con
    tildes y puntuación pegada). El default de LangChain es `texto.split()`:
    sin minúsculas ni limpieza, "Válvula", "válvula," y "válvula." son TRES
    términos distintos y el BM25 queda lisiado en corpus no ingleses. Usamos el
    MISMO tokenizador que la métrica léxica (minúsculas + solo [a-z0-9áéíóúüñ]),
    así recuperación y medición hablan el mismo idioma.
    """
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=_tokenizar)
    bm25.k = k
    return bm25


class _EnsembleRetriever:
    """
    Combinacion PONDERADA de BM25 y similitud vectorial mediante RRF.

    Reciprocal Rank Fusion: cada documento recibe peso_fuente * 1/(rango + 60)
    desde cada recuperador, y ordenamos por la suma. El peso de bm25 (`w_bm25`)
    es la variable que prueba el banco de retrievers; el de la parte vectorial es
    su complemento (1 - w_bm25).

    Dos correcciones respecto a la version anterior:
      - Antes el `w_bm25` se ignoraba (RRF sin pesos), asi que los experimentos
        de ensemble con distinto peso daban EXACTAMENTE el mismo resultado: el
        parametro que el banco creia medir no tenia efecto.
      - La identidad de un documento se calcula ahora con un hash de TODO su
        contenido, no con sus primeros 120 caracteres (que colisionaban entre
        fragmentos con cabeceras o pies repetidos, descartando documentos validos).
    """

    def __init__(self, bm25, vectorial, k: int, w_bm25: float = 0.5):
        self.bm25 = bm25
        self.vectorial = vectorial
        self.k = k
        self.w_bm25 = w_bm25

    def invoke(self, consulta: str) -> list:
        resultados_bm25 = self.bm25.invoke(consulta)
        resultados_vec = self.vectorial.invoke(consulta)

        puntuaciones: dict[str, float] = {}
        documentos: dict[str, Any] = {}

        fuentes = [(resultados_bm25, self.w_bm25),
                   (resultados_vec, 1.0 - self.w_bm25)]
        for fuente, peso in fuentes:
            for rango, doc in enumerate(fuente):
                clave = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                puntuaciones[clave] = puntuaciones.get(clave, 0.0) + peso * (1.0 / (rango + 60))
                documentos[clave] = doc

        ordenados = sorted(puntuaciones, key=lambda c: -puntuaciones[c])
        return [documentos[c] for c in ordenados[: self.k]]


def construir_recuperador(nombre: str, params: dict, vectorstore: Any,
                          chunks: list, k: int) -> Any:
    """
    Devuelve un objeto con metodo .invoke(consulta) -> lista de documentos.

    Parametros
    ----------
    nombre    : 'similarity' | 'mmr' | 'bm25' | 'ensemble'.
    params    : parametros propios de la estrategia (lambda_mult, w_bm25...).
    vectorstore : indice FAISS ya construido.
    chunks    : fragmentos originales (los necesita bm25, que es lexico).
    k         : cuantos documentos recuperar.
    """
    if nombre == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    if nombre == "mmr":
        # fetch_k venia en params desde el banco pero aqui se IGNORABA (se usaba
        # k*2): los experimentos mmr_k5 corrian con un pool de 10 candidatos, la
        # mitad del que el banco creia probar. Ahora params manda.
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k,
                           "fetch_k": int(params.get("fetch_k", max(k * 2, 20))),
                           "lambda_mult": params.get("lambda_mult", 0.5)},
        )

    if nombre == "bm25":
        return _construir_bm25(chunks, k)

    if nombre == "ensemble":
        # Listas de candidatos MAS PROFUNDAS que k antes de fusionar: con k+k
        # fusionados a k, el RRF apenas puede reordenar. Profundidad estandar:
        # cada fuente aporta max(3k, 20) y la fusion se queda con los k mejores.
        profundidad = int(params.get("profundidad", max(3 * k, 20)))
        bm25 = _construir_bm25(chunks, profundidad)
        vectorial = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": profundidad}
        )
        return _EnsembleRetriever(bm25=bm25, vectorial=vectorial, k=k,
                                  w_bm25=params.get("w_bm25", 0.5))

    # Estrategia desconocida: similarity siempre funciona.
    return vectorstore.as_retriever(search_kwargs={"k": k})
