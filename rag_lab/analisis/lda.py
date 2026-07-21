"""
Análisis LDA (Latent Dirichlet Allocation) del texto estudiado.

Qué hace y por qué es útil aquí
-------------------------------
LDA descubre, sin supervisión, los temas latentes de un conjunto de textos y, por
cada tema, qué palabras lo caracterizan. Sobre los fragmentos de un documento nos
dice cuántos "asuntos" contiene y cómo de mezclados están.

El meta-supervisor lo usa como señal cualitativa: un documento mono-temático
tolera fragmentos grandes (todo habla de lo mismo); uno con muchos temas dispersos
se beneficia de fragmentos más pequeños para no mezclar asuntos en un chunk.

Cambio P1: ahora filtra palabras vacías en ESPAÑOL e INGLÉS (antes solo inglés,
con lo que en castellano los temas se llenaban de 'el', 'la', 'de', 'que'...). El
idioma se puede forzar; por defecto filtra ambos, que es lo correcto para un
corpus bilingüe (es / en / es-en).

Nota: "LDA" también es Linear Discriminant Analysis. Aquí es topic modeling, que
es lo que tiene sentido sobre texto no etiquetado.
"""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Palabras vacías bilingües. Filtrar solo inglés dejaba pasar las castellanas y
# ensuciaba los temas en documentos en español o mixtos.
_STOP_EN = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of", "to",
    "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "we", "you",
    "they", "he", "she", "i", "not", "no", "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might", "must", "into", "over",
    "under", "than", "such", "which", "who", "what", "when", "where", "how", "why",
    "their", "there", "here", "our", "your", "his", "her", "them", "also", "been",
}
_STOP_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "u", "pero",
    "si", "no", "de", "del", "al", "a", "ante", "con", "contra", "desde", "en",
    "entre", "hacia", "hasta", "para", "por", "segun", "según", "sin", "sobre",
    "tras", "que", "se", "su", "sus", "le", "les", "lo", "es", "son", "era", "fue",
    "ser", "estar", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "como", "más", "mas", "menos", "ya", "muy", "también", "tambien", "donde",
    "cuando", "porque", "cual", "cuales", "nos", "ellos", "ellas", "él", "el",
    "ella", "yo", "tú", "tu", "mi", "han", "ha", "he", "hay", "fueron", "todo",
    "todos", "toda", "todas", "otro", "otra", "cada", "entre", "cabe", "durante",
}
_STOP_BILINGUE = sorted(_STOP_EN | _STOP_ES)


def _stopwords(idioma: str) -> list[str]:
    if idioma == "en":
        return sorted(_STOP_EN)
    if idioma == "es":
        return sorted(_STOP_ES)
    return _STOP_BILINGUE


def analizar_temas(documentos: list[Any], n_temas: int = 5,
                  palabras_por_tema: int = 8, idioma: str = "es/en") -> dict:
    """
    Ejecuta LDA sobre el texto del documento y resume los temas encontrados.

    Parámetros
    ----------
    documentos        : documentos LangChain (cada uno con .page_content).
    n_temas           : cuántos temas pedir (se recorta si hay menos documentos).
    palabras_por_tema : cuántas palabras-clave por tema.
    idioma            : 'es', 'en' o 'es/en' (solo afecta a las palabras vacías).

    Devuelve un dict con los temas y métricas globales, o {"disponible": False,...}.
    """
    textos = [d.page_content for d in documentos if getattr(d, "page_content", "").strip()]
    if len(textos) < 2:
        return {"disponible": False, "motivo": "Texto insuficiente para LDA."}

    n_temas_real = max(2, min(n_temas, len(textos)))

    # Vectorizamos en recuentos de palabras, quitando vacías (es/en) y términos
    # demasiado raros o demasiado frecuentes. token_pattern unicode para no
    # romper palabras acentuadas.
    # min_df adaptativo: en corpus grandes exigimos que un termino aparezca en 2+
    # documentos (filtra ruido); en corpus pequenos eso vaciaria el vocabulario,
    # asi que bajamos a 1.
    min_df = 2 if len(textos) >= 8 else 1
    vectorizador = CountVectorizer(
        stop_words=_stopwords(idioma),
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
        max_df=0.95, min_df=min_df, max_features=1000, lowercase=True,
    )
    try:
        matriz = vectorizador.fit_transform(textos)
    except ValueError:
        return {"disponible": False, "motivo": "Vocabulario vacío tras el filtrado."}

    vocabulario = vectorizador.get_feature_names_out()
    if len(vocabulario) < n_temas_real:
        return {"disponible": False,
                "motivo": f"Vocabulario demasiado pequeño ({len(vocabulario)} términos)."}

    modelo = LatentDirichletAllocation(
        n_components=n_temas_real, random_state=0, learning_method="batch", max_iter=20)
    distribucion_doc_tema = modelo.fit_transform(matriz)

    temas = []
    for indice, pesos in enumerate(modelo.components_):
        mejores = pesos.argsort()[: -palabras_por_tema - 1: -1]
        temas.append({"tema": indice, "palabras": [str(vocabulario[i]) for i in mejores]})

    # tema_dominante_medio cerca de 1 => fragmentos mono-temáticos;
    # dispersion_tematica alta => documento variado (favorece trocear más fino).
    peso_dominante = float(distribucion_doc_tema.max(axis=1).mean())
    return {
        "disponible": True,
        "n_temas": n_temas_real,
        "temas": temas,
        "tema_dominante_medio": round(peso_dominante, 3),
        "dispersion_tematica": round(1.0 - peso_dominante, 3),
        "n_fragmentos_analizados": len(textos),
        "idioma": idioma,
    }
