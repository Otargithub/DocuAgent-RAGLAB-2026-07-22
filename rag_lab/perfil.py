"""
PERFIL DEL DOCUMENTO — caracterizacion determinista, sin LLM.

Sustituye a la clasificacion por LLM (`doc_type`) y a los metadatos
estructurales predichos. Seis features accionables:

  densidad_textual      tokens/pagina, normalizada a 0..1 (escala fija)
  prop_tablas           proporcion de paginas con lineas tabulares (heuristica)
  idioma                'es' | 'en' | 'mixto' (stopwords, determinista)
  repetitividad_lexica  Jaccard medio de vocabulario entre segmentos (0..1)
  dispersion_lda        se rellena tras el analisis LDA (0..1)
  silueta_umap          se rellena tras el analisis UMAP (-1..1)

La similitud entre documentos se calcula en este espacio (idioma compatible +
1 - diferencia media de las numericas normalizadas). `descriptor()` produce una
etiqueta legible SOLO para logs; nunca es una clave de aprendizaje.
"""
import re
from typing import Any

# Escala fija para normalizar densidad (tokens/pagina). 600 tokens/pagina o mas
# se considera densidad maxima (texto corrido a pagina completa).
_DENSIDAD_TOPE = 600.0

_STOP_ES = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un",
            "una", "es", "se", "por", "con", "para", "su", "al", "lo", "como",
            "mas", "más", "pero", "sus", "este", "esta", "entre", "cuando"}
_STOP_EN = {"the", "of", "and", "to", "in", "a", "is", "that", "it", "for",
            "on", "with", "as", "are", "this", "be", "by", "from", "or",
            "an", "at", "which", "was", "their", "has", "have", "not"}

NUMERICAS = ["densidad_textual", "prop_tablas", "repetitividad_lexica",
             "dispersion_lda", "silueta_umap"]


def _palabras(texto: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+", texto.lower())


def _es_linea_tabular(linea: str) -> bool:
    """Heuristica de linea CANDIDATA a tabla, endurecida dos veces con casos
    reales: (1) un '|' suelto o una fecha de diario ('15   Lunes   1954')
    disparaban el criterio viejo; (2) hay extractores de PDF que emiten \t
    como separador DE PALABRA (Annie: 'Mi\tmadre\tempezo' en cada linea de
    prosa => 91.5% de "tablas" en un diario). Regla: los tabs y los huecos
    anchos son SEPARADORES, y una fila de tabla ademas lleva carga real de
    digitos, que la prosa no tiene."""
    s = linea.strip()
    if s.count("|") >= 2:
        return True
    separadores = len(re.findall(r"\s{3,}", s)) + s.count("\t")
    digitos = sum(c.isdigit() for c in s)
    return separadores >= 2 and digitos >= 3 and digitos / max(len(s), 1) >= 0.08


def idioma_de(texto: str) -> str:
    """'es', 'en' o 'mixto' por conteo de stopwords. Determinista."""
    palabras = set(_palabras(texto))
    es = len(palabras & _STOP_ES)
    en = len(palabras & _STOP_EN)
    if es == 0 and en == 0:
        return "mixto"
    menor, mayor = min(es, en), max(es, en)
    if mayor and menor / mayor >= 0.5:
        return "mixto"
    return "es" if es > en else "en"


def repetitividad_lexica(textos_segmentos: list[str]) -> float:
    """
    Jaccard medio del vocabulario entre pares de segmentos consecutivos.
    Alta (→1) = el documento reusa el mismo vocabulario en todas partes
    (senal de que bm25/lexico discriminara bien y el denso tendra empates).
    """
    vocabularios = [set(_palabras(t)) for t in textos_segmentos if t.strip()]
    vocabularios = [v for v in vocabularios if v]
    if len(vocabularios) < 2:
        return 0.0
    jaccards = []
    for a, b in zip(vocabularios, vocabularios[1:]):
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 0.0)
    return round(sum(jaccards) / len(jaccards), 4)


def perfil_de_documento(documentos: list[Any]) -> dict:
    """
    Calcula las features linguisticas del perfil a partir de las paginas
    cargadas (objetos con .page_content). LDA/UMAP se rellenan despues
    (`dict.update`) cuando el meta corre los analisis.
    """
    paginas = [getattr(d, "page_content", "") or "" for d in documentos]
    texto = "\n".join(paginas)
    n_pag = max(1, len(paginas))

    tokens = len(_palabras(texto))
    densidad = min(1.0, (tokens / n_pag) / _DENSIDAD_TOPE)

    def _tiene_bloque_tabular(p: str, minimo: int = 3) -> bool:
        # Una tabla es un BLOQUE: lineas tabulares CONSECUTIVAS. Contar lineas
        # sueltas por toda la pagina confundia diarios (fechas dispersas) con
        # tablas; tres fechas en tres entradas distintas no son una tabla.
        racha = 0
        for l in p.splitlines():
            racha = racha + 1 if _es_linea_tabular(l) else 0
            if racha >= minimo:
                return True
        return False

    pag_con_tabla = sum(1 for p in paginas if _tiene_bloque_tabular(p))
    prop_tablas = pag_con_tabla / n_pag

    return {
        "densidad_textual": round(densidad, 4),
        "prop_tablas": round(prop_tablas, 4),
        "idioma": idioma_de(texto),
        "repetitividad_lexica": repetitividad_lexica(paginas),
        "dispersion_lda": None,   # se rellena tras el analisis LDA
        "silueta_umap": None,     # se rellena tras el analisis UMAP
        "n_paginas": len(paginas),
    }


def similitud(a: dict, b: dict) -> float:
    """
    Similitud 0..1 entre dos perfiles: idioma incompatible => 0; si no,
    1 - diferencia media absoluta de las features numericas presentes en ambos.
    La silueta (-1..1) se reescala a 0..1 antes de comparar.
    """
    if not a or not b:
        return 0.0
    ia, ib = a.get("idioma"), b.get("idioma")
    if ia and ib and ia != ib and "mixto" not in (ia, ib):
        return 0.0
    difs = []
    for k in NUMERICAS:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            continue
        if k == "silueta_umap":
            va, vb = (va + 1) / 2, (vb + 1) / 2
        difs.append(abs(float(va) - float(vb)))
    if not difs:
        return 0.0
    return round(max(0.0, 1.0 - sum(difs) / len(difs)), 4)


def descriptor(perfil: dict) -> str:
    """Etiqueta legible SOLO para logs (nunca clave de aprendizaje)."""
    if not perfil:
        return "sin-perfil"
    partes = [str(perfil.get("idioma") or "?")]
    d = perfil.get("densidad_textual")
    if d is not None:
        partes.append("denso" if d >= 0.6 else "ligero" if d < 0.3 else "medio")
    if (perfil.get("prop_tablas") or 0) >= 0.2:
        partes.append("tablas")
    r = perfil.get("repetitividad_lexica")
    if r is not None:
        partes.append(f"lex-{'alta' if r >= 0.35 else 'baja' if r < 0.15 else 'media'}")
    return "|".join(partes)
