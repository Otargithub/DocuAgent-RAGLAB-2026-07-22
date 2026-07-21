"""
Trucos estadísticos para darle NÚMERO a la confianza de cada veredicto.

No reducen el error de una media a fuerza bruta (eso solo lo hace más N), sino
que (a) lo CUANTIFICAN —intervalo de confianza por bootstrap— y (b) reducen el
ruido de la DECISIÓN comparando configuraciones de forma pareada (mismas
preguntas para todas: la dificultad de cada pregunta es ruido común que se
cancela en la diferencia). Todo es cómputo local sobre los scores por pregunta
que ya se calcularon: coste de API cero.

Esto es infraestructura de MEDICIÓN (como el número de preguntas de test), no una
perilla del plan: el supervisor no decide usarlo, simplemente mide mejor y recibe
la incertidumbre en su contexto de decisión.

Determinismo
------------
El remuestreo se siembra con una semilla textual (p. ej. "doc_type|banco|metrica")
vía un generador propio de numpy, de modo que dos ejecuciones del mismo caso dan
exactamente los mismos intervalos. Reproducible entre procesos.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _generador(semilla: str) -> np.random.Generator:
    """Generador numpy sembrado de forma determinista desde una cadena."""
    digest = hashlib.sha512(semilla.encode("utf-8")).digest()
    semilla_int = int.from_bytes(digest[:8], "big")
    return np.random.default_rng(semilla_int)


def bootstrap_ic(valores: list[float], n_boot: int = 1000, alpha: float = 0.05,
                 semilla: str = "ic") -> dict:
    """
    Intervalo de confianza de la MEDIA por bootstrap (percentil).

    Remuestrea con reemplazo `valores` n_boot veces, calcula la media de cada
    remuestreo y toma los percentiles alpha/2 y 1-alpha/2.

    Devuelve {media, ic_bajo, ic_alto, amplitud, n}. Con menos de 2 valores no
    hay incertidumbre estimable: el IC colapsa a la propia media.
    """
    v = np.asarray([x for x in valores if x is not None], dtype=float)
    n = len(v)
    if n == 0:
        return {"media": 0.0, "ic_bajo": 0.0, "ic_alto": 0.0, "amplitud": 0.0, "n": 0}
    media = float(np.mean(v))
    if n < 2:
        return {"media": round(media, 4), "ic_bajo": round(media, 4),
                "ic_alto": round(media, 4), "amplitud": 0.0, "n": n}
    rng = _generador(f"{semilla}|ic|{n}")
    idx = rng.integers(0, n, size=(n_boot, n))
    medias = v[idx].mean(axis=1)
    lo = float(np.percentile(medias, 100 * alpha / 2))
    hi = float(np.percentile(medias, 100 * (1 - alpha / 2)))
    return {"media": round(media, 4), "ic_bajo": round(lo, 4),
            "ic_alto": round(hi, 4), "amplitud": round(hi - lo, 4), "n": n}


def comparacion_pareada(scores_a: list[float], scores_b: list[float],
                        n_boot: int = 1000, alpha: float = 0.05,
                        semilla: str = "par") -> dict:
    """
    Compara dos configuraciones PAREADAS (evaluadas sobre las MISMAS preguntas).

    Trabaja sobre la diferencia por pregunta d_i = a_i - b_i. Como la dificultad
    de cada pregunta afecta por igual a A y a B, se cancela en la diferencia y la
    varianza de la media de d es mucho menor que la de cada media por separado:
    reducción de varianza sin un solo dato extra.

    Devuelve:
      dif_media   : media de (A - B). Positiva => A mejor.
      ic_bajo/alto: IC de esa diferencia por bootstrap.
      prob_a_mejor: fracción de remuestreos con media(A-B) > 0.
      concluyente : True si el IC no cruza 0 (la ventaja es estadísticamente clara).

    Si los vectores no están pareados (distinta longitud) o hay menos de 2 pares,
    se informa con `concluyente=False` y `motivo`.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        dif = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else 0.0
        return {"dif_media": round(dif, 4), "ic_bajo": 0.0, "ic_alto": 0.0,
                "prob_a_mejor": None, "concluyente": False,
                "motivo": "muestras no pareadas o insuficientes"}
    d = a - b
    n = len(d)
    dif_media = float(np.mean(d))
    rng = _generador(f"{semilla}|par|{n}")
    idx = rng.integers(0, n, size=(n_boot, n))
    medias = d[idx].mean(axis=1)
    lo = float(np.percentile(medias, 100 * alpha / 2))
    hi = float(np.percentile(medias, 100 * (1 - alpha / 2)))
    # SERIES IDENTICAS (visto en la Fase B con clones k10=k5): con d todo ceros,
    # mean(medias > 0) = 0.0, que se lee como "0% de que A sea mejor" cuando la
    # verdad es un empate perfecto. Ademas, en general, prob solo con ">" ignora
    # los remuestreos empatados: los contamos como medio punto (mid-p), que para
    # el caso todo-ceros da el 0.5 honesto y en los demas casos es mas fino.
    identicos = not bool(np.any(d))
    # VARIANZA CERO EN LA DIFERENCIA (todas las d iguales, != 0): el bootstrap
    # da un IC de anchura cero y CUALQUIER diferencia, por minuscula que sea,
    # saldria "concluyente" — division por cero disfrazada. Pasa cuando se
    # comparan MEDIAS DE RUNS deterministas (idénticas al decimal por la
    # reproducibilidad), no series por pregunta. Sin varianza no hay
    # incertidumbre estimable: NO se puede declarar concluyente.
    var_cero = bool(np.ptp(d) == 0) and not identicos
    prob = float(np.mean(medias > 0) + 0.5 * np.mean(medias == 0))
    concluyente = bool(lo > 0 or hi < 0) and not var_cero
    out = {"dif_media": round(dif_media, 4), "ic_bajo": round(lo, 4),
           "ic_alto": round(hi, 4), "prob_a_mejor": round(prob, 3),
           "concluyente": concluyente}
    if identicos:
        out["identicos"] = True
    if var_cero:
        out["motivo"] = ("varianza cero en la diferencia (¿medias de runs "
                         "deterministas en vez de series por pregunta?): la "
                         "diferencia puntual no tiene incertidumbre estimable, "
                         "NO es concluyente")
        out["prob_a_mejor"] = None
        out["motivo"] = "series identicas por pregunta (empate perfecto)"
    return out


# --------------------------------------------------------------------------- #
# Correlación de rangos (Spearman) con RANGOS MEDIOS en los empates
# --------------------------------------------------------------------------- #
# Unificada aquí en la fase 1 del refactor (2026-07-16): vivían dos copias
# (bancos/base._corr_rangos y anclaje_humano.spearman) con contratos casi
# iguales — divergencia silenciosa esperando ocurrir. Los rangos medios son
# imprescindibles con empates masivos (humano en {0,1,2}; lex empatando en 0):
# sin ellos, el coeficiente depende del ORDEN de la lista.
def _rangos_medios(v):
    v = np.asarray(v, dtype=float)
    orden = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[orden] = np.arange(1, len(v) + 1)
    for val in np.unique(v):
        m = v == val
        r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    """Spearman con rangos medios. NaN si alguna serie es constante."""
    rx, ry = _rangos_medios(x), _rangos_medios(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
