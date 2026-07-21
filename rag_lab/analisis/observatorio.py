"""
Observatorio: la capa de análisis TRANSVERSAL sobre la memoria del laboratorio.

Qué es
------
El meta-supervisor decide POR DOCUMENTO y persiste su evidencia en rag_memory.
El observatorio mira ESA memoria por encima de los documentos: qué configs
ganan en qué perfiles, duelos pareados entre dos configs cualesquiera con toda
la evidencia acumulada, correlaciones entre rasgos del perfil y rendimiento, y
la fiabilidad histórica de los priors. Todo con las MISMAS funciones
estadísticas del laboratorio (`bootstrap_ic`, `comparacion_pareada`): aquí no
se inventa estadística nueva, se reutiliza la auditada.

Principios
----------
- DETERMINISTA: numpy + estadistica.py; ninguna llamada a LLM. Esta capa es el
  motor de cálculo; si un agente (DocuAgent) quiere narrar encima, estas
  funciones son sus tools y cada número de su prosa debe salir de aquí.
- ERA-CONSCIENTE: casi todo acepta `desde` (ISO date). Las métricas cambiaron de
  escala en las recalibraciones; mezclar eras produce comparaciones sin sentido.
  Pásale la fecha de tu último reset de memoria.
- Fuente de datos: el `_ranking` compacto {etiqueta: valor} que cada experimento
  persiste desde base.py, más `_primaria`/`_finalistas` y las sesiones (perfil).

Uso rápido
----------
    from rag_lab.memory import Memoria
    from rag_lab.analisis import observatorio as obs
    mem = Memoria()
    print(obs.panorama(mem))
    print(obs.duelo(mem, "embedding", "bge-m3", "granite-311m-r2"))
    obs.informe(mem, desde="2026-07-05")
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..estadistica import bootstrap_ic, comparacion_pareada


# --------------------------------------------------------------------------- #
# Acceso base (filtrado por era)
# --------------------------------------------------------------------------- #
def _experimentos(mem, banco: str | None = None, desde: str | None = None) -> list[dict]:
    exps = mem._leer().get("experimentos", [])
    if banco:
        exps = [e for e in exps if e.get("banco") == banco]
    if desde:
        exps = [e for e in exps if (e.get("guardado_en") or "") >= desde]
    return exps


def _sesiones(mem, desde: str | None = None) -> list[dict]:
    ses = mem._leer().get("sesiones", [])
    if desde:
        ses = [s for s in ses if (s.get("guardado_en") or "") >= desde]
    return ses


def _perfil_por_run(mem, desde: str | None = None) -> dict:
    """run_id -> (nombre_pdf, perfil dict). La sesión más reciente gana."""
    return {s.get("run_id"): (s.get("pdf_nombre"), s.get("perfil") or {})
            for s in _sesiones(mem, desde)}


def _ranking_de(exp: dict) -> dict:
    """
    {etiqueta: valor_primaria} de un experimento. Preferimos el `_ranking`
    compacto (parrilla completa); si el experimento es de una era anterior a su
    introducción, degradamos a `_finalistas` (líder + empatados) — cobertura
    parcial pero honesta.
    """
    v = exp.get("veredicto", {}) or {}
    if v.get("_ranking"):
        return {k: float(x) for k, x in v["_ranking"].items() if x is not None}
    out = {}
    for f in v.get("_finalistas") or []:
        if f.get("etiqueta") and f.get("valor") is not None:
            out[str(f["etiqueta"])] = float(f["valor"])
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# EDA / análisis
# --------------------------------------------------------------------------- #
def panorama(mem, desde: str | None = None) -> dict:
    """Inventario de la evidencia: documentos, bancos, experimentos, cobertura."""
    ses = _sesiones(mem, desde)
    exps = _experimentos(mem, desde=desde)
    por_banco = Counter(e.get("banco") for e in exps)
    con_ranking = sum(1 for e in exps if (e.get("veredicto") or {}).get("_ranking"))
    return {
        "documentos": len({s.get("doc_hash") for s in ses}),
        "sesiones": len(ses),
        "experimentos": dict(por_banco),
        "experimentos_total": len(exps),
        "con_ranking_completo": con_ranking,
        "aviso": (None if con_ranking == len(exps) else
                  f"{len(exps) - con_ranking} experimentos son de una era sin "
                  f"_ranking: los análisis los ven parcialmente (solo finalistas)."),
    }


def etiquetas_con_evidencia(mem, banco: str, desde: str | None = None) -> dict:
    """
    {etiqueta: n_mediciones} de un banco. Es la herramienta de DESCUBRIMIENTO:
    un agente no puede adivinar identificadores, y sin esta funcion probaba
    nombres inventados ("openai", "cohere"), recibia "sin mediciones" y
    concluia que no habia evidencia... con 200 experimentos en memoria.
    """
    from collections import Counter
    n = Counter()
    for e in _experimentos(mem, banco, desde):
        for et in _ranking_de(e):
            n[et] += 1
    return dict(sorted(n.items(), key=lambda kv: -kv[1]))


def rendimiento(mem, banco: str, etiqueta: str, desde: str | None = None) -> dict:
    """Media + IC bootstrap de una config sobre TODAS sus mediciones."""
    vals = [r[etiqueta] for e in _experimentos(mem, banco, desde)
            for r in [_ranking_de(e)] if etiqueta in r]
    if not vals:
        return {"disponible": False, "etiqueta": etiqueta, "n": 0}
    ic = bootstrap_ic(vals, semilla=f"obs|{banco}|{etiqueta}")
    return {"disponible": True, "etiqueta": etiqueta, "n": len(vals),
            "media": round(float(np.mean(vals)), 4), **ic}


def duelo(mem, banco: str, a: str, b: str, desde: str | None = None) -> dict:
    """
    Comparación PAREADA de dos configs sobre todas las mediciones en que
    coincidieron (mismo run_id + run => mismas preguntas). Devuelve el veredicto
    global y el desglose por documento — la vista que revela "A gana en
    documentos densos pero pierde en los léxicos".
    """
    perfiles = _perfil_por_run(mem)
    va, vb, por_doc = [], [], defaultdict(lambda: [[], []])
    for e in _experimentos(mem, banco, desde):
        r = _ranking_de(e)
        if a in r and b in r:
            va.append(r[a]); vb.append(r[b])
            doc = perfiles.get(e.get("run_id"), ("?", {}))[0] or "?"
            por_doc[doc][0].append(r[a]); por_doc[doc][1].append(r[b])
    if len(va) < 3:
        # Con < 3 mediciones no hay test pareado, pero SI se puede describir la
        # diferencia observada. Damosla para que quien decide concluya con lo
        # que hay: perseguir mas runs es inutil si la fuente (experimento
        # dirigido) topa en 2 — la diferencia trivial ya es la respuesta.
        m_a = float(np.mean(va)) if va else 0.0
        m_b = float(np.mean(vb)) if vb else 0.0
        return {"disponible": False, "n": len(va),
                "media_a": round(m_a, 4), "media_b": round(m_b, 4),
                "dif_observada": round(m_a - m_b, 4),
                "motivo": (f"solo {len(va)} mediciones: sin test pareado formal "
                           f"(minimo 3). Diferencia observada {m_a - m_b:+.4f} "
                           f"(A={m_a:.4f}, B={m_b:.4f}). Si es trivial frente a "
                           f"la escala, ES un empate: concluye, no pidas mas "
                           f"runs (el experimento dirigido topa en 2).")}
    global_ = comparacion_pareada(va, vb, semilla=f"obs|{banco}|{a}|{b}")
    desglose = {doc: {"n": len(xa), "dif_media": round(float(np.mean(xa) - np.mean(xb)), 4)}
                for doc, (xa, xb) in sorted(por_doc.items())}
    return {"disponible": True, "a": a, "b": b, "n": len(va),
            "global": global_, "por_documento": desglose}


def victorias_por_perfil(mem, banco: str, desde: str | None = None) -> dict:
    """
    Tabla ganador × perfil: qué etiqueta lidera en qué tipo de documento. El
    ganador por documento es la MODA de los líderes de sus runs (con su n).
    """
    perfiles = _perfil_por_run(mem)
    lideres_por_run_id = defaultdict(list)
    clave = {"splitter": "splitter_recomendado", "embedding": "embedding_recomendado",
             "retriever": "retriever_recomendado", "reranker": "reranker_recomendado",
             "metadata": "metadata_recomendado"}.get(banco, f"{banco}_recomendado")
    for e in _experimentos(mem, banco, desde):
        g = (e.get("veredicto") or {}).get(clave)
        if g:
            lideres_por_run_id[e.get("run_id")].append(g)
    tabla = defaultdict(Counter)
    for run_id, lideres in lideres_por_run_id.items():
        ganador = Counter(lideres).most_common(1)[0][0]
        nombre, perfil = perfiles.get(run_id, ("?", {}))
        desc = perfil.get("descriptor") or perfil.get("doc_type")
        if not desc and perfil:
            # eras previas guardaban el perfil numerico SIN descriptor textual
            # (todo caia en '?'): se deriva, que es funcion pura del perfil
            try:
                from ..perfil import descriptor as _descriptor
                desc = _descriptor(perfil)
            except Exception:  # noqa: BLE001
                desc = None
        tabla[desc or "?"][ganador] += 1
    return {perfil: dict(c) for perfil, c in sorted(tabla.items())}


def correlacion_perfil(mem, banco: str, etiqueta: str, campo: str,
                       desde: str | None = None) -> dict:
    """
    ¿El rendimiento de una config depende de un rasgo NUMÉRICO del perfil
    (densidad, lex, tablas...)? Spearman entre el rasgo del documento y el valor
    medio de la etiqueta en ese documento. n = documentos con ambos datos.
    """
    perfiles = _perfil_por_run(mem)
    por_run = defaultdict(list)
    for e in _experimentos(mem, banco, desde):
        r = _ranking_de(e)
        if etiqueta in r:
            por_run[e.get("run_id")].append(r[etiqueta])
    xs, ys = [], []
    for run_id, vals in por_run.items():
        rasgo = (perfiles.get(run_id, ("?", {}))[1] or {}).get(campo)
        if isinstance(rasgo, (int, float)):
            xs.append(float(rasgo)); ys.append(float(np.mean(vals)))
    rho = spearman(xs, ys)
    return {"banco": banco, "etiqueta": etiqueta, "campo": campo,
            "n_documentos": len(xs), "spearman": None if np.isnan(rho) else round(rho, 3),
            "aviso": "n<5: correlación meramente orientativa" if len(xs) < 5 else None}


def fiabilidad_priors(mem, desde: str | None = None) -> dict:
    """Tasa de acierto histórica de los priors por banco (exacto y por familia)."""
    preds = mem._leer().get("predicciones", [])
    if desde:
        preds = [p for p in preds if (p.get("guardado_en") or "") >= desde]
    out = {}
    for banco in sorted({p.get("banco") for p in preds if p.get("banco")}):
        resueltas = [p for p in preds if p.get("banco") == banco
                     and p.get("acierto") is not None]
        if not resueltas:
            out[banco] = {"n": 0}
            continue
        exact = sum(1 for p in resueltas if p.get("acierto") is True
                    or p.get("acierto") == "exacto")
        out[banco] = {"n": len(resueltas),
                      "acierto_exacto": round(exact / len(resueltas), 3)}
    return out


def informe(mem, desde: str | None = None) -> None:
    """EDA imprimible: panorama + top de configs por banco + victorias por perfil."""
    print("=" * 70, "\n  OBSERVATORIO — análisis transversal de la memoria\n" + "=" * 70)
    pan = panorama(mem, desde)
    print(f"Documentos: {pan['documentos']} | sesiones: {pan['sesiones']} | "
          f"experimentos: {pan['experimentos_total']} {pan['experimentos']}")
    if pan.get("aviso"):
        print(f"AVISO: {pan['aviso']}")
    for banco in sorted(pan["experimentos"]):
        # top de etiquetas por media (con IC), a partir de los rankings compactos
        vals = defaultdict(list)
        for e in _experimentos(mem, banco, desde):
            for et, v in _ranking_de(e).items():
                vals[et].append(v)
        if not vals:
            continue
        print(f"\n[{banco}] {len(vals)} configs con evidencia:")
        filas = sorted(vals.items(), key=lambda kv: -float(np.mean(kv[1])))
        for et, v in filas[:6]:
            ic = bootstrap_ic(v, semilla=f"obs|{banco}|{et}")
            print(f"  {et:22} media={np.mean(v):.4f} "
                  f"IC=[{ic['ic_bajo']}, {ic['ic_alto']}] n={len(v)}")
        tabla = victorias_por_perfil(mem, banco, desde)
        if tabla:
            print(f"  victorias por perfil: {tabla}")
