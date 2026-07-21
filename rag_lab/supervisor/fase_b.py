"""
FASE B — la final factorial en terreno comun.

Toma los FINALISTAS de cada banco (los empatados que la Fase A no resolvio),
arma el producto cartesiano de pipelines COMPLETOS y los evalua END-TO-END sobre
UN set de preguntas comun (el set ampliado de la sesion), RECALCULANDO desde cero:
los valores de Fase A solo SELECCIONAN finalistas; aqui se mide todo en igualdad
de condiciones. Corona por comparacion pareada en la primaria (mrr) y desempata
por secundarios (latencia).

Reutiliza la misma maquinaria que los bancos (mismo splitter, embedding, FAISS,
recuperador, reranker y metrica), de modo que el numero sea comparable.

v1: cruza splitter x embedding x retriever x reranker. La metadata se mantiene en
su configuracion elegida (alpha del enriquecimiento es una extension v2); en los
documentos vistos la metadata casi siempre resuelve a baseline (alpha=1.0), que
es exactamente "sin metadata", por lo que v1 cubre el caso dominante.
"""
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS

from ..corpus import Sesion
from ..estadistica import bootstrap_ic, comparacion_pareada
from ..evaluacion import obtener_base_semantica, obtener_embedding_juez
from ..fabricas import construir_embedding
from ..metricas import (
    PESO_EXACTITUD,
    evaluar_recuperacion,
    gramas_de,
    similitudes_semanticas,
    vector_normalizado,
)
from ..recuperacion import construir_recuperador
from ..bancos.embeddings_banco import _construir_splitter
from ..bancos.rerankers import aplicar_reranker, K_CANDIDATOS, K_FINAL

# Bancos que v1 cruza end-to-end (metadata se fija; ver docstring).
BANCOS_FACTORIAL = ["splitter", "embedding", "retriever", "reranker"]


def _cfg(combo: dict, banco: str) -> dict:
    """Config ejecutable del finalista de un banco dentro del combo (o {})."""
    f = combo.get(banco) or {}
    return f.get("config") or {}


def _norm_splitter(cfg: dict) -> dict:
    """Normaliza las claves del splitter (finalista usa 'splitter'; el builder 'name')."""
    return {
        "name": cfg.get("name") or cfg.get("splitter", "recursive"),
        "chunk_size": cfg.get("chunk_size", 1000),
        "chunk_overlap": cfg.get("chunk_overlap", cfg.get("overlap", 100)),
    }


def evaluar_pipeline_completo(sesion: Sesion, combo: dict,
                              juez: Any, cache: dict) -> dict:
    """
    Monta UN pipeline completo desde un combo de finalistas y lo evalua sobre
    sesion.preguntas_test (el set comun ampliado). Devuelve la serie por pregunta
    de la primaria (mrr) + agregados. Reutiliza exactamente la receta del banco de
    rerankers (split -> embed -> FAISS -> recuperar -> rerank -> medir).
    """
    cfg_spl = _cfg(combo, "splitter") or sesion.config_splitter or {}
    cfg_emb = (_cfg(combo, "embedding") or sesion.config_embedding
               or {"model": "text-embedding-3-small", "provider": "openai"})
    cfg_ret = (_cfg(combo, "retriever") or sesion.config_retriever
               or {"retriever_name": "similarity", "params": {}})
    cfg_rer = _cfg(combo, "reranker") or {"provider": "none", "model": "none"}

    splitter = _construir_splitter(_norm_splitter(cfg_spl))
    chunks = splitter.split_documents(sesion.documentos)
    modelo = construir_embedding({"model": cfg_emb.get("model", "text-embedding-3-small"),
                                  "provider": cfg_emb.get("provider", "openai")})
    vectorstore = FAISS.from_documents(chunks, modelo)

    nombre_ret = cfg_ret.get("retriever_name") or cfg_ret.get("nombre", "similarity")
    k = int(cfg_ret.get("k", K_CANDIDATOS))
    recuperador = construir_recuperador(nombre_ret, cfg_ret.get("params", {}),
                                        vectorstore, chunks, k)

    grams = [gramas_de(q.get("source_context", "")) for q in sesion.preguntas_test]
    emb_obj = [vector_normalizado(q.get("source_context", ""), juez, cache)
               for q in sesion.preguntas_test]
    base_sem = obtener_base_semantica(juez)   # suelo calibrado del árbitro

    mrrs, relevancias, recalls, latencias = [], [], [], []
    usar_rer = str(cfg_rer.get("provider", "none")) not in ("none", "None", "")
    for i, q in enumerate(sesion.preguntas_test):
        docs = recuperador.invoke(q.get("question", ""))
        if usar_rer:
            docs, lat = aplicar_reranker(cfg_rer["provider"], cfg_rer["model"],
                                         q.get("question", ""), docs, K_FINAL)
        else:
            docs, lat = docs[:K_FINAL], 0.0
        textos = [getattr(d, "page_content", "") or "" for d in docs]
        sims = (similitudes_semanticas(textos, emb_obj[i], juez, cache,
                                       base=base_sem)
                if docs else None)
        nota = evaluar_recuperacion(docs, grams[i], sims=sims,
                                    peso_exactitud=PESO_EXACTITUD)
        mrrs.append(nota["mrr"])
        relevancias.append(nota["relevancia"])
        recalls.append(nota["recuperacion_pasaje"])
        latencias.append(lat)

    return {
        "serie_mrr": mrrs,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "relevancia_media": float(np.mean(relevancias)) if relevancias else 0.0,
        "recuperacion_pasaje": float(np.mean(recalls)) if recalls else 0.0,
        "latencia": float(np.mean(latencias)) if latencias else 0.0,
        "n": len(mrrs),
    }


def etiqueta_combo(combo: dict) -> str:
    return " · ".join(f"{b}:{(combo.get(b) or {}).get('etiqueta', '?')}"
                      for b in combo)


def _coste_combo(combo: dict) -> float:
    """Coste API del pipeline (suma de componentes; local=0)."""
    total = 0.0
    for fin in (combo or {}).values():
        cfg = (fin or {}).get("config") or {}
        total += float(cfg.get("cost_per_1k_tokens") or cfg.get("cost_per_1k_queries") or 0.0)
    return total


def coronar(resultados: list[dict]) -> dict:
    """
    Elige el campeon entre pipelines ya evaluados. PURA y testeable.

    Lider por mrr medio. Entran en EMPATE los rivales cuya desventaja frente al
    lider NO sea concluyente (comparacion pareada sobre la serie por pregunta).
    Dentro del empate gana el de MENOR COSTE y, a igualdad, menor latencia:
    si la calidad es estadisticamente indistinguible, pagar mas no se justifica
    (la latencia sola era vacua sin reranker: todas 0.0). Devuelve campeon,
    ranking, la comparacion lider-vs-2o y el IC del campeon.
    """
    if not resultados:
        return {"campeon": None, "ranking": [], "comparacion": None, "ic": None}
    orden = sorted(resultados, key=lambda r: -r.get("mrr", 0.0))
    lider = orden[0]
    empate, comp_vs2, comp_clon, clones = [lider], None, None, []
    for r in orden[1:]:
        comp = comparacion_pareada(lider["serie_mrr"], r["serie_mrr"],
                                   semilla=f"faseb|{r.get('etiqueta', '?')}")
        comp = {**comp, "rival": r.get("etiqueta")}
        if comp.get("identicos"):
            # Clon del lider (misma serie por pregunta; tipico k10=k5 en MRR).
            # No es un rival informativo: la comparacion del informe debe ser
            # contra el mejor NO-clon, o "prob 0.0 vs su gemelo" desinforma.
            clones.append(r.get("etiqueta"))
            comp_clon = comp_clon or comp
        elif comp_vs2 is None:
            comp_vs2 = comp
        if comp.get("concluyente"):
            break
        empate.append(r)
    if comp_vs2 is None:            # todos los rivales eran clones (o no habia)
        comp_vs2 = comp_clon
    campeon = min(empate, key=lambda r: (_coste_combo(r.get("combo")),
                                         r.get("latencia", 0.0)))
    ic = bootstrap_ic(campeon["serie_mrr"], semilla="faseb|campeon") \
        if campeon.get("serie_mrr") else None
    return {"campeon": campeon, "ranking": orden, "comparacion": comp_vs2,
            "ic": ic, "empate": [r.get("etiqueta") for r in empate],
            "clones_del_lider": clones}


def correr_fase_b(sesion: Sesion, candidatos: list[dict]) -> dict:
    """
    Evalua todos los candidatos end-to-end sobre el set comun y corona. Guarda
    cada pipeline: si uno falla (p. ej. un proveedor no instalado), lo salta.
    """
    if not candidatos:
        return {"campeon": None, "ranking": [], "n_evaluados": 0, "candidatos": 0}
    juez = obtener_embedding_juez()
    cache: dict = {}
    resultados = []
    print(f"[fase_b] Final factorial: {len(candidatos)} pipeline(s) sobre "
          f"{len(sesion.preguntas_test)} preguntas de test comunes.")
    for combo in candidatos:
        et = etiqueta_combo(combo)
        try:
            r = evaluar_pipeline_completo(sesion, combo, juez, cache)
            r["combo"], r["etiqueta"] = combo, et
            resultados.append(r)
            print(f"[fase_b]   {et} -> mrr={r['mrr']:.3f} (n={r['n']})")
        except Exception as e:  # noqa: BLE001
            print(f"[fase_b]   pipeline '{et}' fallo: {e}")
    veredicto = coronar(resultados)
    veredicto["n_evaluados"] = len(resultados)
    veredicto["candidatos"] = len(candidatos)
    if veredicto["campeon"]:
        print(f"[fase_b] CAMPEON: {veredicto['campeon']['etiqueta']} "
              f"(mrr={veredicto['campeon']['mrr']:.3f})")
    return veredicto


def producto_finalistas(certezas: dict, max_pipelines: int = 64,
                        max_por_banco: int = 3) -> list[dict]:
    """
    FASE B (nucleo): producto cartesiano de los FINALISTAS de cada banco -> lista
    de pipelines candidatos a evaluar end-to-end en terreno comun.

    - Banco RESUELTO  -> aporta solo su lider (1 opcion).
    - Banco EN EMPATE -> aporta sus finalistas (acotados a `max_por_banco` por
      valor; el lider siempre entra).
    Guardarrail: si el producto excede `max_pipelines` (64: con los 3 bancos
    habituales, 3^3=27, no se activa nunca), colapsa a su lider el banco de IC
    MAS ESTRECHO — el que menos se juega: sus finalistas son los mas parecidos
    entre si, asi que colapsarlo pierde la menor informacion. Cada colapso se
    anuncia en el log.

    Cada pipeline es {banco: config_finalista}, listo para ensamblar y correr.
    """
    def top(fins: list) -> list:
        fins = [f for f in fins if f.get("etiqueta")]
        # lider primero; el resto por valor descendente
        return sorted(fins, key=lambda f: (not f.get("es_lider"),
                                           -(f.get("valor") or 0.0)))[:max_por_banco]

    opciones = {}
    for banco, cer in certezas.items():
        if not cer.get("disponible"):
            continue
        fins = top(cer.get("finalistas") or [])
        if fins:
            opciones[banco] = fins

    def tamano(op: dict) -> int:
        n = 1
        for b in op:
            n *= len(op[b])
        return n

    # Guardarrail (>64): colapsar a su lider el banco de IC mas estrecho.
    def _amplitud(banco: str) -> float:
        ic = (certezas.get(banco, {}) or {}).get("ic_primaria") or {}
        a = ic.get("amplitud")
        return a if a is not None else float("inf")

    op = {b: list(v) for b, v in opciones.items()}
    while tamano(op) > max_pipelines:
        candidatos = [b for b in op if len(op[b]) > 1]
        if not candidatos:
            break
        antes = tamano(op)
        b = min(candidatos, key=_amplitud)
        lider = next((f for f in op[b] if f.get("es_lider")), op[b][0])
        op[b] = [lider]
        print(f"[fase_b] Guardarrail: producto {antes} > {max_pipelines}; "
              f"colapso '{b}' a su lider (IC mas estrecho, "
              f"ampl={_amplitud(b)}) -> {tamano(op)} pipeline(s).")

    combos = [{}]
    for b in op:
        combos = [{**c, b: f} for c in combos for f in op[b]]

    # P3 (perfil->plan): bajo un retriever LEXICO (bm25) el embedding no influye
    # en la recuperacion; dos combos que solo difieren en embedding son el MISMO
    # pipeline. Deduplicamos para no evaluar (ni coronar) duplicados.
    def _clave(c: dict) -> tuple:
        ret = ((c.get("retriever") or {}).get("config") or {})
        lexico = (ret.get("nombre") or ret.get("retriever_name")) == "bm25"
        return tuple((b, (f or {}).get("etiqueta"))
                     for b, f in sorted(c.items())
                     if not (lexico and b == "embedding"))
    vistos, unicos = set(), []
    for c in combos:
        k = _clave(c)
        if k not in vistos:
            vistos.add(k)
            unicos.append(c)
    if len(unicos) < len(combos):
        print(f"[fase_b] Dedupe lexico: {len(combos)} -> {len(unicos)} pipeline(s) "
              "(bajo bm25 el embedding no influye).")
    return unicos


