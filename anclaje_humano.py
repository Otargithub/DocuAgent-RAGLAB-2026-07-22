"""
Anclaje humano de la métrica — versión script.

Valida que la nota automática (α·léxico + (1−α)·semántico con base calibrada)
coincide con TU juicio: etiquetas a ciegas la relevancia de fragmentos
recuperados por un stack concreto y el script mide el acuerdo (Spearman),
barre el umbral contra tu criterio y compara MRR humano vs máquina.

Las etiquetas se guardan en rag_memory (colección `anclaje_humano`), son
reanudables entre sesiones y compatibles con `anclaje_humano.ipynb`.

Uso:
    python anclaje_humano.py pdfs/test_13.pdf
    python anclaje_humano.py pdfs/test_13.pdf --stack "recursive_1000,granite-311m-r2,ensemble_k10_w0.5"
    python anclaje_humano.py pdfs/test_13.pdf --n 10 --criterio 2
    python anclaje_humano.py pdfs/test_13.pdf --analizar-solo
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
# Funciones puras (testeables sin entorno)
# --------------------------------------------------------------------------- #
def parse_splitter(etiqueta: str) -> dict:
    """'recursive_1000' -> config del banco; también 'page' y 'semantic'."""
    if etiqueta in ("page", "semantic"):
        return {"splitter": etiqueta, "chunk_size": 0, "chunk_overlap": 0}
    m = re.fullmatch(r"(recursive|token|character)_(\d+)", etiqueta)
    if not m:
        raise ValueError(f"Splitter no reconocido: {etiqueta!r}")
    size = int(m.group(2))
    return {"splitter": m.group(1), "chunk_size": size,
            "chunk_overlap": int(size * 0.1)}


def parse_retriever(etiqueta: str) -> dict:
    """'ensemble_k10_w0.5' | 'mmr_k5_l0.5' | 'similarity_k5' | 'bm25_k3'."""
    m = re.fullmatch(r"ensemble_k(\d+)_w([\d.]+)", etiqueta)
    if m:
        return {"nombre": "ensemble", "k": int(m.group(1)),
                "params": {"w_bm25": float(m.group(2))}}
    m = re.fullmatch(r"mmr_k(\d+)_l([\d.]+)", etiqueta)
    if m:
        return {"nombre": "mmr", "k": int(m.group(1)),
                "params": {"lambda_mult": float(m.group(2)), "fetch_k": 20}}
    m = re.fullmatch(r"(similarity|bm25)_k(\d+)", etiqueta)
    if m:
        return {"nombre": m.group(1), "k": int(m.group(2)), "params": {}}
    raise ValueError(f"Retriever no reconocido: {etiqueta!r}")


# Spearman y rangos medios: UNIFICADOS en rag_lab.estadistica (fase 1 del
# refactor). Se re-exportan aqui para que todas las celdas y tests que hacen
# `from anclaje_humano import spearman` sigan funcionando tal cual.
from rag_lab.estadistica import _rangos_medios, spearman  # noqa: F401


def auc(nota, humano_bin) -> float:
    """P(nota de un par relevante > nota de uno irrelevante) — Mann-Whitney.
    Para ELEGIR la forma de la nota (alfa, max, ...) es mejor vara que
    Spearman: ignora el orden interno de los bloques (que no importa) y no
    premia los empates degenerados (lex=0 en masa 'acierta' el empate del
    juicio humano y Spearman le regala correlacion por ello)."""
    nota = np.asarray(nota, float); h = np.asarray(humano_bin, int)
    pos, neg = nota[h == 1], nota[h == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    mayor = (pos[:, None] > neg[None, :]).sum()
    empate = (pos[:, None] == neg[None, :]).sum()
    return float((mayor + 0.5 * empate) / (len(pos) * len(neg)))


def barrido_umbral(rel_auto, rel_humano_bin):
    """[(umbral, precision, recall, f1)] en pasos de 0.05, y el óptimo por F1."""
    a = np.asarray(rel_auto, float)
    h = np.asarray(rel_humano_bin, bool)
    filas, mejor = [], (0.5, -1.0)
    for u in np.arange(0.05, 0.96, 0.05):
        pred = a >= u
        tp = float(np.sum(pred & h)); fp = float(np.sum(pred & ~h))
        fn = float(np.sum(~pred & h))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        filas.append((round(float(u), 2), round(prec, 3), round(rec, 3), round(f1, 3)))
        if f1 > mejor[1]:
            mejor = (round(float(u), 2), f1)
    # MESETA: umbrales empatados (±0.005) con el mejor F1. Reportar solo el
    # minimo era engañoso ("optimo 0.3" cuando 0.3-0.95 daban lo mismo).
    meseta = [f[0] for f in filas if abs(f[3] - mejor[1]) <= 0.005]
    return filas, (mejor[0], mejor[1], (min(meseta), max(meseta)))


def mrr_por_pregunta(etiquetas: list[dict], criterio: int, umbral: float):
    """[(qi, mrr_humano, mrr_maquina)] a partir de los registros etiquetados."""
    por_q = defaultdict(list)
    for e in etiquetas:
        por_q[e["qi"]].append(e)
    filas = []
    for qi, lst in sorted(por_q.items()):
        lst = sorted(lst, key=lambda e: e["fi"])
        mh = next((1.0 / (e["fi"] + 1) for e in lst if e["humano"] >= criterio), 0.0)
        ma = next((1.0 / (e["fi"] + 1) for e in lst if e["rel_auto"] >= umbral), 0.0)
        filas.append((qi, mh, ma))
    return filas


# --------------------------------------------------------------------------- #
# Flujo principal (requiere entorno completo)
# --------------------------------------------------------------------------- #
def construir_pipeline(pdf: str, stack: dict):
    from langchain_community.vectorstores import FAISS

    from rag_lab.corpus import preparar_sesion, hash_pdf
    from rag_lab.fabricas import construir_embedding
    from rag_lab.memory import Memoria
    from rag_lab.recuperacion import construir_recuperador
    from rag_lab.bancos.splitters import _construir_splitter_banco
    from rag_lab.config import CONFIG

    mem = Memoria()
    # Resolución de ruta (mismo criterio que las tools del agente): tal cual,
    # PDF_PATH del entorno, y la carpeta estándar de Drive. Una ruta relativa
    # desde otro directorio hacía hash_pdf("") y el assert culpaba al examen.
    import os
    candidatas = [pdf,
                  os.path.join(os.environ.get("PDF_PATH", ""), os.path.basename(pdf)),
                  os.path.join("/content/drive/MyDrive/rag_lab/pdfs",
                               os.path.basename(pdf))]
    pdf = next((c for c in candidatas if c and os.path.exists(c)), pdf)
    doc_hash = hash_pdf(pdf) or ""
    congeladas = mem.preguntas_congeladas(doc_hash)
    assert congeladas, (f"Sin examen congelado para {os.path.basename(pdf)} "
                        f"(hash={doc_hash[:10] or 'VACIO — ¿ruta correcta?'}): "
                        "verifica la ruta del PDF o corre antes una sesión.")
    sesion = preparar_sesion(pdf, preguntas_predefinidas=congeladas)

    spl = stack["splitter"]
    modelo_spl = (construir_embedding(CONFIG.embedding_neutro)
                  if spl["splitter"] == "semantic" else None)
    splitter = _construir_splitter_banco(spl["splitter"], spl["chunk_size"],
                                         spl["chunk_overlap"], modelo_spl)
    chunks = splitter.split_documents(sesion.documentos)
    modelo = construir_embedding(stack["embedding"])
    vs = FAISS.from_documents(chunks, modelo)
    ret = stack["retriever"]
    recuperador = construir_recuperador(ret["nombre"], ret["params"], vs,
                                        chunks, ret["k"])
    print(f"[anclaje] Pipeline: {len(chunks)} chunks | "
          f"{len(sesion.preguntas_test)} preguntas congeladas.")
    return mem, doc_hash, sesion, recuperador


def preparar_pares(sesion, recuperador, n_preguntas: int, semilla: int):
    from rag_lab.evaluacion import obtener_base_semantica, obtener_embedding_juez
    from rag_lab.metricas import (PESO_EXACTITUD, cobertura, gramas_de,
                                  recalibrar_similitud, vector_normalizado)

    rng = np.random.default_rng(semilla)
    idx = sorted(rng.choice(len(sesion.preguntas_test),
                            min(n_preguntas, len(sesion.preguntas_test)),
                            replace=False).tolist())
    juez = obtener_embedding_juez()
    base = obtener_base_semantica(juez)
    alpha = float(PESO_EXACTITUD)
    print(f"[anclaje] α={alpha} | base calibrada={base} | muestra={len(idx)} preguntas")

    cache, pares = {}, []
    for qi in idx:
        q = sesion.preguntas_test[qi]
        cita = q.get("source_context", "")
        grams = gramas_de(cita)
        emb_cita = vector_normalizado(cita, juez, cache)
        for fi, d in enumerate(recuperador.invoke(q.get("question", ""))):
            texto = d.page_content or ""
            lex = cobertura(gramas_de(texto), grams)
            v = vector_normalizado(texto, juez, cache)
            sem = recalibrar_similitud(float(v @ emb_cita), base)
            pares.append({"qi": qi, "fi": fi, "question": q.get("question", ""),
                          "texto": texto, "lex": round(lex, 4), "sem": round(sem, 4),
                          "rel_auto": round(alpha * lex + (1 - alpha) * sem, 4)})
    return pares


def etiquetar(mem, doc_hash: str, etiqueta_stack: str, pares: list[dict]) -> None:
    def clave(p):
        th = hashlib.md5(p["texto"].encode()).hexdigest()[:10]
        return f"{doc_hash[:10]}|{etiqueta_stack}|q{p['qi']}|f{p['fi']}|{th}"

    ya = {e["clave"] for e in mem._leer().get("anclaje_humano", [])
          if e.get("doc_hash") == doc_hash and e.get("stack") == etiqueta_stack}
    pendientes = [p for p in pares if clave(p) not in ya]
    print(f"[anclaje] Ya etiquetados: {len(ya)} | pendientes: {len(pendientes)}\n")

    for n, p in enumerate(pendientes, 1):
        print("=" * 70)
        print(f"[{n}/{len(pendientes)}] PREGUNTA: {p['question']}")
        print("-" * 70)
        print(p["texto"][:900] + ("..." if len(p["texto"]) > 900 else ""))
        r = input("¿Responde? 0=no 1=parcial 2=sí | s=saltar q=parar > ").strip().lower()
        if r == "q":
            print("[anclaje] Pausado: el progreso queda guardado.")
            return
        if r not in ("0", "1", "2"):
            continue
        mem._anadir("anclaje_humano", {
            "clave": clave(p), "doc_hash": doc_hash, "stack": etiqueta_stack,
            "qi": p["qi"], "fi": p["fi"], "humano": int(r),
            "lex": p["lex"], "sem": p["sem"], "rel_auto": p["rel_auto"]})
    print("\n[anclaje] Muestra completa etiquetada.")


def analizar(mem, doc_hash: str, etiqueta_stack: str, criterio: int) -> None:
    from rag_lab.metricas import UMBRAL_RELEVANCIA

    E = [e for e in mem._leer().get("anclaje_humano", [])
         if e.get("doc_hash") == doc_hash and e.get("stack") == etiqueta_stack]
    if len(E) < 10:
        print(f"[anclaje] Solo {len(E)} etiquetas: etiqueta más antes de analizar.")
        return
    h = [e["humano"] for e in E]
    a = [e["rel_auto"] for e in E]

    print("\n" + "=" * 70)
    print(f"  ANÁLISIS DEL ANCLAJE  (n={len(E)} pares, stack={etiqueta_stack})")
    print("=" * 70)
    print(f"Spearman(humano, nota combinada) = {spearman(h, a):.3f}")
    print(f"Spearman(humano, solo léxico)    = {spearman(h, [e['lex'] for e in E]):.3f}")
    print(f"Spearman(humano, solo semántico) = {spearman(h, [e['sem'] for e in E]):.3f}")
    hb = [1 if x >= criterio else 0 for x in h]
    print(f"AUC(combinada)={auc(a, hb):.3f} | AUC(léxico)={auc([e['lex'] for e in E], hb):.3f}"
          f" | AUC(semántico)={auc([e['sem'] for e in E], hb):.3f}")
    print("  [AUC = P(par relevante puntúe > irrelevante); para comparar FORMAS"
          " de la nota, fíate del AUC antes que del Spearman]")

    filas, (u_opt, f1_opt, meseta) = barrido_umbral(a, [x >= criterio for x in h])
    print(f"\n{'umbral':>7} {'prec':>6} {'rec':>6} {'F1':>6}")
    for u, p, r, f1 in filas:
        marca = "  <- vigente" if abs(u - UMBRAL_RELEVANCIA) < 0.026 else ""
        print(f"{u:7.2f} {p:6.2f} {r:6.2f} {f1:6.2f}{marca}")
    lo, hi = meseta
    rango = f" (meseta {lo}-{hi})" if hi > lo else ""
    print(f"\nUmbral óptimo según tu criterio (F1): {u_opt}{rango} (F1={f1_opt:.2f})")
    if meseta[0] - 0.026 <= UMBRAL_RELEVANCIA <= meseta[1] + 0.026:
        print(f"=> El vigente ({UMBRAL_RELEVANCIA}) esta en la meseta optima.")
    elif abs(u_opt - UMBRAL_RELEVANCIA) >= 0.10:
        print(f"=> Difiere del vigente ({UMBRAL_RELEVANCIA}): considera "
              f"RAG_UMBRAL_RELEVANCIA={u_opt}")
    else:
        print(f"=> Compatible con el vigente ({UMBRAL_RELEVANCIA}).")

    filas_m = mrr_por_pregunta(E, criterio, UMBRAL_RELEVANCIA)
    mh = float(np.mean([f[1] for f in filas_m]))
    ma = float(np.mean([f[2] for f in filas_m]))
    print(f"\nMRR humano={mh:.3f} | MRR máquina={ma:.3f} | sesgo={ma - mh:+.3f}")
    print("(+) la máquina se apunta aciertos que tú no le darías; "
          "(-) descarta aciertos reales.")


def main():
    ap = argparse.ArgumentParser(description="Anclaje humano de la métrica")
    ap.add_argument("pdf", help="PDF con examen congelado")
    ap.add_argument("--stack", default="recursive_1000,granite-311m-r2,ensemble_k10_w0.5",
                    help="'splitter,embedding,retriever' por etiquetas del banco")
    ap.add_argument("--n", type=int, default=8, help="preguntas de la muestra")
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--criterio", type=int, default=1, choices=[1, 2],
                    help="'relevante' humano = etiqueta >= criterio")
    ap.add_argument("--analizar-solo", action="store_true",
                    help="salta el etiquetado y analiza lo acumulado")
    args = ap.parse_args()

    from rag_lab.bancos.embeddings_banco import EMBEDDING_REGISTRY
    spl_et, emb_et, ret_et = [x.strip() for x in args.stack.split(",")]
    stack = {"splitter": parse_splitter(spl_et),
             "embedding": EMBEDDING_REGISTRY[emb_et],
             "retriever": parse_retriever(ret_et)}
    etiqueta_stack = f"{spl_et}|{emb_et}|{ret_et}"

    mem, doc_hash, sesion, recuperador = construir_pipeline(args.pdf, stack)
    if not args.analizar_solo:
        pares = preparar_pares(sesion, recuperador, args.n, args.seed)
        etiquetar(mem, doc_hash, etiqueta_stack, pares)
    analizar(mem, doc_hash, etiqueta_stack, args.criterio)


if __name__ == "__main__":
    main()
