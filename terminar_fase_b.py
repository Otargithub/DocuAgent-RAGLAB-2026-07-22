"""
Reanuda la FASE B de una sesión que se interrumpió después de la Fase A.

Cuándo sirve: la sesión pagó todos los runs de los bancos (sus experimentos
están en `rag_memory` con su run_id) pero cascó antes de coronar. Este script
NO repite la Fase A: localiza la última sesión del documento, reconstruye la
sesión con el examen CONGELADO (mismas preguntas contra las que midieron los
bancos), lee las certezas de memoria, arma el producto de finalistas y corre la
final end-to-end.

Uso:
    python terminar_fase_b.py pdfs/test_14.pdf
    python terminar_fase_b.py pdfs/test_14.pdf --bancos splitter,embedding,retriever
    python terminar_fase_b.py pdfs/test_14.pdf --persistir     # registra la coronación
    python terminar_fase_b.py pdfs/test_14.pdf --run-id 20260705T...  # sesión concreta
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="Reanudar la Fase B desde memoria")
    ap.add_argument("pdf", help="PDF de la sesión interrumpida")
    ap.add_argument("--bancos", default="splitter,embedding,retriever",
                    help="bancos del plan original (coma-separados)")
    ap.add_argument("--run-id", default=None,
                    help="run_id concreto; por defecto, la última sesión del documento")
    ap.add_argument("--persistir", action="store_true",
                    help="registrar la decisión coronada en memoria (por defecto, solo informa)")
    args = ap.parse_args()

    from rag_lab.corpus import hash_pdf, preparar_sesion
    from rag_lab.memory import Memoria
    from rag_lab.bancos import CLAVE_DECISION
    from rag_lab.supervisor.fase_b import (BANCOS_FACTORIAL, correr_fase_b,
                                           producto_finalistas)

    bancos = [b.strip() for b in args.bancos.split(",") if b.strip()]
    mem = Memoria()
    doc_hash = hash_pdf(args.pdf) or ""

    # ---- 1) Localizar la sesión interrumpida -------------------------------
    run_id = args.run_id
    if run_id is None:
        for s in reversed(mem._leer().get("sesiones", [])):
            if s.get("doc_hash") == doc_hash:
                run_id = s.get("run_id")
                break
    assert run_id, "No hay ninguna sesión registrada para este documento."
    n_exps = len(mem.experimentos_de_documento(run_id))
    print(f"[reanudar] Sesión {run_id} | {n_exps} experimentos en memoria.")
    assert n_exps, ("La sesión no tiene experimentos guardados: aquí no hay nada "
                    "que reanudar; relanza la corrida completa.")

    # ---- 2) Certezas desde memoria (la misma verdad que usaría el meta) ----
    certezas = {b: mem.certeza(run_id, b, CLAVE_DECISION[b])
                for b in BANCOS_FACTORIAL if b in bancos}
    print("\n[reanudar] Certezas recuperadas:")
    for b, c in certezas.items():
        if not c.get("disponible"):
            print(f"  {b}: SIN DATOS")
            continue
        print(f"  {b}: {'RESUELTO' if c.get('resuelto') else 'EN DISPUTA'} | "
              f"lider={c.get('ganador')} | finalistas="
              f"{[f.get('etiqueta') for f in (c.get('finalistas') or [])]}")

    # ---- 3) Sesión con el examen CONGELADO (mismas preguntas) --------------
    congeladas = mem.preguntas_congeladas(doc_hash)
    assert congeladas, ("El documento no tiene examen congelado: sin él la Fase B "
                        "no mediría contra las mismas preguntas que la Fase A.")
    sesion = preparar_sesion(args.pdf, preguntas_predefinidas=congeladas)
    print(f"[reanudar] Examen congelado: {len(sesion.preguntas_test)} preguntas.")

    # ---- 4) Producto de finalistas + final end-to-end ----------------------
    candidatos = producto_finalistas(certezas)
    if len(candidatos) <= 1:
        print("[reanudar] Sin empates reales que dirimir: el líder de cada banco "
              "ya es el stack. Nada que coronar.")
        return
    veredicto = correr_fase_b(sesion, candidatos)

    campeon = veredicto.get("campeon") or {}
    comp = veredicto.get("comparacion") or {}
    print(f"\n[reanudar] STACK FINAL (coronado end-to-end): "
          f"{campeon.get('etiqueta')} | mrr={campeon.get('mrr')} | "
          f"concluyente={comp.get('concluyente')}")

    # ---- 5) Persistencia opcional (mismo formato que el meta) --------------
    if args.persistir and campeon:
        stack = {
            "stack_recomendado": {
                banco: {"etiqueta": fin.get("etiqueta"), "config": fin.get("config"),
                        "resuelto": True, "en_disputa": False, "via": "fase_b"}
                for banco, fin in (campeon.get("combo") or {}).items()},
            "fase_b": {"campeon": campeon.get("etiqueta"), "mrr": campeon.get("mrr"),
                       "ic": veredicto.get("ic"), "empate_final": veredicto.get("empate"),
                       "n_pipelines": veredicto.get("n_evaluados"),
                       "comparacion": comp},
            "confianza_global": "media" if comp.get("concluyente") else "baja",
            "reanudado": True,
        }
        mem.registrar_decision(run_id, stack)
        print("[reanudar] Decisión coronada PERSISTIDA en memoria.")
    elif campeon:
        print("[reanudar] (No persistido: relanza con --persistir si lo apruebas.)")


if __name__ == "__main__":
    main()
