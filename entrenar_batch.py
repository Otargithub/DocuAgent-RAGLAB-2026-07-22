"""
entrenar_batch.py — automatiza el entrenamiento por lotes y REANUDA donde lo dejó.

Idea
----
Procesa una carpeta de PDFs acumulando experiencia en la memoria unificada. La
reanudación NO necesita un fichero de progreso aparte: la memoria es la fuente de
verdad. Un documento ya estudiado y COMPLETADO (misma huella de contenido + con
decisión registrada) se OMITE automáticamente (lo hace `meta.ejecutar`). Una
corrida cortada a la mitad (sin decisión, p. ej. un corte de Colab) no se marca
como hecha, así que se reprocesa. Conclusión: para reanudar, basta volver a
lanzar el mismo lote.

Pensado para tandas desatendidas en Colab:
  - resiliente: si un PDF falla, lo registra y sigue con el siguiente;
  - acotable: `maximo` (documentos NUEVOS por corrida) y `presupuesto_min`
    (minutos de pared) para parar limpio antes de que Colab corte;
  - observable: imprime progreso y, opcionalmente, escribe un log JSONL por
    documento que puedes seguir con `tail -f`.

Uso desde el notebook
----------------------
    from entrenar_batch import entrenar_carpeta, pendientes
    from rag_lab import Plan

    # ¿Qué queda por hacer? (no gasta nada)
    print(pendientes("pdfs"))

    # Procesa hasta 20 documentos nuevos o 50 min, lo que llegue antes:
    resumen = entrenar_carpeta("pdfs", plan=Plan(n_runs=3),
                               maximo=20, presupuesto_min=50,
                               log_path="rag_memory/entrenamiento.jsonl")
    print(resumen)

Uso por CLI (desde la raíz del proyecto, con rag_lab importable)
----------------------------------------------------------------
    python entrenar_batch.py pdfs --maximo 20 --presupuesto-min 50 --n-runs 3
"""

from __future__ import annotations

import glob
import json
import pathlib
import time
import traceback
from datetime import datetime
from typing import Any, Callable

from rag_lab import MetaSupervisor, Plan
from rag_lab.memory import Memoria
from rag_lab.corpus import hash_pdf


def listar_pdfs(carpeta: str, patron: str = "*.pdf", recursivo: bool = False) -> list[str]:
    """Lista los PDFs de la carpeta en orden determinista (por nombre)."""
    base = pathlib.Path(carpeta)
    rutas = base.rglob(patron) if recursivo else base.glob(patron)
    return sorted(str(p) for p in rutas if p.is_file())


def pendientes(carpeta: str, memoria: Memoria | None = None,
              patron: str = "*.pdf", recursivo: bool = False) -> dict:
    """
    Ensayo en seco: clasifica el lote en 'hechos' y 'pendientes' SIN procesar nada
    (solo hash de contenido + consulta a memoria). Útil para saber qué queda.
    """
    memoria = memoria or Memoria()
    hechos, por_hacer = [], []
    for pdf in listar_pdfs(carpeta, patron, recursivo):
        ya = memoria.documento_estudiado(doc_hash=hash_pdf(pdf), pdf_path=pdf)
        (hechos if ya else por_hacer).append(pdf)
    return {"total": len(hechos) + len(por_hacer),
            "hechos": hechos, "pendientes": por_hacer,
            "n_hechos": len(hechos), "n_pendientes": len(por_hacer)}


def _anotar_log(log_path: str | None, registro: dict) -> None:
    """Añade una línea JSON al log (si se pidió). Escritura simple append."""
    if not log_path:
        return
    p = pathlib.Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def entrenar_carpeta(carpeta: str,
                    meta: MetaSupervisor | None = None,
                    memoria: Memoria | None = None,
                    plan: Plan | dict | None = None,
                    instruccion: str = "",
                    maximo: int | None = None,
                    presupuesto_min: float | None = None,
                    patron: str = "*.pdf",
                    recursivo: bool = False,
                    forzar: bool = False,
                    revisor: Callable[[dict], dict] | None = None,
                    log_path: str | None = None,
                    al_terminar_doc: Callable[[dict], None] | None = None) -> dict:
    """
    Procesa los PDFs de `carpeta`, acumulando en la memoria unificada, y reanuda
    saltando los ya estudiados.

    Parámetros
    ----------
    carpeta         : ruta con los PDFs.
    meta / memoria  : reutilizables; si no se pasan, se crean (memoria persistente
                      en CONFIG.fichero_memoria, meta sobre esa memoria).
    plan            : Plan o dict para todos los documentos (modo 'plan fijado').
                      Si es None, cada documento se planifica de forma autónoma.
    instruccion     : instrucción en lenguaje natural (solo si plan es None).
    maximo          : tope de documentos NUEVOS a procesar en ESTA corrida (los
                      omitidos no cuentan). None = sin tope.
    presupuesto_min : minutos de pared máximos; antes de empezar un documento nuevo
                      se comprueba y, si se excede, se para limpio. None = sin tope.
    patron/recursivo: filtro de ficheros.
    forzar          : reprocesa aunque ya estén estudiados (ignora la reanudación).
    revisor         : callback HITL; None = aprueba el plan automáticamente.
    log_path        : si se indica, escribe una línea JSONL por documento (para
                      seguir una tanda nocturna con `tail -f`).
    al_terminar_doc : callback opcional que recibe el registro de cada documento.

    Devuelve un resumen con conteos, errores y lo que queda pendiente.
    """
    memoria = memoria or Memoria()
    meta = meta or MetaSupervisor(memoria)
    plan_dict = plan.como_dict() if isinstance(plan, Plan) else plan

    todos = listar_pdfs(carpeta, patron, recursivo)
    t0 = time.monotonic()
    res = {"carpeta": carpeta, "total_en_carpeta": len(todos),
           "procesados": 0, "omitidos": 0, "fallidos": 0,
           "errores": {}, "detenido_por": None, "inicio": datetime.now().isoformat(timespec="seconds")}

    if not todos:
        print(f"[entrenar] No hay PDFs en '{carpeta}' (patrón {patron}).")
        res["detenido_por"] = "sin_documentos"
        return res

    print(f"[entrenar] {len(todos)} documentos en la carpeta. "
          f"maximo={maximo or '∞'} | presupuesto={presupuesto_min or '∞'} min | "
          f"plan={'fijado' if plan_dict else 'autónomo'}")

    for i, pdf in enumerate(todos, 1):
        # ¿Presupuesto de tiempo agotado? Paramos ANTES de empezar uno nuevo.
        if presupuesto_min is not None and (time.monotonic() - t0) / 60.0 >= presupuesto_min:
            print(f"[entrenar] Presupuesto de {presupuesto_min} min agotado; paro limpio.")
            res["detenido_por"] = "presupuesto_tiempo"
            break
        # ¿Tope de documentos nuevos alcanzado?
        if maximo is not None and res["procesados"] >= maximo:
            print(f"[entrenar] Alcanzado el tope de {maximo} documentos nuevos; paro.")
            res["detenido_por"] = "maximo"
            break

        etiqueta = pathlib.Path(pdf).name
        print(f"\n[entrenar] [{i}/{len(todos)}] {etiqueta}")
        registro = {"momento": datetime.now().isoformat(timespec="seconds"), "pdf": pdf}
        try:
            stack = meta.ejecutar(pdf, instruccion=instruccion, plan=plan_dict,
                                  revisor=revisor, forzar=forzar)
            if stack.get("_ya_estudiado") and not forzar:
                res["omitidos"] += 1
                registro["estado"] = "omitido"
                print(f"   omitido (ya estudiado, run {stack.get('_run_id_original')})")
            else:
                res["procesados"] += 1
                registro["estado"] = "procesado"
                registro["confianza"] = stack.get("confianza_global")
                registro["stack"] = list(stack.get("stack_recomendado", {}).keys())
                print(f"   procesado | confianza={stack.get('confianza_global')} "
                      f"| stack={registro['stack']}")
        except Exception as error:  # noqa: BLE001
            res["fallidos"] += 1
            res["errores"][pdf] = f"{type(error).__name__}: {error}"
            registro["estado"] = "fallido"
            registro["error"] = res["errores"][pdf]
            print(f"   ERROR: {type(error).__name__}: {error}")
            traceback.print_exc()

        _anotar_log(log_path, registro)
        if al_terminar_doc:
            try:
                al_terminar_doc(registro)
            except Exception:  # noqa: BLE001 — un callback roto no debe tumbar la tanda
                pass

    # Recuento de lo que aún queda pendiente tras esta corrida.
    res["pendientes_restantes"] = pendientes(carpeta, memoria, patron, recursivo)["n_pendientes"]
    res["duracion_min"] = round((time.monotonic() - t0) / 60.0, 2)
    res["memoria"] = memoria.resumen()
    print(f"\n[entrenar] Fin. procesados={res['procesados']} omitidos={res['omitidos']} "
          f"fallidos={res['fallidos']} | pendientes={res['pendientes_restantes']} "
          f"| {res['duracion_min']} min")
    return res


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Entrenamiento por lotes de rag_lab, reanudable.")
    ap.add_argument("carpeta", help="Carpeta con los PDFs.")
    ap.add_argument("--maximo", type=int, default=None, help="Tope de documentos nuevos por corrida.")
    ap.add_argument("--presupuesto-min", type=float, default=None, help="Minutos de pared máximos.")
    ap.add_argument("--bancos", nargs="+", default=None,
                    help="Subconjunto de bancos (splitter embedding metadata retriever reranker).")
    ap.add_argument("--n-runs", type=int, default=None, help="Repeticiones por banco (1-3).")
    ap.add_argument("--instruccion", default="", help="Instrucción en lenguaje natural (si no fijas plan).")
    ap.add_argument("--patron", default="*.pdf", help="Patrón de ficheros (por defecto *.pdf).")
    ap.add_argument("--recursivo", action="store_true", help="Buscar PDFs en subcarpetas.")
    ap.add_argument("--forzar", action="store_true", help="Reprocesar aunque ya estén estudiados.")
    ap.add_argument("--log", default=None, help="Ruta de un log JSONL por documento.")
    ap.add_argument("--solo-pendientes", action="store_true", help="Solo listar qué queda; no procesa.")
    args = ap.parse_args()

    if args.solo_pendientes:
        info = pendientes(args.carpeta, patron=args.patron, recursivo=args.recursivo)
        print(f"{info['n_pendientes']} pendientes de {info['total']} "
              f"({info['n_hechos']} ya hechos):")
        for p in info["pendientes"]:
            print("  -", p)
        return

    # Construir el plan solo si se especificó algo; si no, modo autónomo.
    plan = None
    if args.bancos or args.n_runs:
        campos = {}
        if args.bancos:
            campos["bancos"] = args.bancos
        if args.n_runs:
            campos["n_runs"] = args.n_runs
        plan = Plan(**campos)

    entrenar_carpeta(args.carpeta, plan=plan, instruccion=args.instruccion,
                     maximo=args.maximo, presupuesto_min=args.presupuesto_min,
                     patron=args.patron, recursivo=args.recursivo,
                     forzar=args.forzar, log_path=args.log)


if __name__ == "__main__":
    _main()
