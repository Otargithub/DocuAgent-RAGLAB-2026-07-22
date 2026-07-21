# --------------------------------------------------------------------------- #
# GOLDEN TEST del ensamblado POST-CRUCE de la Fase B (apertura de la fase 3
# del refactor). Congela: coronacion sobre el stack (vias fase_b/fase_a),
# economico post-final dentro del empate (con el cero de bm25), payload
# fase_b, coda de la justificacion, ascenso de confianza, persistencia de la
# decision y resolucion del prior DIFERIDO de splitter contra la coronacion.
# Costuras: veredicto de la final PRE-SEMBRADO en el memo (correr_fase_b no
# se ejecuta), certezas factoriales congeladas, memoria en tmp.
# Regenerar SOLO ante cambios deseados: UPDATE_GOLDEN=1 python3 tests/test_golden_fase_b.py
# --------------------------------------------------------------------------- #
import json
import os
import sys
import tempfile
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
GOLDEN = Path(__file__).parent / "golden" / "fase_b_golden.json"

try:
    import langchain_core  # noqa: F401
except Exception:
    import importlib.abc
    import importlib.machinery

    PREF = ("langchain", "langgraph", "faiss", "sklearn", "torch",
            "transformers", "openai", "anthropic", "rank_bm25", "tiktoken",
            "sentence_transformers", "umap", "hdbscan", "pydantic",
            "numpy_financial")

    class _M(types.ModuleType):
        def __getattr__(self, n):
            if n == "__path__":
                return []
            return type(n, (), {"__init__": lambda self, *a, **k: None,
                                "__call__": lambda self, *a, **k: self,
                                "__getattr__": lambda self, nombre: self})

    class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, full, path=None, target=None):
            if full.split(".")[0].startswith(PREF):
                return importlib.machinery.ModuleSpec(full, self, is_package=True)
        def create_module(self, spec):
            return _M(spec.name)
        def exec_module(self, m):
            pass

    sys.meta_path.insert(0, _F())

os.environ["RAG_MEMORY_DIR"] = tempfile.mkdtemp(prefix="golden_fase_b_")

from rag_lab.supervisor.meta import MetaSupervisor  # noqa: E402
from rag_lab.memory import Memoria  # noqa: E402


def certezas_factorial() -> dict:
    """Dos bancos empatados (los de pendiente_fase_b) -> 4 candidatos."""
    return {
        "embedding": {
            "disponible": True, "resuelto": True, "es_empate": False,
            "ganador": "emb-pago",
            "finalistas": [
                {"etiqueta": "emb-pago", "es_lider": True,
                 "config": dict(EMB_PAGO)},
            ]},
        "splitter": {
            "disponible": True, "resuelto": False, "es_empate": True,
            "ganador": "token_250",
            "finalistas": [
                {"etiqueta": "token_250", "es_lider": True,
                 "config": {"splitter": "token", "chunk_size": 250,
                            "chunk_overlap": 25}},
                {"etiqueta": "token_500", "es_lider": False,
                 "config": {"splitter": "token", "chunk_size": 500,
                            "chunk_overlap": 50}},
            ]},
        "retriever": {
            "disponible": True, "resuelto": False, "es_empate": True,
            "ganador": "ensemble_k10_w0.7",
            "finalistas": [
                {"etiqueta": "ensemble_k10_w0.7", "es_lider": True,
                 "config": {"nombre": "ensemble", "k": 10,
                            "params": {"w_bm25": 0.7}}},
                {"etiqueta": "bm25_k10", "es_lider": False,
                 "config": {"nombre": "bm25", "k": 10, "params": {}}},
            ]},
    }


EMB_PAGO = {"model": "sintetico-pago", "provider": "openai",
            "cost_per_1k_tokens": 0.02}


def veredicto_congelado() -> dict:
    """La final ya corrida (congelada): campeon ensemble, empate con un bm25
    mas barato -> ejercita el economico post-final y el cero de bm25."""
    combo_campeon = {
        "splitter": {"etiqueta": "token_500",
                     "config": {"splitter": "token", "chunk_size": 500,
                                "chunk_overlap": 50}},
        "embedding": {"etiqueta": "emb-pago", "config": dict(EMB_PAGO)},
        "retriever": {"etiqueta": "ensemble_k10_w0.7",
                      "config": {"nombre": "ensemble", "k": 10,
                                 "params": {"w_bmz5": 0.7}}}}
    combo_barato = {
        "splitter": {"etiqueta": "token_500",
                     "config": {"splitter": "token", "chunk_size": 500,
                                "chunk_overlap": 50}},
        "embedding": {"etiqueta": "emb-pago", "config": dict(EMB_PAGO)},
        "retriever": {"etiqueta": "bm25_k10",
                      "config": {"nombre": "bm25", "k": 10, "params": {}}}}
    return {
        "campeon": {"etiqueta": "token_500|emb-pago|ensemble_k10_w0.7",
                    "combo": combo_campeon, "mrr": 0.601},
        "ranking": [
            {"etiqueta": "token_500|emb-pago|ensemble_k10_w0.7",
             "combo": combo_campeon, "mrr": 0.601},
            {"etiqueta": "token_500|emb-pago|bm25_k10",
             "combo": combo_barato, "mrr": 0.588},
            {"etiqueta": "token_250|emb-pago|ensemble_k10_w0.7",
             "combo": None, "mrr": 0.522},
        ],
        "empate": ["token_500|emb-pago|ensemble_k10_w0.7",
                   "token_500|emb-pago|bm25_k10"],
        "ic": {"ic_bajo": 0.512, "ic_alto": 0.689},
        "n_evaluados": 4,
        "comparacion": {"concluyente": True,
                        "rival": "token_250|emb-pago|ensemble_k10_w0.7"},
        "clones_del_lider": [],
    }


def correr_nodo() -> dict:
    meta = MetaSupervisor(memoria=Memoria())
    run_id = "golden_fb_1"
    sesion = types.SimpleNamespace(
        run_id=run_id, doc_hash="golden_doc", pdf_path="/tmp/golden.pdf",
        doc_type="golden", documentos=[], preguntas_test=[{"q": 1}] * 12,
        restricciones={}, verificacion=None, config_splitter={},
        config_embedding=None, config_retriever=None, perfil={})
    meta._sesiones[run_id] = sesion

    # prior DIFERIDO de splitter (la sintesis no lo resolvio): aqui debe
    # calificarse contra la CORONACION (token_500) y ACERTAR.
    meta.memoria.registrar_prediccion(run_id, "splitter", "token_500")

    # costuras
    cert = certezas_factorial()
    meta.memoria.certeza = lambda run, b, clave: cert[b]
    meta._memo()[("fase_b", run_id)] = veredicto_congelado()

    estado = {
        "run_id": run_id,
        "plan": {"objetivo": "maxima calidad", "n_runs": 2,
                 "bancos": ["splitter", "embedding", "retriever"]},
        "stack_final": {
            "stack_recomendado": {
                "splitter": {"etiqueta": "token_250", "resuelto": False,
                             "en_disputa": True, "config": {}},
                "embedding": {"etiqueta": "emb-pago", "resuelto": True,
                              "en_disputa": False, "config": dict(EMB_PAGO)},
                "retriever": {"etiqueta": "ensemble_k10_w0.7", "resuelto": False,
                              "en_disputa": True, "config": {}},
            },
            "stack_alternativo_economico": {"pre_final": {"etiqueta": "x"}},
            "pendiente_fase_b": {"splitter": ["token_250", "token_500"],
                                  "retriever": ["ensemble_k10_w0.7", "bm25_k10"]},
            "justificacion": "NARRACION-GOLDEN.",
            "confianza_global": "baja",
        },
        "_veto_fase_b": [],
    }
    salida = meta._nodo_fase_b(estado)

    stack = (salida or {}).get("stack_final") or {}
    mem = meta.memoria._leer()
    predicciones = [{k: p.get(k) for k in ("banco", "esperado", "real", "acierto")}
                    for p in mem.get("predicciones", [])]
    rec = stack.get("stack_recomendado") or {}
    return {
        "recomendado": {b: {"etiqueta": v.get("etiqueta"), "via": v.get("via"),
                            "resuelto": v.get("resuelto"),
                            "en_disputa": v.get("en_disputa")}
                        for b, v in rec.items()},
        "economico": stack.get("stack_alternativo_economico"),
        "fase_b": stack.get("fase_b"),
        "justificacion": stack.get("justificacion"),
        "confianza_global": stack.get("confianza_global"),
        "n_decisiones_registradas": len(mem.get("decisiones", [])),
        "predicciones": predicciones,
    }


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


if __name__ == "__main__":
    foto = correr_nodo()
    if os.environ.get("UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(canon(foto))
        print(f"GOLDEN generado: {GOLDEN}")
    else:
        esperado, actual = GOLDEN.read_text(), canon(foto)
        if actual != esperado:
            import difflib
            print("GOLDEN FASE B ROTO — diferencias:")
            for l in difflib.unified_diff(esperado.splitlines(),
                                          actual.splitlines(), lineterm=""):
                print(l)
            sys.exit(1)
        print("GOLDEN FASE B OK: el ensamblado post-cruce es byte a byte la referencia")
