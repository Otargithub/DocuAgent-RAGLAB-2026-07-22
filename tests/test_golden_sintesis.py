# --------------------------------------------------------------------------- #
# GOLDEN TEST de la capa de sintesis (fase 0 del refactor de meta.py).
#
# Congela el comportamiento COMPLETO de _nodo_sintetizar sobre entradas
# sinteticas deterministas: stack recomendado, alternativa economica, bancos
# fijados por el usuario, registro de la decision en memoria, resolucion de
# predicciones (con el DIFERIDO del splitter si hay Fase B pendiente) y el
# payload devuelto. Si un refactor cambia UN byte de ese comportamiento, este
# test lo delata.
#
# Costuras (deliberadas, documentadas):
#   - _certezas_por_banco se sustituye por certezas CONGELADAS (formato real).
#   - la narracion del LLM se pre-siembra en el memo -> el juez NUNCA se llama.
#   - la memoria vive en un directorio temporal.
# Regenerar el golden (solo cuando un cambio de comportamiento sea DESEADO):
#   UPDATE_GOLDEN=1 python3 tests/test_golden_sintesis.py
# --------------------------------------------------------------------------- #
import json
import os
import sys
import tempfile
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
GOLDEN = Path(__file__).parent / "golden" / "sintesis_golden.json"

# ---- arnes de imports: usa el entorno real si existe; si no, stubs --------- #
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
            # permisivo: instanciable con cualquier argumento (Field(...),
            # BaseModel subclaseable, decoradores, etc.)
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

os.environ["RAG_MEMORY_DIR"] = tempfile.mkdtemp(prefix="golden_sintesis_")

from rag_lab.supervisor.meta import MetaSupervisor  # noqa: E402
from rag_lab.memory import Memoria  # noqa: E402


# ---- entradas congeladas (formato de las certezas reales de los payloads) -- #
def certezas_congeladas() -> dict:
    return {
        "splitter": {
            "disponible": True, "resuelto": False, "es_empate": True,
            "ganador": "token_250", "acuerdo": 0.5, "concluyente": True,
            "metrica_primaria": "recuperacion_pasaje",
            "ic_primaria": {"ic_bajo": 0.581, "ic_alto": 0.652, "amplitud": 0.071},
            "motivo": "desacuerdo_entre_runs",
            "finalistas": [
                {"etiqueta": "token_250", "es_lider": True, "valor": 0.622,
                 "config": {"splitter": "token", "chunk_size": 250,
                            "chunk_overlap": 25}},
                {"etiqueta": "token_500", "es_lider": False, "valor": 0.615,
                 "config": {"splitter": "token", "chunk_size": 500,
                            "chunk_overlap": 50}},
            ],
        },
        "embedding": {
            "disponible": True, "resuelto": True, "es_empate": False,
            "ganador": "granite-311m-r2", "acuerdo": 1.0, "concluyente": True,
            "metrica_primaria": "mrr",
            "ic_primaria": {"ic_bajo": 0.171, "ic_alto": 0.242, "amplitud": 0.071},
            "motivo": None,
            "finalistas": [
                {"etiqueta": "granite-311m-r2", "es_lider": True, "valor": 0.209,
                 "config": {"model": "ibm-granite/granite-embedding-311m-multilingual-r2",
                            "provider": "huggingface"}},
            ],
        },
        "retriever": {
            "disponible": True, "resuelto": False, "es_empate": True,
            "ganador": "ensemble_k10_w0.7", "acuerdo": 0.5, "concluyente": False,
            "metrica_primaria": "mrr",
            "ic_primaria": {"ic_bajo": 0.402, "ic_alto": 0.512, "amplitud": 0.110},
            "motivo": "margen_no_concluyente",
            "finalistas": [
                {"etiqueta": "ensemble_k10_w0.7", "es_lider": True, "valor": 0.452,
                 "config": {"nombre": "ensemble", "k": 10,
                            "params": {"w_bm25": 0.7}}},
                {"etiqueta": "bm25_k10", "es_lider": False, "valor": 0.447,
                 "config": {"nombre": "bm25", "k": 10, "params": {}}},
            ],
        },
    }


def correr_nodo() -> dict:
    meta = MetaSupervisor(memoria=Memoria())
    run_id = "golden_run_1"

    # sesion sintetica minima (solo los atributos que el nodo consume)
    sesion = types.SimpleNamespace(
        run_id=run_id, doc_hash="golden_doc", pdf_path="/tmp/golden.pdf",
        doc_type="golden", documentos=[], preguntas_test=[{"q": 1}] * 12,
        restricciones={}, verificacion=None, config_splitter={},
        config_embedding=None, config_retriever=None,
        perfil={"descriptor": "es|medio|lex-baja", "densidad": 0.43,
                "lex": 0.15})
    meta._sesiones[run_id] = sesion

    # memoria: la sesion y dos PREDICCIONES (el diferido del splitter es parte
    # del comportamiento congelado: hay fase B pendiente -> splitter NO se
    # resuelve aqui; embedding SI, y ACIERTA)
    meta.memoria.registrar_sesion = getattr(meta.memoria, "registrar_sesion", None)
    d = meta.memoria._leer()
    d.setdefault("sesiones", []).append({"run_id": run_id, "doc_hash": "golden_doc",
                                         "pdf_nombre": "golden.pdf"})
    meta.memoria._escribir(d) if hasattr(meta.memoria, "_escribir") else None
    meta.memoria.registrar_prediccion(run_id, "splitter", "token_500")
    meta.memoria.registrar_prediccion(run_id, "embedding", "granite-311m-r2")

    # costuras: certezas congeladas + narracion pre-sembrada (el juez no se llama)
    meta._certezas_por_banco = lambda run, bancos: certezas_congeladas()
    meta._memo()[("narracion", run_id)] = {
        "confianza_global": "media",
        "justificacion": "NARRACION-GOLDEN (fija, sin LLM).",
        "alertas": ["ALERTA-GOLDEN"]}

    estado = {
        "run_id": run_id,
        "plan": {"objetivo": "maxima calidad con coste razonable",
                 "bancos": ["splitter", "embedding", "retriever"],
                 "n_runs": 2, "max_preguntas": 40, "presupuesto": {"max_runs": 2, "max_preguntas": 40},
                 "restricciones": {"reranker": {"fijar": {
                     "etiqueta": "ce-minilm-l6", "modelo": "ms-marco-MiniLM-L-6-v2"}}}},
        "veredictos": {},
        "runs_hechos": 2,
    }
    salida = meta._nodo_sintetizar(estado)

    # foto completa y canonica del comportamiento observable
    mem = meta.memoria._leer()
    predicciones = [{k: p.get(k) for k in ("banco", "esperado", "real", "acierto")}
                    for p in mem.get("predicciones", [])]
    decisiones = mem.get("decisiones", [])
    stack = (salida or {}).get("stack_final") or {}
    return {
        "stack_recomendado": stack.get("stack_recomendado"),
        "stack_economico": stack.get("stack_alternativo_economico"),
        "certezas": stack.get("certezas"),
        "pendiente_fase_b": stack.get("pendiente_fase_b"),
        "confianza_global": stack.get("confianza_global"),
        "justificacion": stack.get("justificacion"),
        "alertas": stack.get("alertas"),
        "n_decisiones_registradas": len(decisiones),
        "decision_tiene_stack": bool((decisiones or [{}])[0].get("stack")),
        "predicciones": sorted(predicciones, key=lambda p: p["banco"] or ""),
    }


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


if __name__ == "__main__":
    foto = correr_nodo()
    if os.environ.get("UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(canon(foto))
        print(f"GOLDEN {'re' if GOLDEN.exists() else ''}generado: {GOLDEN}")
        print(canon(foto))
    else:
        esperado = GOLDEN.read_text()
        actual = canon(foto)
        if actual != esperado:
            import difflib
            print("GOLDEN ROTO — diferencias:")
            for l in difflib.unified_diff(esperado.splitlines(),
                                          actual.splitlines(), lineterm=""):
                print(l)
            sys.exit(1)
        print("GOLDEN OK: la sintesis se comporta byte a byte como la referencia")
