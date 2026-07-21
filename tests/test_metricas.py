"""
Batería de verificación de las métricas de rag_lab.

Se ejecuta SIN dependencias pesadas (solo numpy): registra el paquete sin pasar
por rag_lab/__init__.py (que arrastra LangGraph) y stubbbea langchain/pydantic
solo para el test de validación de citas.

    python3 tests/test_metricas.py
"""
import sys, types, zlib
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
pkg = types.ModuleType("rag_lab"); pkg.__path__ = [str(RAIZ / "rag_lab")]
sys.modules["rag_lab"] = pkg

from rag_lab import metricas as M          # noqa: E402
from rag_lab import evaluacion as E        # noqa: E402


class Doc:
    def __init__(self, t): self.page_content = t


def test_aritmetica_a_mano():
    cita = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10"          # 6 cinco-gramas
    A = Doc("x y w1 w2 w3 w4 w5 w6 w7 z")            # 3/6 grams
    B = Doc("nada que ver con esto en absoluto de verdad")
    r = M.evaluar_recuperacion([B, A], cita, sims=None)
    assert r == {"relevancia": 0.5, "mrr": 0.5, "recuperacion_pasaje": 0.5}
    Aa, Cc = Doc("w1 w2 w3 w4 w5 w6"), Doc("w5 w6 w7 w8 w9 w10")
    r2 = M.evaluar_recuperacion([Aa, Cc], cita, sims=None)
    assert r2["recuperacion_pasaje"] == round(4 / 6, 4)            # unión reensambla
    assert r2["relevancia"] == round(2 / 6, 4) and r2["mrr"] == 0.0


def test_cita_corta_ngrama_efectivo():
    r = M.evaluar_recuperacion([Doc("hoy el gato come pescado fresco")],
                               "gato come pescado", sims=None)
    assert r["relevancia"] == 1.0


def test_guarda_sims_paralelas():
    try:
        M.evaluar_recuperacion([Doc("a"), Doc("b")], "gato come pescado", sims=[0.9])
        raise AssertionError("no saltó la guarda de longitud")
    except ValueError:
        pass


def test_alpha_es_perilla():
    cita = "el sistema de recuperacion aumentada mejora la precision de las respuestas"
    paraf = Doc("aumentar la generacion con contexto recuperado eleva la exactitud")
    solo_lex = M.evaluar_recuperacion([paraf], cita, sims=None)["relevancia"]
    a05 = M.evaluar_recuperacion([paraf], cita, sims=[0.95], peso_exactitud=0.5)["relevancia"]
    a02 = M.evaluar_recuperacion([paraf], cita, sims=[0.95], peso_exactitud=0.2)["relevancia"]
    assert solo_lex == 0.0 and a02 > a05 > 0.0


def test_recalibracion():
    assert M.recalibrar_similitud(0.30, 0.3) == 0.0
    assert abs(M.recalibrar_similitud(1.0, 0.3) - 1.0) < 1e-9
    assert abs(M.recalibrar_similitud(0.75, 0.0) - 0.75) < 1e-9   # base 0 = neutro


def test_normalizar_degenerado_neutro():
    assert M.normalizar(7.0, 7.0, 7.0) == 0.5


class JuezSesgado:
    """v = norm(a*u + b*h(texto)): pares no relacionados tienen coseno ~ a^2."""
    model_name = "juez-sintetico"
    def __init__(self, a=0.8, dim=64):
        self.a, self.b, self.dim = a, (1 - a ** 2) ** 0.5, dim
        rng = np.random.default_rng(7)
        self.u = rng.normal(size=dim); self.u /= np.linalg.norm(self.u)
    def _h(self, texto):
        # crc32, NO hash(): el hash de str está aleatorizado por proceso
        # (PYTHONHASHSEED) y hacía el test flaky entre ejecuciones.
        rng = np.random.default_rng(zlib.crc32(texto.encode()) & 0xFFFFFFFF)
        v = rng.normal(size=self.dim)
        v -= (v @ self.u) * self.u
        return v / np.linalg.norm(v)
    def embed_documents(self, textos):
        return [(self.a * self.u + self.b * self._h(t)).tolist() for t in textos]


def test_calibracion_empirica():
    juez = JuezSesgado(a=0.8)                    # suelo teórico 0.64
    E.CONFIG.base_semantica = None
    E.fijar_embedding_juez(juez)
    base = E.obtener_base_semantica(juez)
    assert abs(base - 0.64) < 0.05
    assert E.obtener_base_semantica(juez) == base          # caché
    E.CONFIG.base_semantica = 0.2
    assert E.obtener_base_semantica(juez) == 0.2           # override manual manda
    E.CONFIG.base_semantica = None
    va = M.vector_normalizado("la bolsa subió hoy", juez)
    vb = M.vector_normalizado("el delfín nada en el océano", juez)
    assert M.recalibrar_similitud(float(va @ vb), base) < 0.15


def test_guardarrail_arbitro():
    E.EMBEDDING_JUEZ = {"model": "intfloat/multilingual-e5-small",
                        "provider": "huggingface"}
    assert E.relacion_con_arbitro({"model": "intfloat/multilingual-e5-small"}) == "identico"
    assert E.relacion_con_arbitro({"model": "intfloat/multilingual-e5-large"}) == "familia"
    assert E.relacion_con_arbitro({"model": "BAAI/bge-m3"}) == "ok"


def test_estadistica():
    from rag_lab.estadistica import bootstrap_ic, comparacion_pareada
    v = [0.5, 0.7, 0.6, 0.8, 0.4]
    assert bootstrap_ic(v, semilla="x") == bootstrap_ic(v, semilla="x")
    a = [0.9, 0.8, 0.85, 0.95, 0.9, 0.88]
    b = [0.5, 0.4, 0.45, 0.55, 0.5, 0.48]
    c = comparacion_pareada(a, b, semilla="y")
    assert c["concluyente"] and c["dif_media"] > 0
    assert not comparacion_pareada(a, a[:3], semilla="w")["concluyente"]


def test_validar_citas():
    for mod in ["langchain_community", "langchain_community.document_loaders",
                "langchain_core", "langchain_core.prompts", "pydantic"]:
        sys.modules.setdefault(mod, types.ModuleType(mod))
    sys.modules["langchain_community.document_loaders"].PyPDFLoader = object
    sys.modules["langchain_core.prompts"].ChatPromptTemplate = object
    class _BM: pass
    sys.modules["pydantic"].BaseModel = _BM
    sys.modules["pydantic"].ConfigDict = dict
    for mod in ["rag_lab.fabricas", "rag_lab.perfil"]:
        m = types.ModuleType(mod); sys.modules[mod] = m
    sys.modules["rag_lab.fabricas"].construir_llm_clasificador = lambda *a, **k: None
    sys.modules["rag_lab.perfil"].perfil_de_documento = lambda *a, **k: {}
    sys.modules["rag_lab.perfil"].descriptor = lambda *a, **k: "?"
    from rag_lab import corpus as C
    doc = [Doc("La arquitectura RAG combina un recuperador denso con un generador. "
               "El recuperador selecciona pasajes relevantes y el generador los usa "
               "como contexto para responder con mayor fidelidad factual.")]
    reales = [{"question": f"real{i}", "source_context":
        "El recuperador selecciona pasajes relevantes y el generador los usa como contexto"}
        for i in range(4)]
    inventada = {"question": "INVENTADA", "source_context":
        "El modelo fue entrenado con refuerzo sobre preferencias humanas"}
    ruido = {"question": "RUIDO_PDF", "source_context":
        "La arquitectura RAG combina un recupe-\nrador denso con un gene-\nrador"}
    etqs = {q["question"] for q in C.validar_citas(reales + [inventada, ruido], doc)}
    assert "INVENTADA" not in etqs and "RUIDO_PDF" in etqs


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted({k: v for k, v in globals().items()
                              if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"  OK   {nombre}")
        except Exception as e:  # noqa: BLE001
            fallos += 1
            print(f"  FAIL {nombre}: {e}")
    print("\nTODOS LOS TESTS OK" if not fallos else f"\n{fallos} TESTS FALLARON")
    sys.exit(1 if fallos else 0)
