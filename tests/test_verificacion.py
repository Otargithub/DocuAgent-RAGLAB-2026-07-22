"""
Tests de verificar_preguntas (capas 2/3/4), guardarraíl y examen congelado.
Sin dependencias pesadas:  python3 tests/test_verificacion.py
"""
import sys, types, zlib, tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
pkg = types.ModuleType("rag_lab"); pkg.__path__ = [str(RAIZ / "rag_lab")]
sys.modules["rag_lab"] = pkg

# Stubs mínimos para importar corpus sin langchain
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
sys.modules["rag_lab.perfil"].similitud = lambda *a, **k: 0.0

from rag_lab import corpus as C


class Doc:
    def __init__(self, t): self.page_content = t


class JuezBoW:
    """Embedding bolsa-de-palabras determinista: coseno alto = tokens compartidos."""
    model_name = "juez-bow"
    def embed_documents(self, textos):
        out = []
        for t in textos:
            v = np.zeros(256)
            for tok in set(t.lower().split()):
                rng = np.random.default_rng(zlib.crc32(tok.encode()) & 0xFFFFFFFF)
                v += rng.normal(size=256)
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


CITA_VOLCAN = ("la erupcion del volcan deposito ceniza sobre el valle y "
               "obligo a evacuar a los habitantes de la ladera norte")
PAG_VOLCAN = Doc(CITA_VOLCAN + " durante tres semanas seguidas de actividad")
PAG_GATOS = Doc("los gatos domesticos duermen dieciseis horas y cazan raton al amanecer")
DOCS = [PAG_VOLCAN, PAG_GATOS]


def pregunta(q, cita=CITA_VOLCAN, ans="ceniza sobre el valle"):
    return {"question": q, "expected_answer": ans, "source_context": cita}


def test_capa3_autocontencion():
    assert C._es_autocontenida("¿Qué depositó la erupción sobre el valle?")
    for mala in ("¿Qué dice este documento sobre la ceniza?",
                 "Según el texto, ¿cuántas horas duermen los gatos?",
                 "¿Qué muestra la tabla 3?",
                 "¿Qué afirma el autor del texto?"):
        assert not C._es_autocontenida(mala), mala


def test_capa4_discriminacion():
    juez = JuezBoW()
    buenas = [pregunta("que deposito la erupcion del volcan sobre el valle"),
              pregunta("sobre que zona cayo la ceniza volcanica"),
              pregunta("cual fue el residuo del volcan en los campos del valle")]
    # pregunta sobre GATOS con cita del VOLCÁN: más afín a la página ajena
    mala = pregunta("cuantas horas duermen los gatos domesticos y que cazan")
    res = C.verificar_preguntas([buenas[0], mala, buenas[1], buenas[2]], DOCS,
                                juez=juez, base=0.0, nivel="barata")
    veredictos = [i["veredicto"] for i in res["informe"]]
    assert veredictos == ["ok", "descartada", "ok", "ok"]
    assert "no discrimina" in res["informe"][1]["motivo"]


def test_capa2_roundtrip_inyectado():
    juez = JuezBoW()
    # cuatro preguntas GENUINAMENTE distintas sobre el volcan (el dedupe de la
    # capa 0 elimina reformulaciones; las fixtures ya no pueden ser clones)
    qs = [pregunta("que deposito la erupcion del volcan sobre el valle"),
          pregunta("que dejo caer el volcan tras el estallido"),
          pregunta("sobre que zona cayo la ceniza volcanica"),
          pregunta("cual fue el residuo del volcan en los campos del valle")]
    def verificador(items):
        # el juez falso dice que la pregunta 1 no coincide y la 2 no es respondible
        out = []
        for it in items:
            v = {"indice": it["indice"], "respondible": True, "coincide": True}
            if it["indice"] == 1: v["coincide"] = False
            if it["indice"] == 2: v["respondible"] = False
            out.append(v)
        return out
    res = C.verificar_preguntas(qs, DOCS, juez=juez, base=0.0,
                                nivel="completa", verificador_qa=verificador)
    veredictos = [i["veredicto"] for i in res["informe"]]
    assert veredictos == ["ok", "descartada", "descartada", "ok"]
    assert res["informe"][1]["motivo"] == "la respuesta esperada no se sostiene en la cita"
    assert res["informe"][2]["motivo"] == "la cita no responde a la pregunta"
    assert res["tasa"] == 0.5


def test_guardarrail_minimo():
    juez = JuezBoW()
    malas = [pregunta("cuantas horas duermen los gatos domesticos y que cazan"),
             pregunta("que animal caza ratones al amanecer en la casa"),
             pregunta("cual es la rutina de sueño del felino domestico")]
    res = C.verificar_preguntas(malas, DOCS,
                                juez=juez, base=0.0, nivel="barata")
    assert res.get("guardarrail") and len(res["validas"]) == 3   # conserva y avisa


def test_nivel_citas_no_toca_nada():
    qs = [pregunta("¿Qué dice este documento sobre la ceniza?")]  # deíctica
    res = C.verificar_preguntas(qs, DOCS, nivel="citas")
    assert res["validas"] == qs                                   # capa 1 ya se hizo fuera


def test_capa4_degrada_sin_arbitro():
    class JuezRoto:
        model_name = "roto"
        def embed_documents(self, textos): raise RuntimeError("sin red")
    qs = [pregunta("que deposito la erupcion sobre el valle"),
          pregunta("que dejo caer el volcan tras el estallido"),
          pregunta("cual fue el residuo volcanico en los campos")]
    res = C.verificar_preguntas(qs, DOCS, juez=JuezRoto(), base=0.0, nivel="barata")
    assert len(res["validas"]) == 3                               # no rompe la sesión
    assert res["informe"][0]["capas"]["discrimina"] is None


def test_examen_congelado_memoria():
    from rag_lab.memory import Memoria
    with tempfile.TemporaryDirectory() as d:
        mem = Memoria(fichero=Path(d) / "m.json")
        assert mem.preguntas_congeladas("h1") is None
        ex1 = [pregunta("v1")]
        mem.congelar_preguntas("h1", ex1, info={"tasa": 1.0})
        assert mem.preguntas_congeladas("h1") == ex1
        ex2 = [pregunta("v2a"), pregunta("v2b")]
        mem.congelar_preguntas("h1", ex2)                         # re-congelar
        assert mem.preguntas_congeladas("h1") == ex2              # gana el último
        assert mem.preguntas_congeladas("otro") is None
        mem.congelar_preguntas("", ex1)                           # hash vacío: no-op
        mem.congelar_preguntas("h2", [])                          # vacío: no-op
        assert mem.preguntas_congeladas("h2") is None


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted({k: v for k, v in globals().items()
                              if k.startswith("test_")}.items()):
        try:
            fn(); print(f"  OK   {nombre}")
        except Exception as e:  # noqa: BLE001
            fallos += 1; print(f"  FAIL {nombre}: {e}")
    print("\nTODOS LOS TESTS OK" if not fallos else f"\n{fallos} TESTS FALLARON")
    sys.exit(1 if fallos else 0)


def test_capa0_dedupe_gemelas():
    # el caso REAL de Annie: dos parafrasis de la misma pregunta, misma cita
    cita = "descubro a mi madre atada al sillon. Creia que no ibas a venir"
    gemela_a = {"question": "¿Que recuerda el narrador sobre su madre al ver su apariencia en el sillon?",
                "source_context": cita, "expected_answer": "x"}
    gemela_b = {"question": "¿Que le recuerda al narrador la imagen de su madre en el sillon?",
                "source_context": cita, "expected_answer": "x"}
    # legitima: pregunta DISTINTA sobre la MISMA cita (no debe caer)
    legitima = {"question": "¿Quien ato a la madre y por que segun las enfermeras?",
                "source_context": cita, "expected_answer": "y"}
    unicas, fuera = C._dedupe_preguntas([gemela_a, gemela_b, legitima])
    assert len(unicas) == 2 and len(fuera) == 1
    assert "imagen" in fuera[0]
    assert any("Quien ato" in u["question"] for u in unicas)
