"""Tests del observatorio con memoria sintética.  python3 tests/test_observatorio.py"""
import sys, types, tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
pkg = types.ModuleType("rag_lab"); pkg.__path__ = [str(RAIZ / "rag_lab")]
sys.modules["rag_lab"] = pkg
m = types.ModuleType("rag_lab.perfil"); sys.modules["rag_lab.perfil"] = m
m.similitud = lambda *a, **k: 0.0
m.perfil_de_documento = lambda *a, **k: {}
m.descriptor = lambda *a, **k: "?"

from rag_lab.memory import Memoria
from rag_lab.analisis import observatorio as obs


def _memoria_sintetica(d):
    mem = Memoria(fichero=Path(d) / "m.json")
    # dos documentos con perfiles distintos; A gana en el denso, B en el léxico
    docs = [("den", "denso.pdf", {"descriptor": "denso", "densidad": 0.8, "lex": 0.1}),
            ("lex", "lexico.pdf", {"descriptor": "lexico", "densidad": 0.3, "lex": 0.6})]
    for rid, nombre, perfil in docs:
        mem.registrar_sesion(rid, nombre, f"hash-{rid}", perfil, 40)
        for run in (1, 2, 3):
            if rid == "den":
                ranking = {"A": 0.80 + 0.01 * run, "B": 0.60 + 0.01 * run, "C": 0.40}
            else:
                ranking = {"A": 0.55, "B": 0.75 + 0.01 * run, "C": 0.45}
            ganador = max(ranking, key=ranking.get)
            mem.registrar_experimento(rid, "embedding", run=run, veredicto={
                "embedding_recomendado": ganador,
                "_primaria": {"nombre": "mrr", "valor": ranking[ganador],
                              "ic_bajo": 0.5, "ic_alto": 0.9, "n": 40},
                "_ranking": ranking})
    return mem


def test_panorama_y_rendimiento():
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        pan = obs.panorama(mem)
        assert pan["documentos"] == 2 and pan["experimentos_total"] == 6
        assert pan["con_ranking_completo"] == 6 and pan["aviso"] is None
        r = obs.rendimiento(mem, "embedding", "A")
        assert r["n"] == 6 and 0.6 < r["media"] < 0.75
        assert obs.rendimiento(mem, "embedding", "Z")["disponible"] is False


def test_duelo_pareado_y_desglose():
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        duo = obs.duelo(mem, "embedding", "A", "B")
        assert duo["disponible"] and duo["n"] == 6
        # global: A gana en denso (+0.20) y pierde en léxico (-0.20) => empate ~0
        assert abs(duo["global"]["dif_media"]) < 0.05
        assert duo["por_documento"]["denso.pdf"]["dif_media"] > 0.15
        assert duo["por_documento"]["lexico.pdf"]["dif_media"] < -0.15
        # duelo sin datos suficientes
        assert obs.duelo(mem, "embedding", "A", "Z")["disponible"] is False


def test_victorias_por_perfil():
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        tabla = obs.victorias_por_perfil(mem, "embedding")
        assert tabla == {"denso": {"A": 1}, "lexico": {"B": 1}}


def test_correlacion_perfil():
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        c = obs.correlacion_perfil(mem, "embedding", "A", "densidad")
        # solo 2 documentos: correlación no calculable de forma fiable
        assert c["n_documentos"] == 2 and c["spearman"] is None


def test_filtro_por_era():
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        assert obs.panorama(mem, desde="2099-01-01")["experimentos_total"] == 0
        assert obs.panorama(mem, desde="2000-01-01")["experimentos_total"] == 6


def test_degradacion_sin_ranking():
    with tempfile.TemporaryDirectory() as d:
        mem = Memoria(fichero=Path(d) / "m.json")
        mem.registrar_sesion("r1", "x.pdf", "h1", {"descriptor": "x"}, 10)
        mem.registrar_experimento("r1", "embedding", run=1, veredicto={
            "embedding_recomendado": "A",
            "_finalistas": [{"etiqueta": "A", "es_lider": True, "valor": 0.7},
                            {"etiqueta": "B", "valor": 0.65}]})
        pan = obs.panorama(mem)
        assert pan["con_ranking_completo"] == 0 and pan["aviso"]
        r = obs.rendimiento(mem, "embedding", "B")
        assert r["disponible"] and r["n"] == 1     # cae a finalistas


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


def test_plan_dirigido_topes():
    from rag_lab.experimento_dirigido import plan_dirigido, LIMITES
    p = plan_dirigido("embedding", ["bge-m3", "granite-311m-r2"], n_runs=1)
    assert p["bancos"] == ["embedding"] and p["pausas"] == []
    assert p["restricciones"] == {"embedding": {"etiquetas": ["bge-m3", "granite-311m-r2"]}}
    assert plan_dirigido("splitter", ["x"], n_runs=99)["n_runs"] == LIMITES["max_runs"]
    for mala in (lambda: plan_dirigido("metadata_x", ["a"]),
                 lambda: plan_dirigido("embedding", []),
                 lambda: plan_dirigido("embedding", ["a", "b", "c", "d", "e"])):
        try:
            mala(); raise AssertionError("no validó")
        except ValueError:
            pass


def test_etiquetas_con_evidencia():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mem = _memoria_sintetica(d)
        n = obs.etiquetas_con_evidencia(mem, "embedding")
        assert n == {"A": 6, "B": 6, "C": 6}
        assert obs.etiquetas_con_evidencia(mem, "reranker") == {}
        assert obs.etiquetas_con_evidencia(mem, "embedding", desde="2099-01-01") == {}
