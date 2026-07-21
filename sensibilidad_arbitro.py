"""
Sensibilidad al árbitro: ¿el veredicto del banco de embeddings depende de quién
puntúa el parecido semántico?

La pregunta correcta
--------------------
Cambiar de árbitro "a ver si deja de ganar X" es amañar al árbitro. La pregunta
legítima es: ¿e5-small (u otro) favorece sistemáticamente a los modelos cuyo
espacio se parece al suyo? Este script la responde con datos: los MISMOS
pipelines, el MISMO examen congelado, puntuados bajo VARIOS árbitros de linajes
distintos.

Lectura del resultado
---------------------
- Ranking IDÉNTICO bajo todos los árbitros -> los veredictos son robustos: el
  ganador gana por recuperar mejor, no por caerle bien al juez. No cambies nada.
- Ranking que BAILA según el árbitro -> hallazgo metodológico: la nota es
  sensible al árbitro y toca decidir uno con criterio (linaje ajeno a TODOS los
  concursantes) y reanclar con el juicio humano (anclaje_humano.py), que es el
  único árbitro que no se puede amañar.

No escribe NADA en memoria: es un análisis, no un experimento del laboratorio.

Uso:
    python sensibilidad_arbitro.py pdfs/test_13.pdf
    python sensibilidad_arbitro.py pdfs/test_13.pdf \\
        --etiquetas "bge-m3,granite-311m-r2,granite-97m-r2" \\
        --arbitros "intfloat/multilingual-e5-small,sentence-transformers/LaBSE"
"""
from __future__ import annotations

import argparse
from collections import defaultdict


# Árbitros por defecto: tres LINAJES distintos, todos multilingües y cargables
# con sentence-transformers estándar (sin trust_remote_code):
#   - e5-small  (Microsoft/intfloat)  -> el titular actual
#   - mpnet multilingual (SBERT/UKP)  -> linaje paraphrase, anterior y ajeno
#   - LaBSE (Google)                  -> linaje BERT bilingüe alineado
ARBITROS_DEFAULT = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
]


def ranking_de(mrrs: dict[str, float]) -> list[str]:
    """Etiquetas ordenadas de mejor a peor (empates: orden alfabético estable)."""
    return [e for e, _ in sorted(mrrs.items(), key=lambda kv: (-kv[1], kv[0]))]


def veredicto_estabilidad(por_arbitro: dict[str, dict[str, float]]) -> dict:
    """
    Función pura: dado {arbitro: {etiqueta: mrr}}, dictamina si el ranking es
    estable entre árbitros y cuánto se mueve el líder.
    """
    rankings = {a: ranking_de(m) for a, m in por_arbitro.items()}
    lista = list(rankings.values())
    estable = all(r == lista[0] for r in lista)
    lideres = {a: r[0] for a, r in rankings.items()}
    return {"estable": estable, "rankings": rankings, "lideres": lideres,
            "lider_unanime": len(set(lideres.values())) == 1}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sensibilidad al árbitro semántico")
    ap.add_argument("pdf", help="PDF con examen congelado")
    ap.add_argument("--etiquetas", default="bge-m3,granite-311m-r2,granite-97m-r2",
                    help="concursantes del registro de embeddings (coma-separados)")
    ap.add_argument("--arbitros", default=",".join(ARBITROS_DEFAULT),
                    help="modelos HF a usar como árbitro (coma-separados)")
    ap.add_argument("--splitter", default="recursive_1000",
                    help="splitter fijo del pipeline (mismo para todos)")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    from langchain_community.vectorstores import FAISS

    from rag_lab import evaluacion as ev
    from rag_lab.bancos.embeddings_banco import EMBEDDING_REGISTRY
    from rag_lab.bancos.splitters import _construir_splitter_banco
    from rag_lab.corpus import hash_pdf, preparar_sesion
    from rag_lab.evaluacion import evaluar_sobre_preguntas, fijar_embedding_juez
    from rag_lab.fabricas import construir_embedding
    from rag_lab.memory import Memoria

    etiquetas = [e.strip() for e in args.etiquetas.split(",") if e.strip()]
    arbitros = [a.strip() for a in args.arbitros.split(",") if a.strip()]

    mem = Memoria()
    congeladas = mem.preguntas_congeladas(hash_pdf(args.pdf) or "")
    assert congeladas, "El documento no tiene examen congelado: pasa antes por el gimnasio."
    sesion = preparar_sesion(args.pdf, preguntas_predefinidas=congeladas)
    print(f"[sensibilidad] {len(sesion.preguntas_test)} preguntas congeladas | "
          f"concursantes: {etiquetas} | árbitros: {len(arbitros)}")

    # Pipelines de los concursantes: se construyen UNA vez. La recuperación no
    # depende del árbitro (solo la puntuación), así que los documentos
    # recuperados son idénticos bajo todos los árbitros: comparación limpia.
    import re
    m = re.fullmatch(r"(recursive|token|character)_(\d+)", args.splitter)
    assert m, f"splitter no reconocido: {args.splitter}"
    splitter = _construir_splitter_banco(m.group(1), int(m.group(2)),
                                         int(int(m.group(2)) * 0.1))
    chunks = splitter.split_documents(sesion.documentos)
    retrievers = {}
    for et in etiquetas:
        modelo = construir_embedding(EMBEDDING_REGISTRY[et])
        vs = FAISS.from_documents(chunks, modelo)
        retrievers[et] = (vs.as_retriever(search_kwargs={"k": args.k}), modelo)
    print(f"[sensibilidad] {len(chunks)} chunks | {len(retrievers)} índices construidos.")

    por_arbitro: dict[str, dict[str, float]] = defaultdict(dict)
    for arb in arbitros:
        cfg_arb = {"model": arb, "provider": "huggingface"}
        # inyectar árbitro (limpia cachés y recalibra su suelo semántico) y
        # actualizar el guardarraíl para clasificar a los concursantes
        fijar_embedding_juez(construir_embedding(cfg_arb))
        ev.EMBEDDING_JUEZ = dict(cfg_arb)
        print(f"\n=== ÁRBITRO: {arb} ===")
        for et in etiquetas:
            rel = ev.relacion_con_arbitro(EMBEDDING_REGISTRY[et])
            if rel != "ok":
                print(f"  ⚠ {et}: relación con el árbitro = {rel} "
                      f"(sesgo de afinidad posible; interpreta con cautela)")
            retriever, modelo = retrievers[et]
            met = evaluar_sobre_preguntas(
                lambda consulta, _r=retriever: _r.invoke(consulta),
                sesion.preguntas_test, modelo)
            por_arbitro[arb][et] = float(met.get("mrr") or 0.0)
            print(f"  {et:22} mrr={por_arbitro[arb][et]:.4f}")

    v = veredicto_estabilidad(dict(por_arbitro))
    print("\n" + "=" * 66)
    print("  VEREDICTO DE SENSIBILIDAD AL ÁRBITRO")
    print("=" * 66)
    for a, r in v["rankings"].items():
        print(f"  {a.split('/')[-1]:44} -> {' > '.join(r)}")
    if v["estable"]:
        print("\n✓ RANKING ESTABLE bajo todos los árbitros: los veredictos del "
              "banco son robustos.\n  El ganador gana por recuperar mejor, no "
              "por afinidad con el juez. No cambies el árbitro.")
    elif v["lider_unanime"]:
        print("\n~ El LÍDER es unánime pero el resto del ranking baila: veredicto "
              "principal robusto,\n  posiciones secundarias sensibles al árbitro.")
    else:
        print("\n✗ EL LÍDER DEPENDE DEL ÁRBITRO: hallazgo metodológico serio. "
              "Siguiente paso:\n  anclaje_humano.py bajo cada árbitro candidato y "
              "quédate con el que mejor\n  correlacione con TU juicio. Y recuerda: "
              "cambiar de árbitro = nueva era de métricas.")


if __name__ == "__main__":
    main()
