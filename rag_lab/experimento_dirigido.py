"""
Experimento dirigido: la puerta ACOTADA por la que un agente (o un humano con
prisa) puede lanzar experimentos del laboratorio.

Por qué existe
--------------
Darle a DocuAgent `meta.ejecutar` a pelo seria darle un boton de gastar sin
tope (tandas de 45 minutos de GPU/API) y con HITL anidado (las pausas del meta
son interrupts DENTRO de una tool dentro del grafo del agente). Esta funcion es
la alternativa segura: un solo banco, parrilla acotada por etiquetas (la
maquinaria de restricciones con fallo rapido valida que existan), runs
limitados, sin pausas internas, y con el examen congelado del documento para
que el resultado sea directamente comparable con la evidencia previa.

El resultado se persiste en memoria como cualquier corrida => el observatorio
(y por tanto DocuAgent) lo ve en su siguiente analisis. Bucle cerrado:
analizar -> proponer -> (peaje humano) -> experimentar -> analizar.
"""

from __future__ import annotations

from pathlib import Path

# Topes duros del experimento dirigido. Son deliberadamente TACAÑOS: si algo
# necesita mas, es una tanda del gimnasio decidida por un humano, no una tool.
# max_runs 2->5 (2026-07-17, "potencia configurable" de la hoja de ruta): con
# n=2 el eje de acuerdo entre runs es una moneda al aire — el dirigido corre
# EXACTAMENTE los runs pedidos (minimo=maximo=n_runs: la parada reactiva solo
# actua despues del minimo), asi que quien pida 5 paga 5 a sabiendas. El tope
# en 5 y no mas: por encima, es una tanda del gimnasio, no un dirigido.
LIMITES = {
    "bancos": ("splitter", "embedding", "retriever", "reranker", "metadata"),
    "max_etiquetas": 4,
    "max_runs": 5,
    "max_preguntas": 40,
}


def _resolver_ruta_pdf(pdf: str) -> str:
    """Acepta una RUTA o solo un NOMBRE ('Annie.pdf') y devuelve una ruta que
    existe. El agente solo conoce nombres (listar_documentos no da rutas), asi
    que sin esto no podia acertar nunca el argumento pdf. Cascada: (1) tal cual
    si existe; (2) buscar el nombre en el directorio de PDFs configurado
    (PDF_PATH); (3) devolver tal cual (el assert dara el error claro)."""
    from pathlib import Path
    import os
    p = Path(pdf)
    if p.exists():
        return str(p)
    nombre = p.name
    base = os.getenv("PDF_PATH", "")
    candidatos = []
    if base:
        b = Path(base)
        candidatos += [b / nombre, b.parent / nombre] if b.is_file() else [b / nombre]
    # ubicaciones habituales en Colab, por si PDF_PATH apunta a un fichero suelto
    candidatos += [Path("/content/drive/MyDrive/rag_lab/pdfs") / nombre]
    for c in candidatos:
        if c.exists():
            return str(c)
    return pdf


def catalogo_de_banco(banco: str) -> list[str]:
    """Etiquetas que la PARRILLA de un banco puede generar — independientes de
    la memoria (para que el agente elija nombres validos aunque no haya
    historia). Es la fuente de verdad de 'que puedo probar en este banco'."""
    if banco == "splitter":
        from .bancos.splitters import TAMANOS_FRAGMENTO
        fams = ["recursive", "token", "character"]
        ets = [f"{f}_{t}" for f in fams for t in TAMANOS_FRAGMENTO] + ["semantic"]
        return ets
    if banco == "embedding":
        from .bancos.embeddings_banco import EMBEDDING_REGISTRY
        return list(EMBEDDING_REGISTRY)
    if banco == "retriever":
        from .bancos.retrievers import EXPERIMENTOS_RETRIEVER
        ets = []
        for e in EXPERIMENTOS_RETRIEVER:
            nombre, k, params = e["nombre"], e["k"], e.get("params", {})
            if nombre == "mmr":
                ets.append(f"mmr_k{k}_l{params.get('lambda_mult')}")
            elif nombre == "ensemble":
                ets.append(f"ensemble_k{k}_w{params.get('w_bm25')}")
            else:
                ets.append(f"{nombre}_k{k}")
        return ets
    if banco == "reranker":
        from .bancos.rerankers import RERANKER_REGISTRY
        return list(RERANKER_REGISTRY)
    if banco == "metadata":
        from .bancos.metadata import ALPHAS, CONFIGS_METADATOS
        ets = []
        for a in ALPHAS:
            for cfg in CONFIGS_METADATOS:
                if a >= 1.0 and cfg["name"] != "baseline":
                    continue
                ets.append(f"a{a:.1f}_{cfg['name']}")
        return ets
    return []


def plan_dirigido(banco: str, etiquetas: list[str], n_runs: int = 1) -> dict:
    """
    Construye el plan MINIMO y validado de un experimento dirigido. Funcion
    pura (testeable sin entorno). Lanza ValueError con mensaje accionable si la
    peticion viola los topes — el agente recibe ese mensaje como observacion.
    """
    if banco not in LIMITES["bancos"]:
        raise ValueError(f"banco {banco!r} no permitido; validos: {LIMITES['bancos']}")
    etiquetas = [str(e).strip() for e in (etiquetas or []) if str(e).strip()]
    if not etiquetas:
        raise ValueError("hay que indicar al menos una etiqueta a probar")
    if len(etiquetas) > LIMITES["max_etiquetas"]:
        raise ValueError(f"{len(etiquetas)} etiquetas > tope "
                         f"{LIMITES['max_etiquetas']}: acota la hipotesis")
    n_runs = max(1, min(int(n_runs), LIMITES["max_runs"]))
    return {
        "bancos": [banco],
        "n_runs": n_runs,
        "presupuesto": {"max_runs": n_runs,
                        "max_preguntas": LIMITES["max_preguntas"]},
        "restricciones": {banco: {"etiquetas": etiquetas}},
        "pausas": [],                    # NUNCA pausas internas: el peaje va fuera
    }


def _stack_ganador(mem, doc_hash: str) -> dict | None:
    """El stack CORONADO del documento en su ultima decision persistida, en los
    formatos que el meta espera para fijar: {"splitter": {...}, "embedding":
    {model, provider}, "retriever": {retriever_name, params}}. Solo lee
    coronaciones COMPLETAS (splitter+embedding+retriever); ignora decisiones
    parciales de banco unico. None si no hay coronacion."""
    d = mem._leer()
    runs_doc = {s.get("run_id") for s in d.get("sesiones", [])
                if s.get("doc_hash") == doc_hash}
    for dec in reversed(d.get("decisiones", [])):
        if dec.get("run_id") not in runs_doc:
            continue
        rec = (dec.get("stack") or {}).get("stack_recomendado", {})
        if not ({"splitter", "embedding", "retriever"} <= set(rec)):
            continue
        spl = (rec.get("splitter") or {}).get("config") or {}
        emb = (rec.get("embedding") or {}).get("config") or {}
        ret = (rec.get("retriever") or {}).get("config") or {}
        if not (emb.get("model") or emb.get("clave")):
            continue
        # embedding: resolver clave del registro a {model, provider}
        from .bancos.embeddings_banco import EMBEDDING_REGISTRY
        entrada = EMBEDDING_REGISTRY.get(emb.get("clave") or emb.get("etiqueta"))
        if entrada:
            emb_fijo = {"model": entrada["model"], "provider": entrada["provider"]}
        else:
            emb_fijo = {"model": emb.get("model"),
                        "provider": emb.get("provider", "openai")}
        stack = {"embedding": emb_fijo}
        if spl.get("splitter") or spl.get("name"):
            stack["splitter"] = {"name": spl.get("splitter") or spl.get("name"),
                                 "chunk_size": spl.get("chunk_size", 1000),
                                 "chunk_overlap": spl.get("chunk_overlap", 100)}
        if ret.get("nombre") or ret.get("retriever_name"):
            stack["retriever"] = {"retriever_name": ret.get("nombre")
                                  or ret.get("retriever_name"),
                                  "params": ret.get("params", {}),
                                  "k": ret.get("k")}
        return stack
    return None


def _embedding_ganador(mem, doc_hash: str) -> dict | None:
    """Compatibilidad: solo la pieza de embedding del stack coronado."""
    stack = _stack_ganador(mem, doc_hash)
    return (stack or {}).get("embedding")


def experimento_dirigido(pdf: str, banco: str, etiquetas: list[str],
                         n_runs: int = 1) -> dict:
    """
    Lanza el experimento acotado y devuelve un resumen compacto. Requiere el
    entorno completo (GPU/APIs). El documento DEBE tener examen congelado: sin
    el, los resultados no serian comparables con la evidencia previa y el
    analisis posterior mezclaria examenes.
    """
    from .memory import Memoria
    from .corpus import hash_pdf
    from .bancos import CLAVE_DECISION
    from .supervisor import MetaSupervisor

    pdf = _resolver_ruta_pdf(pdf)
    assert Path(pdf).exists(), f"no existe {pdf}"
    plan = plan_dirigido(banco, etiquetas, n_runs)

    mem = Memoria()
    doc_hash = hash_pdf(pdf) or ""
    if not mem.preguntas_congeladas(doc_hash):
        raise ValueError(
            f"{Path(pdf).name} no tiene examen congelado: pasa antes por el "
            f"gimnasio. Sin examen comun, este experimento no seria comparable "
            f"con la evidencia previa.")

    # Bancos que se enriquecen/indexan SOBRE un embedding: medir sobre el
    # embedding GANADOR del documento, no sobre el default exploratorio (obs. de
    # Xavier: el metadata sobre text-embedding-3-small daba MRR de sotano en
    # Annie porque la base estaba desalineada, no porque el metadata fuera malo).
    if banco in ("metadata", "retriever", "reranker"):
        stack_g = _stack_ganador(mem, doc_hash)
        if stack_g:
            # embedding coronado: base para los tres bancos dependientes
            plan["restricciones"].setdefault("embedding", {})["fijar"] = \
                stack_g["embedding"]
            fijadas = [f"embedding={stack_g['embedding'].get('model')}"]
            # el RERANKER se mide sobre el stack coronado ENTERO (splitter y
            # retriever incluidos): su rerank_gain solo significa "¿vale la
            # pena reordenar?" si se calcula sobre el pipeline que despliegas.
            if banco == "reranker":
                if stack_g.get("splitter"):
                    plan["restricciones"].setdefault("splitter", {})["fijar"] = \
                        stack_g["splitter"]
                    fijadas.append(f"splitter={stack_g['splitter'].get('name')}"
                                   f"_{stack_g['splitter'].get('chunk_size')}")
                if stack_g.get("retriever"):
                    plan["restricciones"].setdefault("retriever", {})["fijar"] = \
                        stack_g["retriever"]
                    fijadas.append(
                        f"retriever={stack_g['retriever'].get('retriever_name')}")
            print(f"[dirigido] {banco} se medira sobre el stack CORONADO del "
                  f"documento: {', '.join(fijadas)}")
        else:
            print(f"[dirigido] sin stack coronado en memoria para este doc: "
                  f"{banco} usara bases exploratorias (local). Para medir sobre "
                  f"el pipeline real, corre primero una corrida completa.")

    meta = MetaSupervisor()
    stack = meta.ejecutar(pdf, plan=plan, forzar=True) or {}

    # Resumen: si hubo Fase B (empate de finalistas), SU coronacion es la
    # respuesta — no el lider provisional de la Fase A (bug visto en Annie:
    # el resumen decia recursive_250 cuando la final corono a token_250).
    run_id = next((s.get("run_id") for s in
                   reversed(mem._leer().get("sesiones", []))
                   if s.get("doc_hash") == doc_hash), None)
    cer = mem.certeza(run_id, banco, CLAVE_DECISION[banco]) if run_id else {}
    ic_a = cer.get("ic_primaria") or {}          # la clave real de certeza()
    fase_b = stack.get("fase_b") or {}
    resumen = {
        "documento": Path(pdf).name,
        "banco": banco,
        "etiquetas_probadas": etiquetas,
        "n_runs": plan["n_runs"],
        "fase_a": {"lider": cer.get("ganador"), "resuelto": cer.get("resuelto"),
                   "ic_primaria": [ic_a.get("ic_bajo"), ic_a.get("ic_alto")],
                   "media": ic_a.get("media")},
        "nota": "resultados persistidos en memoria: el observatorio ya los ve",
    }
    if fase_b.get("campeon"):
        comp = fase_b.get("comparacion") or {}
        resumen["coronado"] = {
            "campeon": fase_b.get("campeon"), "mrr": fase_b.get("mrr"),
            "ic": [(fase_b.get("ic") or {}).get("ic_bajo"),
                   (fase_b.get("ic") or {}).get("ic_alto")],
            "concluyente": comp.get("concluyente"),
            "via": "fase_b (empate de finalistas dirimido en la final)"}
        resumen["respuesta"] = fase_b.get("campeon")
    else:
        resumen["respuesta"] = cer.get("ganador")
    return resumen
