"""
Meta-supervisor.

Es el agente que dirige a los cuatro bancos como ayudantes. Esta construido como
un grafo LangGraph y trabaja en un abanico de modos que van de lo totalmente
autonomo a lo totalmente guiado por el usuario:

  - Autonomo puro      : se le da un PDF y decide solo (plan por defecto).
  - Por iniciativa      : consulta la memoria, se diagnostica, y propone los
                          experimentos que cree que mejoraran el sistema RAG
                          (por ejemplo, repetir los bancos inestables).
  - Human-in-the-loop   : antes de gastar tiempo y dinero, se detiene y te ensena
                          el diagnostico y el plan propuesto para que lo apruebes,
                          lo edites a tu gusto, o lo canceles.
  - Plan fijado         : le pasas un plan concreto desde codigo y lo ejecuta tal
                          cual, sin proponer ni preguntar.

Flujo del grafo:

    planificar -> preparar_sesion -> analizar_texto(LDA) -> diagnosticar
        -> proponer_plan -> revisar_plan [PAUSA HITL]
        -> ejecutar_bancos -> proyectar_embeddings(UMAP) -> sintetizar

Detalle importante de diseno: el estado que circula por el grafo es ligero y
serializable (cadenas y diccionarios). Los objetos pesados (el documento cargado,
los indices FAISS) NO viven en el estado; se guardan en una cache interna del
meta-supervisor indexada por run_id. Asi el checkpointer que exige el HITL no
tiene que serializar nada problematico.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # nombre alternativo en algunas versiones
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver

from ..analisis import analizar_temas, proyectar_embeddings
from ..bancos import (
    BancoEmbeddings,
    BancoMetadata,
    BancoRerankers,
    BancoRetrievers,
    BancoSplitters,
)
from ..bancos.base import extraer_json
from ..bancos.embeddings_banco import EMBEDDING_REGISTRY, _construir_splitter
from ..corpus import Sesion, preparar_sesion, hash_pdf, generar_preguntas_test
from .fase_b import correr_fase_b, BANCOS_FACTORIAL
from ..fabricas import construir_embedding, construir_llm_clasificador, construir_llm_juez
from ..config import CONFIG
from pathlib import Path

from ..memory import Memoria
from ..perfil import descriptor

# PIPELINE/NOMBRES_BANCOS/CLAVE_DECISION viven en bancos/__init__ y se
# re-exportan aqui para compatibilidad (notebooks importan de meta).
from ..bancos import PIPELINE, NOMBRES_BANCOS, CLAVE_DECISION



from .plan import (Plan, _PlanLLM, plan_por_defecto, normalizar_plan,
                   plan_desde_instruccion, pipeline_base)
from .sintesis import stack_economico_det
from .fase_b import producto_finalistas

# Traduccion de veredictos a configs: EXTRAIDA a supervisor/stacks.py (fase 1
# del refactor). Aqui solo quedan los delegados de compatibilidad.
from .informes import diagnostico_a_texto, informe_sintesis
from .stacks import (config_desde_veredicto, config_embedding, config_retriever,
                     config_splitter, stack_desde_certezas)

_config_splitter = config_splitter
_config_embedding = config_embedding
_config_retriever = config_retriever


# --------------------------------------------------------------------------- #
# Estado del grafo (ligero y serializable)
# --------------------------------------------------------------------------- #
class MetaEstado(TypedDict, total=False):
    pdf_path: str
    instruccion: str
    plan: dict
    plan_fijado: bool
    run_id: str
    diagnostico: dict
    veredictos: dict
    runs_ejecutados: int
    runs_ampliacion: int
    stack_final: dict
    saltar_fase_b: bool
    _veto_fase_b: list


class MetaSupervisor:
    """Construye y ejecuta el grafo de orquestacion con HITL e iniciativa propia."""

    def __init__(self, memoria: Memoria | None = None):
        self.memoria = memoria or Memoria()
        # Cache de sesiones: aqui viven los objetos pesados (documentos cargados),
        # fuera del estado del grafo. Clave = run_id.
        self._sesiones: dict[str, Sesion] = {}
        self.grafo = self._construir_grafo()

    # ====================== NODOS ====================== #
    def _nodo_planificar(self, estado: MetaEstado) -> dict:
        """Plan preliminar. Si el usuario fijo un plan, no tocamos nada."""
        if estado.get("plan_fijado"):
            print("[meta] Plan fijado por el usuario; no propongo ni pregunto.")
            return {}
        plan = plan_desde_instruccion(estado.get("instruccion", ""))
        return {"plan": plan}

    def _nodo_preparar_sesion(self, estado: MetaEstado) -> dict:
        """
        Carga y clasifica el documento. Guarda la sesion en la cache interna.

        EXAMEN CONGELADO: si este documento (mismo doc_hash) ya tiene un examen
        verificado en memoria, se reutiliza tal cual: corridas repetidas miden
        contra las MISMAS preguntas. Si el examen es nuevo, se verifica (capas
        de corpus.verificar_preguntas) y se congela.

        Pausa HITL opt-in 'tras_preguntas' (interrupt al principio del bloque
        HITL: el nodo se re-ejecuta al reanudar, por eso la sesion se cachea por
        doc_hash de forma idempotente): el humano ve el examen con su informe de
        verificacion y puede vetar preguntas por indice o por texto exacto.
        """
        # Cache POR INVOCACION (ejecutar/iniciar la vacian al arrancar): LangGraph
        # re-ejecuta este nodo entero al reanudar la pausa de abajo; sin cache,
        # cada reanudacion recargaba el PDF (~5s en documentos grandes).
        if not hasattr(self, "_cache_preparar"):
            self._cache_preparar = {}
        sesion = self._cache_preparar.get(estado["pdf_path"])
        congeladas = None
        if sesion is None:
            congeladas = self.memoria.preguntas_congeladas(
                hash_pdf(estado["pdf_path"]) or "")
            sesion = preparar_sesion(estado["pdf_path"],
                                     preguntas_predefinidas=congeladas)
            self._cache_preparar[estado["pdf_path"]] = sesion
        else:
            congeladas = (sesion.verificacion or {}).get("congelado") and \
                         sesion.preguntas_test
        self._sesiones[sesion.run_id] = sesion

        decision = self._pausa(estado, "tras_preguntas", {
            "mensaje": "Examen de la sesion. Revisa las preguntas verificadas y "
                       "veta las que no te convenzan (por indice o texto exacto).",
            "congelado_reutilizado": bool(congeladas),
            "tasa_verificacion": (sesion.verificacion or {}).get("tasa"),
            "preguntas": [{"i": i, "question": q.get("question", ""),
                           "expected_answer": q.get("expected_answer", ""),
                           "cita": (q.get("source_context", "") or "")[:120]}
                          for i, q in enumerate(sesion.preguntas_test)],
            "informe_verificacion": (sesion.verificacion or {}).get("informe"),
            "como_responder": {"seguir": {},
                               "vetar": {"vetar": ["<indice o texto exacto>"]}},
        })
        veto = decision.get("vetar") or []
        if veto:
            marcas = {str(v) for v in veto}
            sesion.preguntas_test = [
                q for i, q in enumerate(sesion.preguntas_test)
                if str(i) not in marcas and q.get("question", "") not in marcas]
            print(f"[meta] {len(veto)} pregunta(s) vetadas en la pausa; quedan "
                  f"{len(sesion.preguntas_test)}.")

        # Congelar el examen definitivo (nuevo, o re-congelar si hubo veto).
        if (not congeladas or veto) and sesion.doc_hash:
            self.memoria.congelar_preguntas(sesion.doc_hash, sesion.preguntas_test,
                                            info=sesion.verificacion)

        self.memoria.registrar_sesion(sesion.run_id,
                                      pdf_nombre=Path(sesion.pdf_path).name,
                                      doc_hash=sesion.doc_hash,
                                      perfil=sesion.perfil or {},
                                      n_preguntas=len(sesion.preguntas_test))
        return {"run_id": sesion.run_id}

    def _nodo_analizar_texto(self, estado: MetaEstado) -> dict:
        """
        Caracterizacion semantica del documento: LDA (temas) y UMAP (silueta),
        AMBOS ANTES del diagnostico, para que el perfil este completo cuando se
        calculen los priors. UMAP se mide SIEMPRE sobre el pipeline base neutro:
        la silueta es una feature del DOCUMENTO y debe ser comparable entre
        documentos, no depender de que configuracion gano.
        """
        sesion = self._sesiones[estado["run_id"]]
        if estado["plan"].get("hacer_lda", True):
            print("[meta] Analizando temas del documento (LDA)...")
            resultado = analizar_temas(sesion.documentos)
            disp = (resultado or {}).get("dispersion_tematica")
            if disp is not None:
                (sesion.perfil or {}).update({"dispersion_lda": disp})
                self.memoria.actualizar_perfil(sesion.run_id, {"dispersion_lda": disp})
            if resultado.get("disponible"):
                print(f"[meta] LDA: dispersion tematica={resultado['dispersion_tematica']}")
        if estado["plan"].get("hacer_umap", True):
            base = pipeline_base()
            # PROYECTOR DEL PERFIL = EL ARBITRO (2026-07-17): la silueta es
            # instrumentacion (alimenta perfil, vecinos y la conjetura
            # w<->silueta), no una medicion de banco — y el instrumento fijo
            # del laboratorio es el arbitro (local, gratis, sin API). Antes:
            # el neutro de pago. ERA DE PERFILES: las siluetas historicas se
            # midieron con el neutro y NO son conmensurables con las nuevas
            # hasta re-perfilar los documentos con el arbitro.
            from ..evaluacion import obtener_embedding_juez
            print(f"[meta] Proyectando embeddings con UMAP "
                  f"(arbitro, vara fija de perfil)...")
            try:
                splitter = _construir_splitter(base["splitter"])
                chunks = splitter.split_documents(sesion.documentos)
                modelo = obtener_embedding_juez()
                resultado = proyectar_embeddings(chunks, modelo)
            except Exception as e:  # noqa: BLE001
                print(f"[meta] UMAP fallo (sigo sin silueta): {e}")
                resultado = {}
            sil = (resultado or {}).get("separacion_silueta",
                                        (resultado or {}).get("silueta"))
            if sil is not None:
                (sesion.perfil or {}).update({"silueta_umap": sil})
                self.memoria.actualizar_perfil(sesion.run_id, {"silueta_umap": sil})
            if resultado.get("disponible"):
                print(f"[meta] UMAP: separacion semantica={resultado['calidad_espacio']} "
                      f"(silueta={resultado['separacion_silueta']})")
        return {}

    def _nodo_diagnosticar(self, estado: MetaEstado) -> dict:
        """
        Mira la memoria y se hace un diagnostico: que sabe ya de este tipo de
        documento, que componentes estan estables y cuales hacen aguas. Esta es
        la base de su iniciativa.
        """
        sesion = self._sesiones[estado["run_id"]]
        plan = estado["plan"]
        restric = plan.get("restricciones") or {}
        activos = [b for b in plan["bancos"]
                   if not (restric.get(b) or {}).get("fijar")]
        diag = self._diagnosticar(estado["run_id"], activos)
        print(self._diagnostico_a_texto(diag))
        decision = self._pausa(estado, "tras_diagnostico", {
            "mensaje": "Revisa los priors antes de experimentar. Puedes inyectar "
                       "restricciones por banco (fijar/familias/etiquetas).",
            "diagnostico": diag,
            "plan_actual": estado["plan"],
            "como_responder": {"seguir": {}, "restringir": {"restricciones": {"<banco>": {"...": "..."}}}},
        })
        if decision.get("restricciones"):
            nuevo = {**estado["plan"],
                     "restricciones": {**(estado["plan"].get("restricciones") or {}),
                                       **decision["restricciones"]}}
            print(f"[meta] Restricciones inyectadas en la pausa: "
                  f"{list(decision['restricciones'])}")
            return {"diagnostico": diag, "plan": nuevo}
        return {"diagnostico": diag}

    def _nodo_proponer_plan(self, estado: MetaEstado) -> dict:
        """
        Refina el plan usando el diagnostico de memoria. Aqui es donde el agente
        toma la iniciativa: si ve componentes inestables o sin datos, propone
        repetirlos; si todo esta solido, propone confirmar o no insistir.
        """
        if estado.get("plan_fijado"):
            return {}
        plan = self._refinar_plan(estado.get("instruccion", ""),
                                  estado["plan"], estado["diagnostico"])
        print(f"[meta] Plan propuesto: bancos={plan['bancos']} | runs={plan['n_runs']} "
              f"| objetivo='{plan['objetivo']}'")
        return {"plan": plan}

    def _nodo_revisar_plan(self, estado: MetaEstado) -> dict:
        """
        Punto de human-in-the-loop. Se detiene y entrega al humano el diagnostico
        y el plan propuesto. El humano responde con un diccionario:

            {"accion": "aprobar"}                      -> ejecuta el plan propuesto
            {"accion": "editar", "plan": {...}}         -> ejecuta el plan editado
            {"accion": "cancelar"}                      -> no ejecuta nada

        Si el plan venia fijado por el usuario, no interrumpimos.

        Nota: el interrupt va al PRINCIPIO del nodo porque, al reanudar, LangGraph
        vuelve a ejecutar el nodo desde el inicio.
        """
        if estado.get("plan_fijado"):
            return {}

        decision = interrupt({
            "mensaje": "Revisa el plan antes de ejecutar.",
            "diagnostico": estado["diagnostico"],
            "plan_propuesto": estado["plan"],
            "como_responder": {
                "aprobar": {"accion": "aprobar"},
                "editar": {"accion": "editar", "plan": "<plan a tu gusto>"},
                "cancelar": {"accion": "cancelar"},
            },
        })

        decision = decision or {"accion": "aprobar"}
        accion = decision.get("accion", "aprobar")
        if accion == "editar" and decision.get("plan"):
            print("[meta] Plan editado por el usuario.")
            return {"plan": normalizar_plan(decision["plan"])}
        if accion == "cancelar":
            print("[meta] Ejecucion cancelada por el usuario.")
            return {"plan": {**estado["plan"], "bancos": []}}
        print("[meta] Plan aprobado.")
        return {}

    def _memo(self) -> dict:
        """
        Memo POR INVOCACION (ejecutar/iniciar lo vacian). Los nodos con pausa se
        re-ejecutan enteros al reanudar; todo lo CARO o CON EFECTOS que ocurra
        antes del interrupt debe memoizarse aqui o se paga/duplica dos veces
        (el juez narraba dos justificaciones distintas y la decision se
        registraba dos veces en memoria; la Fase B re-evaluaba la final entera).
        """
        if not hasattr(self, "_memo_inv"):
            self._memo_inv = {}
        return self._memo_inv

    # ----------------- HITL por puntos de control (opt-in) ----------------- #
    def _pausa(self, estado: dict, punto: str, payload: dict) -> dict:
        """
        Punto de control HITL. Solo interrumpe si el plan declaro esta pausa en
        `pausas` (opt-in): sin ella, el grafo corre de corrido y el batch queda
        intacto. Devuelve la decision del revisor (o {} si no hubo pausa).
        Interrupt al principio: LangGraph re-ejecuta el nodo al reanudar.
        """
        if punto not in (estado.get("plan", {}).get("pausas") or []):
            return {}
        return interrupt({"punto": punto, **payload}) or {}

    # ----------------- TRAMO 4: reactividad guiada por el IC ----------------- #
    def _reaccion(self, certezas: dict, runs_hechos: int, n_preguntas: int,
                  presupuesto: dict) -> dict:
        """
        Decide la siguiente palanca tras una tanda, con criterio explícito y por
        ORDEN de coste: primero RUNS (baratos) para lo arreglable con runs
        (`desacuerdo`/`pocos_runs`); si ya no quedan, PREGUNTAS para lo que está
        en disputa por IC ancho (`margen_no_concluyente`), que es lo único que
        estrecha el IC. Para cuando todo está resuelto, no hay palanca útil, o se
        agota el presupuesto (esos finalistas pasan a la Fase B). Devuelve también
        los `bancos` objetivo, para escalar SOLO donde hace falta.
        """
        run_fixable = {"desacuerdo_entre_runs", "pocos_runs"}
        disponibles = [c for c in certezas.values() if c.get("disponible")]
        sin_resolver = [c for c in disponibles if not c.get("resuelto")]
        if not sin_resolver:
            return {"accion": "parar", "bancos": [], "razon": "todos los bancos resueltos"}

        # PRIORIDAD AL ESTRECHAMIENTO DEL IC: mas preguntas primero. Con pocas
        # preguntas base, un IC ancho produce tanto `margen_no_concluyente` como
        # `desacuerdo_entre_runs` (el remuestreo de 10 preguntas baila); mas
        # preguntas atacan la raiz de AMBOS. Los runs solo remuestrean lo que ya
        # hay: no estrechan el IC. Por eso preguntas > runs.
        deficit = int(presupuesto.get("max_preguntas", n_preguntas)) - n_preguntas
        # LOTE MINIMO: ampliar el examen re-ejecuta TODA la parrilla de todos los
        # bancos en disputa; hacerlo por +1 pregunta (+2-3% de datos) es pagar
        # una tanda entera por un estrechamiento del IC insignificante. Solo
        # merece la pena si el lote mueve la aguja: >=3 preguntas y >=10% mas.
        lote_minimo = max(3, round(0.1 * n_preguntas))
        if deficit >= lote_minimo:
            bancos = sorted(str(c.get("banco")) for c in sin_resolver)
            return {"accion": "mas_preguntas", "bancos": bancos,
                    "razon": f"{len(bancos)} en disputa ({', '.join(bancos)}); "
                             "prioridad: estrecho el IC con mas preguntas de test"}
        if 0 < deficit < lote_minimo:
            print(f"[meta] Palanca de preguntas descartada: solo caben +{deficit} "
                  f"(lote minimo {lote_minimo}); no compensa re-ejecutar la parrilla.")

        # Respaldo: agotadas las preguntas, si queda desacuerdo entre runs (ya con
        # IC estrecho), mas runs pueden consolidar la estabilidad.
        por_runs = [c for c in sin_resolver if c.get("motivo") in run_fixable]
        if por_runs and runs_hechos < int(presupuesto.get("max_runs", runs_hechos)):
            bancos = sorted(str(c.get("banco")) for c in por_runs)
            return {"accion": "mas_runs", "bancos": bancos,
                    "razon": f"{len(bancos)} con desacuerdo e IC ya estrecho "
                             f"({', '.join(bancos)}); consolido con mas runs"}

        return {"accion": "parar", "bancos": [],
                "razon": f"presupuesto agotado; {len(sin_resolver)} en disputa -> Fase B"}

    def _nodo_ejecutar_bancos(self, estado: MetaEstado) -> dict:
        """
        Dirige los bancos del plan con REACTIVIDAD de DOS palancas (Fase A.2):
        corre al menos `n_runs`; luego, mientras haya disputa y presupuesto,
        escala RUNS (para `desacuerdo`/`pocos_runs`) o, cuando ya no hay nada
        arreglable con runs, PREGUNTAS (para `margen_no_concluyente`, que es lo
        unico que estrecha el IC). Para cuando todo se resuelve o se agotan ambos
        presupuestos (esos finalistas pasan a la Fase B). En Fase A (ceteris
        paribus) el pipeline base queda FIJO: sin herencia, sin propagacion.
        """
        plan = estado["plan"]
        sesion = self._sesiones[estado["run_id"]]
        veredictos: dict = {}
        clases = dict(PIPELINE)
        presupuesto = plan.get("presupuesto") or {"max_runs": max(plan["n_runs"], 5),
                                                  "max_preguntas": 40}
        # Un banco corre si (y solo si) esta en el plan. La metadata es un banco
        # normal mas: si el plan la incluye, experimenta como los demas.
        # Un banco FIJADO por el usuario tampoco experimenta: su config es una
        # orden, no una hipotesis (y no genera certeza ni predicciones).
        restricciones = plan.get("restricciones") or {}
        sesion.restricciones = restricciones
        fijados = {b: (r or {}).get("fijar") for b, r in restricciones.items()
                   if isinstance(r, dict) and (r or {}).get("fijar")}
        for b in fijados:
            print(f"[meta] Restriccion: {b} FIJADO por el usuario -> {fijados[b]}")
        bancos_plan = [n for n in NOMBRES_BANCOS
                       if n in plan["bancos"] and n not in fijados]
        minimo = plan["n_runs"]
        maximo = int(presupuesto.get("max_runs", minimo))
        max_preg = int(presupuesto.get("max_preguntas", len(sesion.preguntas_test)))
        ceteris = plan.get("ceteris_paribus", True)
        if ceteris:
            base = pipeline_base()
            # Un banco estructural FIJADO sustituye a su pieza del baseline: los
            # demas bancos se miden SOBRE la eleccion del usuario.
            if fijados.get("splitter"):
                f = fijados["splitter"]
                base["splitter"] = {"name": f.get("name") or f.get("splitter", "recursive"),
                                    "chunk_size": f.get("chunk_size", 1000),
                                    "chunk_overlap": f.get("chunk_overlap", 100)}
            if fijados.get("embedding"):
                f = fijados["embedding"]
                base["embedding"] = {"model": f.get("model"),
                                     "provider": f.get("provider", "openai")}
            if fijados.get("retriever"):
                f = fijados["retriever"]
                base["retriever"] = {"retriever_name": f.get("retriever_name")
                                     or f.get("nombre", "similarity"),
                                     "params": f.get("params", {})}
            sesion.config_splitter = base["splitter"]
            sesion.config_embedding = base["embedding"]
            sesion.config_retriever = base["retriever"]
            print("[meta] Fase A: ceteris paribus (pipeline base fijo, sin herencia "
                  "entre bancos -> sin propagacion de error).")

        def _correr(run: int, completo: bool) -> None:
            for nombre in bancos_plan:
                banco = clases[nombre](self.memoria)
                veredicto = banco.ejecutar(sesion, run=run, n_runs=maximo,
                                           usar_set_completo=completo)
                veredictos[nombre] = veredicto
                if not ceteris:   # solo en modo herencia el ganador pasa al siguiente
                    if nombre == "splitter":
                        sesion.config_splitter = _config_splitter(veredicto)
                    elif nombre == "embedding":
                        sesion.config_embedding = _config_embedding(veredicto)
                    elif nombre == "retriever":
                        sesion.config_retriever = _config_retriever(veredicto)

        run = 0
        while run < minimo:                     # fase inicial: los runs minimos
            run += 1
            print(f"\n[meta] ===== RUN {run} (min {minimo}, max {maximo}) =====")
            _correr(run, completo=False)

        runs_ampliacion = 0                     # tandas extra por ampliar el examen
        while True:                             # fase reactiva (T4 runs + A.2 preguntas)
            certezas = self._certezas_por_banco(estado["run_id"], bancos_plan)
            n_preg = len(sesion.preguntas_test)
            reaccion = self._reaccion(certezas, run, n_preg, presupuesto)
            print(f"[meta] Reactividad: {reaccion['razon']} -> {reaccion['accion']}")
            if reaccion["accion"] == "parar":
                break
            run += 1
            if reaccion["accion"] == "mas_preguntas":
                nuevas = generar_preguntas_test(sesion, max(0, max_preg - n_preg))
                if not nuevas:
                    print("[meta] Sin preguntas nuevas; detengo la escalada.")
                    break
                sesion.preguntas_test = list(sesion.preguntas_test) + nuevas
                # El examen crecio: re-congelar para que futuras corridas del
                # documento hereden tambien las preguntas ampliadas.
                if nuevas and sesion.doc_hash:
                    self.memoria.congelar_preguntas(sesion.doc_hash,
                                                    sesion.preguntas_test,
                                                    info=sesion.verificacion)
                runs_ampliacion += 1
                print(f"\n[meta] ===== RUN {run}: +{len(nuevas)} preguntas de test "
                      f"(total {len(sesion.preguntas_test)}, max {max_preg}) =====")
                _correr(run, completo=True)     # evalua sobre el set AMPLIADO completo
            else:                               # mas_runs
                print(f"\n[meta] ===== RUN {run} (min {minimo}, max {maximo}) =====")
                _correr(run, completo=False)

        print(f"[meta] Ejecucion terminada: {run} run(s), "
              f"{len(sesion.preguntas_test)} preguntas de test.")
        return {"veredictos": veredictos, "runs_ejecutados": run,
                "runs_ampliacion": runs_ampliacion}

    # ----------------- TRAMO 2: sintesis basada en CERTEZA ----------------- #
    def _certezas_por_banco(self, run_id: str, bancos: list[str]) -> dict:
        """Certeza POR DOCUMENTO (T1) de cada banco del plan."""
        return {n: self.memoria.certeza(run_id, n, CLAVE_DECISION[n]) for n in bancos}

    def _config_desde_veredicto(self, banco: str, veredicto: dict) -> dict:
        return config_desde_veredicto(banco, veredicto)

    def _stack_desde_certezas(self, certezas: dict, veredictos: dict) -> dict:
        return stack_desde_certezas(certezas, veredictos)

    def _nodo_sintetizar(self, estado: MetaEstado) -> dict:
        """
        Sintesis por CERTEZA, determinista: el stack recomendado es el lider
        evaluado por banco (los empates los dirime la Fase B) y el economico es
        el mas barato dentro del empate. El LLM ya NO elige configuraciones:
        solo NARRA (confianza global, justificacion, alertas) a la vista de la
        certeza, el perfil y los priors. Menos libertad = menos que validar.

        FASE 2 del refactor (2026-07-16): nodo troceado en metodos con nombre
        (mismo fichero, mismas costuras, comportamiento byte a byte — golden).
        El orquestador de abajo ES el indice del nodo.
        """
        sesion = self._sesiones[estado["run_id"]]
        plan = estado["plan"]
        veredictos = estado.get("veredictos", {})
        if ("sintesis_registrada", sesion.run_id) not in self._memo():
            print("\n[meta] Sintetizando el stack final (por certeza)...")

        certezas = self._certezas_por_banco(estado["run_id"], plan["bancos"])
        lineas = self._lineas_certeza(certezas)
        stack_rec, stack_eco = self._ensamblar_stacks(certezas, veredictos, plan)
        hay_disputa = any(not c.get("resuelto")
                          for c in certezas.values() if c.get("disponible"))
        datos = self._narrar_sintesis(sesion, plan, lineas, stack_rec, stack_eco)
        confianza, alertas = self._confianza_y_alertas(datos, hay_disputa)
        stack = self._payload_sintesis(certezas, stack_rec, stack_eco,
                                       confianza, alertas, datos)
        self._registrar_sintesis(sesion, estado, plan, certezas, stack, lineas)
        return self._pausa_antes_fase_b(estado, stack)

    @staticmethod
    def _lineas_certeza(certezas: dict) -> list[str]:
        """Las lineas 'CERTEZA POR BANCO' del informe (formato compartido)."""
        lineas = []
        for nombre, cer in certezas.items():
            if not cer.get("disponible"):
                lineas.append(f"  {nombre}: SIN DATOS ({cer.get('motivo')})")
                continue
            ic = cer.get("ic_primaria") or {}
            ic_txt = (f"IC[{ic.get('ic_bajo')},{ic.get('ic_alto')}] ampl={ic.get('amplitud')}"
                      if ic else "IC n/d")
            lineas.append(
                f"  {nombre}: {'RESUELTO' if cer.get('resuelto') else 'EN DISPUTA'} | "
                f"lider={cer.get('ganador')} | acuerdo={cer.get('acuerdo')} | "
                f"concluyente={cer.get('concluyente')} | {cer.get('metrica_primaria')} "
                f"{ic_txt} | motivo={cer.get('motivo')}")
        return lineas

    def _ensamblar_stacks(self, certezas: dict, veredictos: dict,
                          plan: dict) -> tuple[dict, dict]:
        """Lider evaluado por banco + fijados del usuario; y el economico."""
        stack_rec = self._stack_desde_certezas(certezas, veredictos)
        # Los bancos FIJADOS por el usuario entran al stack tal cual, marcados:
        # no se midieron (fueron una orden), asi que sin certeza ni disputa.
        for b, r in (plan.get("restricciones") or {}).items():
            f = (r or {}).get("fijar")
            if f and b not in stack_rec:
                stack_rec[b] = {"etiqueta": f.get("etiqueta") or f.get("name")
                                or f.get("model") or f.get("nombre") or "fijado",
                                "config": f, "resuelto": True, "en_disputa": False,
                                "via": "fijado_por_usuario"}
        return stack_rec, stack_economico_det(certezas)

    def _narrar_sintesis(self, sesion, plan: dict, lineas: list[str],
                         stack_rec: dict, stack_eco: dict) -> dict:
        """La narracion del LLM (confianza/justificacion/alertas). MEMOIZADA:
        la pausa antes_fase_b re-ejecuta el nodo al reanudar y, sin memo, el
        juez se llamaba DOS veces y redactaba dos justificaciones distintas."""
        p = sesion.perfil or {}
        datos = dict(self._memo().get(("narracion", sesion.run_id), {}))
        try:
          if not datos:
              sistema = (
                  f"Eres el meta-supervisor de un laboratorio RAG.\n"
                  f"Objetivo del usuario: {plan['objetivo']}.\n"
                  f"Perfil del documento (deterministico): {p}\n\n"
                  "CERTEZA POR BANCO (evidencia medida en ESTE documento):\n"
                  + "\n".join(lineas) + "\n\n"
                  f"STACK (lider evaluado por banco): "
                  f"{ {b: v.get('etiqueta') for b, v in stack_rec.items()} }\n"
                  f"ALTERNATIVA ECONOMICA (mas barata dentro del empate): "
                  f"{ {b: v.get('etiqueta') for b, v in stack_eco.items()} }\n\n"
                  "Tu papel es NARRAR, no elegir: el stack ya esta decidido por la "
                  "evidencia (y los empates los dirime la fase factorial).\n"
                  "Reglas: la confianza global NO puede ser 'alta' si hay bancos EN "
                  "DISPUTA; se honesto con los IC anchos; alertas concretas.\n\n"
                  "Responde UNICAMENTE con JSON puro:\n"
                  "{\n"
                  '  "confianza_global": "alta|media|baja",\n'
                  '  "justificacion": "...",\n'
                  '  "alertas": ["..."]\n'
                  "}"
              )
              respuesta = construir_llm_juez().invoke(
                  [SystemMessage(content=sistema),
                   HumanMessage(content="Redacta confianza, justificacion y alertas.")])
              datos = extraer_json(respuesta.content) or {}
              self._memo()[("narracion", sesion.run_id)] = dict(datos)
        except Exception as e:  # noqa: BLE001
            print(f"[meta] Narracion LLM fallo (sigo con sintesis determinista): {e}")
        return datos

    @staticmethod
    def _confianza_y_alertas(datos: dict, hay_disputa: bool) -> tuple[str, list]:
        """La regla dura sobre la narracion: con disputa no hay confianza alta."""
        confianza = datos.get("confianza_global", "baja")
        alertas = list(datos.get("alertas") or [])
        if hay_disputa and confianza == "alta":
            confianza = "media"
            alertas.append("Confianza recortada a 'media': hay bancos en disputa.")
        return confianza, alertas

    @staticmethod
    def _payload_sintesis(certezas: dict, stack_rec: dict, stack_eco: dict,
                          confianza: str, alertas: list, datos: dict) -> dict:
        """El payload completo de la sintesis (lo que se persiste y se devuelve)."""
        pendiente_fase_b = {b: [f["etiqueta"] for f in (c.get("finalistas") or [])]
                            for b, c in certezas.items()
                            if c.get("disponible") and c.get("es_empate")}
        return {
            "stack_recomendado": stack_rec,
            "stack_alternativo_economico": stack_eco,
            "confianza_global": confianza,
            "justificacion": datos.get("justificacion", ""),
            "alertas": alertas,
            "certezas": {b: {k: c.get(k) for k in
                             ("ganador", "resuelto", "concluyente", "acuerdo",
                              "motivo", "ic_primaria", "n_finalistas")}
                         for b, c in certezas.items()},
            "pendiente_fase_b": pendiente_fase_b,
        }

    def _registrar_sintesis(self, sesion, estado: MetaEstado, plan: dict,
                            certezas: dict, stack: dict, lineas: list[str]) -> None:
        """Efectos (una sola vez, memoizados): decision en memoria, resolucion
        de predicciones con el DIFERIDO del splitter, e informe por pantalla."""
        if ("sintesis_registrada", sesion.run_id) in self._memo():
            return
        # Con efectos y/o ruido: solo en la PRIMERA ejecucion del nodo (la
        # re-ejecucion tras la pausa duplicaba la decision en memoria y
        # reimprimia el informe con otra redaccion).
        self.memoria.registrar_decision(sesion.run_id, stack)
        ganadores = {b: c.get("ganador") for b, c in certezas.items()
                     if c.get("disponible") and c.get("ganador")}
        # El prior de SPLITTER se califica contra la CORONACION de Fase B,
        # no contra el lider de banco: el prior predice el coronado tipico
        # de los vecinos, y la ley fino-gana-banco hace que el lider de
        # banco sea sistematicamente otro (el 0% exacto historico era un
        # desajuste de contabilidad, no mala punteria). Si hay Fase B
        # pendiente, se difiere alli; si no la habra, se resuelve aqui.
        if stack.get("pendiente_fase_b") and "splitter" in ganadores:
            ganadores = {b: g for b, g in ganadores.items() if b != "splitter"}
        res = self.memoria.resolver_predicciones(estado["run_id"], ganadores)
        if res.get("resueltas"):
            print(f"[meta] Priors comprobados: {res['aciertos']}/{res['resueltas']} "
                  f"exactos | {res.get('aciertos_familia', 0)}/{res['resueltas']} "
                  f"por familia.")
        print(self._informe(sesion, plan, lineas, stack, estado.get("runs_ejecutados"),
                            estado.get("runs_ampliacion", 0) or 0))
        self._memo()[("sintesis_registrada", sesion.run_id)] = True

    def _pausa_antes_fase_b(self, estado: MetaEstado, stack: dict) -> dict:
        """La pausa HITL previa al cruce: seguir, saltar o vetar candidatos."""
        decision = self._pausa(estado, "antes_fase_b", {
            "mensaje": "Stack sintetizado. Antes del cruce factorial puedes vetar "
                       "candidatos (por etiqueta) o saltarte la Fase B.",
            "stack": stack.get("stack_recomendado"),
            "pendiente_fase_b": stack.get("pendiente_fase_b"),
            "como_responder": {"seguir": {}, "saltar_fase_b": {"saltar": True},
                               "vetar": {"vetar": ["<etiqueta>", "..."]}},
        })
        if decision.get("saltar"):
            print("[meta] Fase B omitida por el usuario en la pausa.")
            stack["fase_b"] = {"omitida_por_usuario": True}
            return {"stack_final": stack, "saltar_fase_b": True}
        if decision.get("vetar"):
            vetados = set(decision["vetar"])
            stack["pendiente_fase_b"] = {b: [e for e in ets if e not in vetados]
                                         for b, ets in (stack.get("pendiente_fase_b") or {}).items()}
            print(f"[meta] Candidatos vetados en la pausa: {sorted(vetados)}")
        return {"stack_final": stack, "_veto_fase_b": list(decision.get("vetar") or [])}

    # ------------------- FASE B: final factorial end-to-end ------------------ #
    def _nodo_fase_b(self, estado: MetaEstado) -> dict:
        """
        Dirime los EMPATES reales que la Fase A no resolvio: arma el producto de
        finalistas (solo bancos del factorial; metadata se fija en v1), evalua cada
        pipeline COMPLETO sobre el set comun (las preguntas ampliadas de la sesion,
        recalculando desde cero) y corona por comparacion pareada. Si no hay
        empates reales, pasa de largo sin tocar la sintesis. Guardado de fallos.

        FASE 3 del refactor (2026-07-16): nodo troceado en metodos con nombre
        (mismas costuras, comportamiento congelado en el golden de fase B).
        """
        stack = estado.get("stack_final", {})
        if estado.get("saltar_fase_b") or not (stack.get("pendiente_fase_b") or {}):
            return {}
        sesion = self._sesiones[estado["run_id"]]
        candidatos = self._candidatos_fase_b(estado)
        if len(candidatos) <= 1:
            print("[fase_b] Sin empates reales que dirimir; se mantiene la sintesis.")
            return {}
        veredicto = self._veredicto_fase_b(sesion, candidatos)
        if veredicto is None:
            return {}
        campeon = veredicto.get("campeon")
        if not campeon:
            return {}
        self._coronar_en_stack(stack, campeon)
        self._economico_post_final(stack, veredicto, campeon)
        self._sellar_fase_b(stack, veredicto, campeon)
        return self._persistir_coronacion(estado, stack)

    def _candidatos_fase_b(self, estado: MetaEstado) -> list:
        """Producto de finalistas de los bancos del factorial, respetando los
        candidatos vetados en la pausa 'antes_fase_b'."""
        certezas = {b: self.memoria.certeza(estado["run_id"], b, CLAVE_DECISION[b])
                    for b in BANCOS_FACTORIAL if b in estado["plan"]["bancos"]}
        vetados = set(estado.get("_veto_fase_b") or [])
        if vetados:
            for b, cer in certezas.items():
                cer["finalistas"] = [f for f in (cer.get("finalistas") or [])
                                     if f.get("etiqueta") not in vetados]
        return producto_finalistas(certezas)

    def _veredicto_fase_b(self, sesion, candidatos: list) -> dict | None:
        """La final factorial, MEMOIZADA: la pausa tras_fase_b re-ejecuta este
        nodo; sin memo, la final entera (todos los pipelines x todas las
        preguntas) se evaluaba DOS veces. None si la final fallo."""
        clave_fb = ("fase_b", sesion.run_id)
        veredicto = self._memo().get(clave_fb)
        if veredicto is None:
            try:
                veredicto = correr_fase_b(sesion, candidatos)
            except Exception as e:  # noqa: BLE001
                print(f"[fase_b] Fase B fallo; se mantiene la sintesis: {e}")
                return None
            self._memo()[clave_fb] = veredicto
        return veredicto

    @staticmethod
    def _coronar_en_stack(stack: dict, campeon: dict) -> None:
        """COHERENCIA del stack persistido (el observatorio y DocuAgent tratan
        la memoria como verdad; esta era la fuente de tres incoherencias vistas
        en produccion):
          1. `via` distingue COMO se resolvio cada banco: 'fase_b' solo si la
             final lo dirimio de verdad (tenia >1 finalistas pendientes); un
             banco resuelto en la Fase A que entro como pieza fijada conserva
             via='fase_a' (fiabilidades distintas: ver tension primaria/MRR).
          2. Los flags de la vista economica se sincronizan con el estado
             post-final (decian 'en_disputa' de un banco ya dirimido).
          3. La justificacion (narrada ANTES de la final) recibe una coda con
             el desenlace (ver _sellar_fase_b)."""
        dirimidos = set((stack.get("pendiente_fase_b") or {}).keys())
        for banco, fin in (campeon.get("combo") or {}).items():
            if banco in stack.get("stack_recomendado", {}):
                stack["stack_recomendado"][banco] = {
                    "etiqueta": fin.get("etiqueta"), "config": fin.get("config"),
                    "resuelto": True, "en_disputa": False,
                    "via": "fase_b" if banco in dirimidos else "fase_a"}

    @staticmethod
    def _economico_post_final(stack: dict, veredicto: dict, campeon: dict) -> None:
        """ECONOMICO POST-FINAL: el de la sintesis era "lo mas barato dentro del
        empate de cada BANCO" (pre-final) y tras la coronacion quedaba obsoleto
        (en Annie salio identico al recomendado: oferta vacia). La pregunta
        correcta ahora es: el PIPELINE mas barato dentro del empate_final —
        estadisticamente indistinguible del campeon y mas barato de operar
        (menos k = menos contexto que pagar aguas abajo)."""
        empate_ets = set(veredicto.get("empate") or [])
        empatados = [r for r in (veredicto.get("ranking") or [])
                     if r.get("etiqueta") in empate_ets and r.get("combo")]
        if len(empatados) > 1:
            def _coste(r):
                combo = r.get("combo") or {}
                emb = (combo.get("embedding") or {}).get("config") or {}
                ret = (combo.get("retriever") or {}).get("config") or {}
                coste_emb = float(emb.get("cost_per_1k_tokens") or 0.0)
                if (ret.get("nombre") or "").casefold() == "bm25":
                    # bajo bm25 puro el embedding NO se usa (misma premisa que
                    # el dedupe lexico): no cobrarselo. Sin este cero, la ganga
                    # real (bm25: sin indice denso, sin llamadas de embedding)
                    # perdia contra ensembles por un coste fantasma.
                    coste_emb = 0.0
                return (coste_emb, int(ret.get("k") or 0),
                        -float(r.get("mrr") or 0.0))
            barato = min(empatados, key=_coste)
            if barato.get("etiqueta") != campeon.get("etiqueta"):
                stack["stack_alternativo_economico"] = {
                    banco: {"etiqueta": fin.get("etiqueta"),
                            "config": fin.get("config"), "en_disputa": False,
                            "dentro_de": "empate_final"}
                    for banco, fin in (barato.get("combo") or {}).items()}
                stack["stack_alternativo_economico"]["_nota"] = (
                    f"pipeline mas barato del empate de la final "
                    f"(mrr={barato.get('mrr'):.3f} vs campeon "
                    f"{campeon.get('mrr'):.3f}, no concluyente)")
            else:
                stack.pop("stack_alternativo_economico", None)
        else:
            # sin empate real: no hay alternativa que ofrecer
            stack.pop("stack_alternativo_economico", None)

    @staticmethod
    def _sellar_fase_b(stack: dict, veredicto: dict, campeon: dict) -> None:
        """El payload fase_b, la coda de la justificacion (sin ella, la prosa
        persistida contradecia al stack) y el ascenso de confianza si la final
        fue concluyente."""
        comp = veredicto.get("comparacion") or {}
        stack["fase_b"] = {
            "campeon": campeon.get("etiqueta"), "mrr": campeon.get("mrr"),
            "ic": veredicto.get("ic"), "empate_final": veredicto.get("empate"),
            "n_pipelines": veredicto.get("n_evaluados"), "comparacion": comp,
            "clones_del_lider": veredicto.get("clones_del_lider") or [],
        }
        ic_fb = veredicto.get("ic") or {}
        coda = (f" [FASE B] La final factorial ({veredicto.get('n_evaluados')} "
                f"pipelines sobre el examen comun) corono a "
                f"{campeon.get('etiqueta')} (mrr={campeon.get('mrr'):.3f}, "
                f"IC=[{ic_fb.get('ic_bajo')}, {ic_fb.get('ic_alto')}], "
                f"{'concluyente' if comp.get('concluyente') else 'no concluyente'}"
                f"{' frente a ' + str(comp.get('rival')) if comp.get('rival') else ''}); "
                f"esta coronacion sustituye a los lideres provisionales del parrafo anterior.")
        stack["justificacion"] = (stack.get("justificacion") or "") + coda
        if comp.get("concluyente") and stack.get("confianza_global") == "baja":
            stack["confianza_global"] = "media"
        print(f"\n[fase_b] STACK FINAL (coronado end-to-end): "
              f"{campeon.get('etiqueta')} | mrr={campeon.get('mrr'):.3f} | "
              f"concluyente={comp.get('concluyente')}")

    def _persistir_coronacion(self, estado: MetaEstado, stack: dict) -> dict:
        """Pausa 'tras_fase_b' (aprobar o vetar ANTES de persistir, para que un
        coronamiento rechazado no contamine memoria ni priors), persistencia de
        la decision CON la coronacion (sin esto, la memoria se quedaba con el
        stack pre-Fase B) y el prior de splitter DIFERIDO calificado contra el
        CORONADO (ver sintesis: fino-gana-banco)."""
        decision = self._pausa(estado, "tras_fase_b", {
            "mensaje": "Stack coronado end-to-end. Aprueba para persistir, o veta "
                       "para descartar la coronacion de Fase B.",
            "campeon": stack.get("fase_b"),
            "stack": stack.get("stack_recomendado"),
            "como_responder": {"aprobar": {}, "vetar": {"vetar": True}},
        })
        if decision.get("vetar"):
            print("[meta] Coronacion de Fase B vetada por el usuario: no se persiste.")
            return {}
        self.memoria.registrar_decision(estado["run_id"], stack)
        spl_coronado = ((stack.get("stack_recomendado") or {})
                        .get("splitter") or {}).get("etiqueta")
        if spl_coronado:
            res = self.memoria.resolver_predicciones(
                estado["run_id"], {"splitter": spl_coronado})
            if res.get("resueltas"):
                print(f"[meta] Prior de splitter comprobado contra la "
                      f"CORONACION ({spl_coronado}): "
                      f"{res['aciertos']}/{res['resueltas']} exactos.")
        return {"stack_final": stack}

    # ====================== RUTAS ====================== #
    def _ruta_post_revision(self, estado: MetaEstado) -> str:
        """Si el plan se quedo sin bancos (cancelado), terminamos sin ejecutar."""
        return "ejecutar_bancos" if estado["plan"].get("bancos") else END

    # ====================== AYUDA: DIAGNOSTICO E INICIATIVA ====================== #
    def _diagnosticar(self, run_id: str, bancos_activos: list[str] | None = None) -> dict:
        """
        Diagnostico por PRIORS DE PERFIL: los k documentos mas parecidos (perfil
        numerico, no etiqueta) y, por banco, que gano alli, cuanto costo
        resolverlo (frac_resuelto, ic_medio) y el motivo dominante. Es la unica
        vista entre-documentos: sesga el plan (B1), nunca decide por el.
        Cada prior emitido se REGISTRA como prediccion, para medir despues si
        acerto (tasa_acierto): el aprendizaje es comprobable.
        """
        sesion = self._sesiones[run_id]
        perfil = sesion.perfil or {}
        bancos = {}
        for nombre in NOMBRES_BANCOS:
            pr = self.memoria.priors(perfil, nombre, CLAVE_DECISION[nombre],
                                     excluir_run_id=run_id,
                                     excluir_doc_hash=sesion.doc_hash)
            bancos[nombre] = pr
            # Solo se PREDICE lo que se va a medir: un banco fuera del plan (o
            # fijado) no tendra ganador contra el que resolver -> prediccion
            # huerfana que ensucia el diario. Se muestra el prior, no se apuesta.
            if bancos_activos is not None and nombre not in bancos_activos:
                continue
            if pr.get("disponible") and pr.get("ganador") and pr.get("acuerdo", 0) >= 0.5:
                self.memoria.registrar_prediccion(
                    run_id, nombre, pr["ganador"],
                    base={"acuerdo": pr.get("acuerdo"),
                          "n_vecinos": pr.get("n_vecinos"),
                          "sim_media": pr.get("sim_media")},
                    autorreferencial=bool(pr.get("autorreferencial")))
        return {"perfil": perfil, "descriptor": descriptor(perfil), "bancos": bancos}

    def _refinar_plan(self, instruccion: str, preliminar: dict, diagnostico: dict) -> dict:
        """
        El agente decide, a la vista del diagnostico, que conviene ejecutar para
        mejorar el sistema. Delegamos el juicio en el LLM, pero pasandole hechos
        concretos de la memoria. Ante cualquier fallo, devolvemos el plan
        preliminar (degradacion segura).
        """
        try:
            llm = construir_llm_juez().with_structured_output(_PlanLLM)
            sistema = (
                "Eres el meta-supervisor de un laboratorio RAG. A partir del diagnostico "
                "de memoria y de la instruccion del usuario, decide que bancos ejecutar y "
                "cuantos runs para RESOLVER el sistema (que cada banco tenga un ganador "
                "concluyente).\n"
                "Heuristicas:\n"
                "- Prioriza bancos SIN datos o con 'frac_resuelto' bajo (rara vez se "
                "resuelven): necesitan mas evidencia.\n"
                "- El 'motivo_dominante' dice la palanca correcta: 'margen_no_concluyente' "
                "o 'amplitud_ic_media' grande -> el IC es ancho, harian falta MAS PREGUNTAS "
                "de test; 'desacuerdo_entre_runs' o 'pocos_runs' -> MAS RUNS.\n"
                "- Un banco con 'frac_resuelto' alto NO necesita mas, salvo que el usuario "
                "lo pida.\n"
                "- El diagnostico incluye un PRONOSTICO por vecinos (documentos parecidos): "
                "uselo como pista para anticipar configuraciones, pero NUNCA como certeza; "
                "si su confianza es baja o nula, trata el banco como inexplorado.\n"
                "- Respeta cualquier acotacion explicita de la instruccion.\n"
                f"Diagnostico de memoria: {diagnostico}\n"
                f"Plan preliminar (de la instruccion): {preliminar}"
            )
            humano = instruccion or "Decide tu el mejor plan para mejorar el sistema."
            propuesta = llm.invoke([SystemMessage(content=sistema),
                                    HumanMessage(content=humano)])
            return normalizar_plan(propuesta.model_dump())
        except Exception as error:  # noqa: BLE001
            print(f"[meta] No pude refinar el plan ({error}); uso el preliminar.")
            return preliminar

    # ====================== MONTAJE ====================== #
    def _construir_grafo(self):
        c = StateGraph(MetaEstado)
        c.add_node("planificar", self._nodo_planificar)
        c.add_node("preparar_sesion", self._nodo_preparar_sesion)
        c.add_node("analizar_texto", self._nodo_analizar_texto)
        c.add_node("diagnosticar", self._nodo_diagnosticar)
        c.add_node("proponer_plan", self._nodo_proponer_plan)
        c.add_node("revisar_plan", self._nodo_revisar_plan)
        c.add_node("ejecutar_bancos", self._nodo_ejecutar_bancos)
        c.add_node("sintetizar", self._nodo_sintetizar)
        c.add_node("fase_b", self._nodo_fase_b)

        c.add_edge(START, "planificar")
        c.add_edge("planificar", "preparar_sesion")
        c.add_edge("preparar_sesion", "analizar_texto")
        c.add_edge("analizar_texto", "diagnosticar")
        c.add_edge("diagnosticar", "proponer_plan")
        c.add_edge("proponer_plan", "revisar_plan")
        # Tras la revision: ejecutar, o terminar si se cancelo.
        c.add_conditional_edges("revisar_plan", self._ruta_post_revision,
                                ["ejecutar_bancos", END])
        c.add_edge("ejecutar_bancos", "sintetizar")
        c.add_edge("sintetizar", "fase_b")
        c.add_edge("fase_b", END)

        # El checkpointer es obligatorio para poder pausar en el HITL.
        return c.compile(checkpointer=MemorySaver())

    # ====================== API PUBLICA ====================== #
    def plan_de_entrenamiento(self) -> dict:
        """
        Introspeccion simple sobre la memoria nueva: cuantos documentos hay,
        que tasa de acierto tienen los priors por banco y cuantas predicciones
        siguen sin resolver. (El plan fino por zonas frias volvera cuando el
        perfil acumule corpus.)
        """
        return {"resumen": self.memoria.resumen(),
                "tasa_acierto_priors": self.memoria.tasa_acierto()}

    def ejecutar(self, pdf_path: str, instruccion: str = "", plan: Plan | dict | None = None,
                revisor: Callable[[dict], dict] | None = None,
                thread_id: str | None = None, forzar: bool = False) -> dict:
        """
        Ejecuta el meta-supervisor y devuelve el stack final.

        Parametros
        ----------
        pdf_path    : ruta del documento a estudiar.
        instruccion : texto en lenguaje natural. Vacio -> modo autonomo/iniciativa.
        plan        : si lo pasas (Plan o dict), se ejecuta tal cual, sin proponer
                      ni preguntar (modo "plan fijado").
        revisor     : funcion human-in-the-loop. Recibe el payload de la pausa
                      (diagnostico + plan propuesto) y devuelve la decision
                      ({"accion": "aprobar"|"editar"|"cancelar", ...}). Si es None,
                      se aprueba automaticamente (sin humano).
        thread_id   : identificador de la conversacion para el checkpointer.
        forzar      : si True, reprocesa aunque el documento ya se haya estudiado.

        Deduplicacion: si el documento ya fue estudiado y COMPLETADO antes (misma
        huella de contenido y con decision registrada), NO se reprocesa: se
        devuelve el stack que ya se decidio (con `_ya_estudiado=True`). Asi el
        batch nocturno es idempotente y no malgasta tiempo ni dinero. `forzar=True`
        lo salta.

        Para un human-in-the-loop manual (por ejemplo, desde una interfaz), usa
        en su lugar `iniciar` y `reanudar`.
        """
        if not forzar:
            ya = self.memoria.documento_estudiado(doc_hash=hash_pdf(pdf_path),
                                                  pdf_path=pdf_path)
            if ya:
                print(f"[meta] '{ya['pdf_nombre'] or pdf_path}' ya estudiado el "
                      f"{ya['estudiado_en']} (run {ya['run_id']}). Lo omito; usa "
                      f"forzar=True para reprocesar.")
                return {**ya.get("stack", {}), "_ya_estudiado": True,
                        "_run_id_original": ya["run_id"]}

        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        entrada: MetaEstado = {"pdf_path": pdf_path, "instruccion": instruccion,
                               "plan_fijado": plan is not None}
        if plan is not None:
            entrada["plan"] = plan.como_dict() if isinstance(plan, Plan) else normalizar_plan(plan)

        self._cache_preparar = {}      # sesion fresca por invocacion (ver nodo)
        self._memo_inv = {}            # memo de nodos con pausa (ver _memo)
        estado = self.grafo.invoke(entrada, config)
        # Bucle de reanudacion: mientras el grafo se pause, pedimos decision y seguimos.
        while estado.get("__interrupt__"):
            payload = estado["__interrupt__"][0].value
            decision = revisor(payload) if revisor else {"accion": "aprobar"}
            # NUNCA reanudar con un valor FALSY: un resume {} puede tratarse como
            # "sin resume" -> el interrupt no lo consume, el nodo re-pausa y el
            # bucle se vuelve infinito (las pausas cuyo "seguir" es {} lo
            # disparaban; revisar_plan nunca, porque su decision es truthy).
            estado = self.grafo.invoke(
                Command(resume=decision or {"seguir": True}), config)
        return estado.get("stack_final", {})

    # --- API de bajo nivel para HITL manual (interfaces, notebooks) --- #
    def iniciar(self, pdf_path: str, instruccion: str = "",
               thread_id: str | None = None) -> tuple[str, dict | None]:
        """
        Arranca la ejecucion y la detiene en la revision del plan.

        Devuelve (thread_id, payload). `payload` es lo que hay que ensenar al
        humano (diagnostico + plan propuesto); si es None, no hubo pausa.
        """
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        self._cache_preparar = {}
        self._memo_inv = {}
        estado = self.grafo.invoke(
            {"pdf_path": pdf_path, "instruccion": instruccion, "plan_fijado": False}, config)
        if estado.get("__interrupt__"):
            return thread_id, estado["__interrupt__"][0].value
        return thread_id, None

    def reanudar(self, thread_id: str, decision: dict) -> dict:
        """Reanuda una ejecucion pausada con la decision del humano. Devuelve el stack final."""
        config = {"configurable": {"thread_id": thread_id}}
        estado = self.grafo.invoke(Command(resume=decision or {"seguir": True}), config)
        while estado.get("__interrupt__"):
            # No deberia volver a pausar, pero por robustez aprobamos si reaparece.
            estado = self.grafo.invoke(Command(resume={"accion": "aprobar"}), config)
        return estado.get("stack_final", {})

    # ====================== INFORMES ====================== #
    def _diagnostico_a_texto(self, diag: dict) -> str:
        return diagnostico_a_texto(diag, self.memoria.tasa_acierto())

    def _informe(self, sesion: Sesion, plan: dict, lineas: list[str], stack: dict,
                 runs_ejecutados: int | None = None,
                 runs_ampliacion: int = 0) -> str:
        return informe_sintesis(sesion, plan, lineas, stack,
                                runs_ejecutados, runs_ampliacion)
