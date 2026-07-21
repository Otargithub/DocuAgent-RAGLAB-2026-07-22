"""
Grafos exportados a LangGraph Studio (ver guia_langgraph_studio.md).

REGLA DE ORO para Studio: exportar el grafo COMPILADO SIN checkpointer — el
servidor (`langgraph dev`) inyecta su propia persistencia; si compilas con
MemorySaver propio, el servidor lo ignora/avisa. Por eso este módulo construye
DocuAgent aparte en vez de reutilizar el compile() de los notebooks.

DocuAgent es el ciudadano ideal de Studio: sus tools son numpy sobre la memoria
(baratas y sin GPU), así que puedes darle a "Run" cien veces sin remordimiento.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag_lab.memory import Memoria
from rag_lab.analisis import observatorio as obs

mem = Memoria()

SISTEMA = ("Eres DocuAgent, analista de experimentos RAG. Todo número sale de una "
           "herramienta; 'gana' solo si el duelo dice CONCLUYENTE=True; n siempre "
           "visible; hipótesis marcadas como hipótesis. Los nombres de las "
           "configuraciones NUNCA se adivinan: descubre las etiquetas exactas "
           "con etiquetas_disponibles antes de pedir rendimientos o duelos.")


@tool
def listar_documentos() -> str:
    """Documentos estudiados con su perfil. Úsala primero para conocer la evidencia."""
    filas = [f"- {s.get('pdf_nombre')} | perfil={(s.get('perfil') or {}).get('descriptor', '?')}"
             for s in mem._leer().get("sesiones", [])]
    return "\n".join(filas) or "No hay documentos estudiados."


@tool
def etiquetas_disponibles(banco: str) -> str:
    """Etiquetas EXACTAS con evidencia en un banco ('splitter', 'embedding',
    'retriever'...), con su numero de mediciones. USALA SIEMPRE antes de pedir
    rendimientos o duelos: los nombres deben ser exactos, nunca adivinados."""
    n = obs.etiquetas_con_evidencia(mem, banco)
    if not n:
        return f"Sin evidencia en el banco {banco}."
    return " | ".join(f"{et} (n={x})" for et, x in n.items())


@tool
def rendimiento_de_etiqueta(banco: str, etiqueta: str) -> str:
    """Media + IC bootstrap de una config sobre todas sus mediciones históricas."""
    r = obs.rendimiento(mem, banco, etiqueta)
    if not r.get("disponible"):
        return f"Sin mediciones de '{etiqueta}' en {banco}."
    return f"{etiqueta}: n={r['n']} media={r['media']} IC=[{r['ic_bajo']}, {r['ic_alto']}]"


@tool
def duelo(banco: str, a: str, b: str) -> str:
    """Comparación PAREADA con desglose por documento. Única fuente válida para
    afirmar que una config gana a otra."""
    d = obs.duelo(mem, banco, a, b)
    if not d.get("disponible"):
        return f"Evidencia insuficiente: {d.get('motivo')}"
    g = d["global"]
    return (f"{a} vs {b}: dif={g['dif_media']:+} IC=[{g['ic_bajo']}, {g['ic_alto']}] "
            f"CONCLUYENTE={g['concluyente']} | por doc: {d['por_documento']}")


@tool
def victorias_por_perfil(banco: str) -> str:
    """Tabla ganador x tipo de documento: patrones del corpus."""
    t = obs.victorias_por_perfil(mem, banco)
    return str(t) if t else f"Sin veredictos del banco {banco}."


TOOLS = [listar_documentos, etiquetas_disponibles, rendimiento_de_etiqueta,
         duelo, victorias_por_perfil]
_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0).bind_tools(TOOLS)


def _agente(estado: MessagesState) -> dict:
    return {"messages": [_llm.invoke([SystemMessage(content=SISTEMA)]
                                     + estado["messages"])]}


_c = StateGraph(MessagesState)
_c.add_node("agente", _agente)
_c.add_node("tools", ToolNode(TOOLS))
_c.add_edge(START, "agente")
_c.add_conditional_edges("agente", tools_condition)
_c.add_edge("tools", "agente")

# SIN checkpointer: lo pone el servidor de Studio.
docuagent = _c.compile()


# --------------------------------------------------------------------------- #
# DocuAgent INVESTIGADOR: puede LANZAR experimentos acotados, con peaje humano
# --------------------------------------------------------------------------- #
from langgraph.types import interrupt
from langchain_core.messages import ToolMessage

from rag_lab.experimento_dirigido import LIMITES, experimento_dirigido, plan_dirigido


@tool
def catalogo_de_experimentos(banco: str) -> str:
    """Bancos que se pueden experimentar y las etiquetas EXACTAS que cada uno
    admite (la parrilla completa, exista o no en memoria). USALA SIEMPRE antes
    de lanzar_experimento: el parametro 'banco' debe ser uno de la lista, y las
    'etiquetas' deben copiarse literalmente de aqui — nunca inventadas ni con
    nombres genericos tipo 'baseline'/'default' sin su prefijo."""
    from rag_lab.experimento_dirigido import LIMITES, catalogo_de_banco
    if banco not in LIMITES["bancos"]:
        return (f"Banco {banco!r} no valido. Bancos que se pueden experimentar: "
                f"{', '.join(LIMITES['bancos'])}. Llama de nuevo con uno de esos.")
    ets = catalogo_de_banco(banco)
    return f"Banco '{banco}' — etiquetas validas ({len(ets)}): " + ", ".join(ets)


@tool
def lanzar_experimento(pdf: str, banco: str, etiquetas: list[str],
                       n_runs: int = 1) -> str:
    """Lanza un experimento REAL y acotado del laboratorio: un solo banco, hasta
    4 etiquetas concretas, 1-2 runs, sobre el examen congelado del documento.
    CUESTA tiempo y dinero (minutos de GPU/API): úsala solo cuando el analisis
    de la evidencia existente sea insuficiente y una hipotesis concreta lo
    justifique. El resultado queda en memoria y las tools de analisis lo ven."""
    try:
        res = experimento_dirigido(pdf, banco, etiquetas, n_runs)
    except (ValueError, AssertionError) as e:
        return f"RECHAZADO: {e}"
    return str(res)


TOOLS_CARAS = {"lanzar_experimento"}
TOOLS_INVESTIGADOR = TOOLS + [catalogo_de_experimentos, lanzar_experimento]
_llm_inv = ChatAnthropic(model="claude-haiku-4-5-20251001",
                         temperature=0).bind_tools(TOOLS_INVESTIGADOR)

SISTEMA_INV = (SISTEMA + " Puedes lanzar experimentos con lanzar_experimento, "
               "pero SOLO cuando la evidencia existente no baste para responder "
               "y siempre formulando primero la hipotesis que el experimento "
               "va a comprobar. ANTES de lanzar_experimento llama SIEMPRE a "
               "catalogo_de_experimentos para elegir el banco correcto y copiar "
               "las etiquetas EXACTAS (nunca inventes nombres como 'default' o "
               "'baseline' sin su prefijo). Nota: etiquetas_disponibles muestra "
               "lo que hay EN MEMORIA (vacio = sin historia aun); "
               "catalogo_de_experimentos muestra lo que se PUEDE probar. "
               "Un empate no concluyente con n>=30 PREGUNTAS suele ser un "
               "empate REAL, no falta de datos: no lo persigas. OJO: no "
               "confundas 'runs' (repeticiones, topadas en 2 por el "
               "experimento dirigido) con 'preguntas' (el examen, ~40). Si un "
               "duelo dice 'solo 2 mediciones' NO puedes conseguir mas por "
               "diseno: decide con la diferencia observada. Diferencias "
               "triviales frente a la escala (p.ej. 0.10 vs 0.12) son empates: "
               "concluye que no hay mejora, no lances mas experimentos. "
               "DESCONFIA de un duelo con IC de anchura CERO (p.ej. "
               "IC=[0.0043, 0.0043]): no es certeza infinita, es que estas "
               "comparando runs deterministas identicos, y una diferencia de "
               "la 3a-4a cifra decimal NO es una mejora real por mucho que "
               "diga CONCLUYENTE. Reportala como empate.")


def _agente_inv(estado: MessagesState) -> dict:
    return {"messages": [_llm_inv.invoke([SystemMessage(content=SISTEMA_INV)]
                                         + estado["messages"])]}


def _peaje(estado: MessagesState) -> dict:
    """Antes de una tool cara: interrupt. El humano ve QUE quiere lanzar el
    agente y su coste estimado, y autoriza o deniega. Mismo patron que las
    pausas del meta, pero al nivel del agente (nunca HITL anidado en la tool)."""
    ultimo = estado["messages"][-1]
    caras = [tc for tc in (getattr(ultimo, "tool_calls", None) or [])
             if tc["name"] in TOOLS_CARAS]
    if not caras:
        return {}
    detalle = []
    for tc in caras:
        a = tc["args"]
        n_et = len(a.get("etiquetas") or [])
        runs = max(1, min(int(a.get("n_runs", 1)), LIMITES["max_runs"]))
        detalle.append({"peticion": f"{tc['name']}({a})",
                        "coste_estimado": f"~{n_et * runs} experimentos "
                                          f"(~{n_et * runs * 15}s + juez)"})
    decision = interrupt({
        "mensaje": "DocuAgent quiere LANZAR experimentos reales. ¿Autorizas?",
        "peticiones": detalle,
        "como_responder": {"autorizar": {}, "denegar": {"denegar": True}},
    })
    # Un resume falsy ({}) es una AUTORIZACION valida, no "sin resume": sin este
    # or, LangGraph puede re-ejecutar el interrupt en bucle (mismo bug que
    # cazamos en el meta con Command(resume={}) y dict vacio).
    decision = decision or {}
    if decision.get("denegar"):
        return {"messages": [ToolMessage(
            content="DENEGADO por el humano: no lances el experimento; razona "
                    "con la evidencia existente y di explicitamente que la "
                    "peticion fue denegada.",
            tool_call_id=caras[0]["id"])]}
    return {}


_ci = StateGraph(MessagesState)
_ci.add_node("agente", _agente_inv)
_ci.add_node("peaje", _peaje)
_ci.add_node("tools", ToolNode(TOOLS_INVESTIGADOR))
_ci.add_edge(START, "agente")
_ci.add_conditional_edges("agente", tools_condition, {"tools": "peaje", END: END})
_ci.add_conditional_edges(
    "peaje",
    lambda e: "tools" if getattr(e["messages"][-1], "tool_calls", None) else "agente")
_ci.add_edge("tools", "agente")

docuagent_investigador = _ci.compile()
