"""
PLAN — el contrato entre el usuario y el meta-supervisor.

Que bancos correr, con que presupuesto (runs/preguntas), en que regimen
(ceteris paribus) y con que objetivo. Incluye el parser de instrucciones en
lenguaje natural y el pipeline BASE de la Fase A (el control fijo sobre el que
se varia un banco cada vez).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..bancos import NOMBRES_BANCOS

# El uso principal del laboratorio: buscar el optimo del TRIO ESTRUCTURAL
# (splitter, embedding, retriever) — los que mueven la aguja en el corpus visto.
# metadata y reranker (aditivos) NO corren por defecto: se anaden con un plan
# explicito, idealmente sobre el trio ya resuelto. Coherente con la evidencia
# (metadata->baseline; no_reranker competitivo) y con "banco no en el plan = no
# se usa". Para los cinco: Plan(bancos=list(NOMBRES_BANCOS)).
BANCOS_ESTRUCTURALES = ["splitter", "embedding", "retriever"]
from ..bancos.base import extraer_json
from ..config import CONFIG
from ..fabricas import construir_llm_clasificador, construir_llm_juez


# --------------------------------------------------------------------------- #
# El plan: que experimentos queremos hacer. Se maneja como diccionario dentro
# del grafo (para que el checkpointer lo serialice sin problemas) y como
# dataclass comoda de cara al usuario que prefiera construirlo desde codigo.
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    bancos: list[str] = field(default_factory=lambda: list(BANCOS_ESTRUCTURALES))
    n_runs: int = 2
    objetivo: str = "maxima calidad de recuperacion con un coste razonable"
    hacer_lda: bool = True
    hacer_umap: bool = True
    presupuesto: dict = field(default_factory=lambda: {"max_runs": 5, "max_preguntas": 40})
    ceteris_paribus: bool = True
    # restricciones por banco:
    #   {"splitter": {"fijar": {"name": "recursive", "chunk_size": 500, "chunk_overlap": 50}},
    #    "retriever": {"familias": ["bm25", "ensemble"]},
    #    "embedding": {"etiquetas": ["bge-m3"]}}
    # fijar   -> el banco NO experimenta; su config sustituye a la del baseline.
    # familias/etiquetas -> el banco experimenta solo dentro del subconjunto.
    restricciones: dict = field(default_factory=dict)
    # Puntos de control HITL (opt-in). Vacio => corre de corrido (batch intacto).
    # Validos: "tras_preguntas", "tras_diagnostico", "antes_fase_b", "tras_fase_b".
    pausas: list = field(default_factory=list)

    def como_dict(self) -> dict:
        return {"bancos": self.bancos, "n_runs": self.n_runs, "objetivo": self.objetivo,
                "hacer_lda": self.hacer_lda, "hacer_umap": self.hacer_umap,
                "presupuesto": self.presupuesto, "ceteris_paribus": self.ceteris_paribus,
                "restricciones": self.restricciones, "pausas": self.pausas}


def plan_por_defecto() -> dict:
    """Plan autonomo: los cuatro bancos, dos runs, con LDA y UMAP."""
    return Plan().como_dict()


def normalizar_plan(bruto: dict) -> dict:
    """
    Sanea un plan venga de donde venga (LLM, usuario, edicion HITL).

    Garantiza que los bancos existan, esten en el orden del pipeline, y que el
    resto de campos tengan valores razonables. Asi ningun nodo posterior recibe
    un plan malformado.
    """
    base = plan_por_defecto()
    bancos = [b for b in bruto.get("bancos", base["bancos"]) if b in NOMBRES_BANCOS]
    bancos = [n for n in NOMBRES_BANCOS if n in bancos] or list(NOMBRES_BANCOS)
    # n_runs ya NO esta capado a 3: es el numero MINIMO de runs. La reactividad
    # (T4) puede escalar hasta `presupuesto.max_runs` cuando un banco sigue en
    # disputa por un motivo arreglable con runs. El tope es un presupuesto, no un
    # numero magico.
    n_runs = max(1, min(10, int(bruto.get("n_runs", base["n_runs"]))))
    presup = bruto.get("presupuesto") or {}
    mr = presup.get("max_runs", bruto.get("max_runs", 5))
    max_runs = max(n_runs, min(20, int(mr)))
    mp = presup.get("max_preguntas", bruto.get("max_preguntas", 40))
    max_preguntas = max(int(CONFIG.n_preguntas_test), min(60, int(mp)))
    return {
        "bancos": bancos,
        "n_runs": n_runs,
        "presupuesto": {"max_runs": max_runs, "max_preguntas": max_preguntas},
        "objetivo": bruto.get("objetivo", base["objetivo"]),
        "hacer_lda": bool(bruto.get("hacer_lda", base["hacer_lda"])),
        "hacer_umap": bool(bruto.get("hacer_umap", base["hacer_umap"])),
        "ceteris_paribus": bool(bruto.get("ceteris_paribus", base.get("ceteris_paribus", True))),
        "restricciones": dict(bruto.get("restricciones") or {}),
        "pausas": list(bruto.get("pausas") or []),
    }


class _PlanLLM(BaseModel):
    """Esquema con el que el LLM propone o interpreta un plan."""
    bancos: list[str] = Field(description="Subconjunto de: splitter, embedding, "
                              "retriever, metadata, reranker. POR DEFECTO usa solo "
                              "['splitter','embedding','retriever'] (el trio que decide "
                              "el optimo); anade 'metadata' o 'reranker' SOLO si el "
                              "usuario lo pide explicitamente.")
    n_runs: int = Field(description="Numero MINIMO de runs por banco (1-10); la "
                        "reactividad puede escalar mas si hay disputa arreglable con runs")
    max_runs: int = Field(default=5, description="Presupuesto: maximo de runs si un "
                          "banco sigue en disputa por motivo arreglable con runs (>= n_runs)")
    max_preguntas: int = Field(default=40, description="Presupuesto: maximo de preguntas "
                          "de test si un banco sigue en disputa por IC ancho (margen_no_concluyente)")
    objetivo: str = Field(description="Que priorizar: coste, calidad, velocidad...")
    hacer_lda: bool = Field(description="Si conviene el analisis de temas LDA")
    hacer_umap: bool = Field(description="Si conviene la proyeccion UMAP")


def plan_desde_instruccion(instruccion: str) -> dict:
    """Convierte la instruccion inicial del usuario en un plan preliminar."""
    if not instruccion or not instruccion.strip():
        return plan_por_defecto()
    try:
        llm = construir_llm_clasificador().with_structured_output(_PlanLLM)
        sistema = (
            "Traduce la instruccion del usuario a un plan de experimentos para un "
            "laboratorio RAG. Bancos posibles: splitter, embedding, metadata, retriever, "
            "reranker (metadata prueba fusionar contenido y metadatos). Si el usuario no "
            "acota, incluye todos. n_runs entre 1 y 3."
        )
        propuesta = llm.invoke([SystemMessage(content=sistema),
                                HumanMessage(content=instruccion)])
        return normalizar_plan(propuesta.model_dump())
    except Exception as error:  # noqa: BLE001
        print(f"[meta] No pude interpretar la instruccion ({error}); uso plan por defecto.")
        return plan_por_defecto()




def pipeline_base() -> dict:
    """
    Pipeline BASE de la Fase A (ceteris paribus): componentes de comportamiento
    conocido sobre los que se varia UN banco cada vez SIN heredar del anterior.
    Asi el ruido de un banco no contamina la medicion de otro (mata la
    propagacion de error). El embedding base es el neutro de CONFIG; el resto,
    elecciones estables y razonables como control.
    """
    return {
        # Alineado con la etiqueta `recursive_1000` del banco de splitters
        # (overlap = 10%): asi el baseline ES una config medible del banco y los
        # veredictos del splitter describen tambien al pipeline base. Antes
        # (overlap 200) el baseline era una config fantasma que no aparecia en
        # ninguna parrilla (418 vs 449 chunks con el mismo nombre "recursive").
        "splitter": {"name": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
        "embedding": dict(CONFIG.embedding_neutro),
        "retriever": {"retriever_name": "similarity", "params": {}},
    }


