"""
Banco de retrievers: que estrategia de recuperacion encuentra mejor lo relevante.

La variable aqui es el retriever. Para que la comparacion sea justa, el indice
vectorial se construye UNA sola vez con el splitter y el embedding heredados, y
todas las estrategias buscan sobre ese mismo indice.

La malla de experimentos reproduce la del banco original:
  - similarity : k_final en {3, 5, 10}
  - mmr        : k_final en {5, 10} x lambda_mult en {0.3, 0.5, 0.7}
  - bm25       : k_final en {3, 5, 10}
  - ensemble   : k_final en {5, 10} x peso_bm25 en {0.3, 0.5, 0.7}
"""

from __future__ import annotations

import time

import numpy as np
from langchain_community.vectorstores import FAISS

from .cache_documento import por_documento, huella_troceado
from ..metricas import CV_SIN_SENAL
from ..corpus import Sesion
from ..evaluacion import evaluar_sobre_preguntas
from ..fabricas import construir_embedding
from ..recuperacion import construir_recuperador
from .base import BancoBase
from .embeddings_banco import _construir_splitter

K_CANDIDATOS = 20


def _construir_malla() -> list[dict]:
    """Genera las 18 combinaciones de la malla, igual que el banco original."""
    experimentos = []
    for k in [3, 5, 10]:
        experimentos.append({"nombre": "similarity", "k": k, "params": {}})
    for k in [5, 10]:
        for lm in [0.3, 0.5, 0.7]:
            experimentos.append({"nombre": "mmr", "k": k,
                                 "params": {"lambda_mult": lm, "fetch_k": K_CANDIDATOS}})
    for k in [3, 5, 10]:
        experimentos.append({"nombre": "bm25", "k": k, "params": {}})
    for k in [5, 10]:
        for w in [0.3, 0.5, 0.7]:
            experimentos.append({"nombre": "ensemble", "k": k, "params": {"w_bm25": w}})
    return experimentos


EXPERIMENTOS_RETRIEVER = _construir_malla()


class BancoRetrievers(BancoBase):
    nombre = "retriever"
    clave_decision = "retriever_recomendado"
    descripcion = "estrategias de recuperacion (retrievers)"
    esquema_json = (
        "{\n"
        '  "retriever_recomendado": "<similarity|mmr|bm25|ensemble>",\n'
        '  "k_final": <int>,\n'
        '  "params": {"<param>": <valor>},\n'
        '  "confianza": "<alta|media|baja>",\n'
        '  "justificacion": "<2-3 frases basadas en los datos>",\n'
        '  "alertas": ["<aviso si hay algo preocupante>"]\n'
        "}"
    )
    # Primaria: MRR (es lo que un retriever optimiza: lo relevante arriba).
    # Secundarios: latencia y uniformidad entre preguntas.
    metrica_primaria = ("mrr", "max")
    # Calidad antes que coste: con latencia primero, bm25 (siempre el mas
    # rapido con diferencia) ganaba TODOS los desempates estadisticos en
    # silencio — sesgo sistematico visible en el historico de victorias.
    objetivos_secundarios = [("recuperacion_pasaje", "max"), ("relevancia_media", "max"),
                             ("latency", "min"), ("cv_variance", "min")]
    metricas_contexto = ("relevancia_media", "recuperacion_pasaje")

    def preparar_contexto(self, sesion: Sesion) -> dict:
        """Construye el indice una vez con splitter + embedding heredados."""
        splitter = _construir_splitter(sesion.config_splitter)
        chunks = splitter.split_documents(sesion.documentos)

        from .embeddings_banco import embedding_base_exploratorio
        config_emb = sesion.config_embedding or embedding_base_exploratorio()
        modelo = construir_embedding(config_emb)
        # Indice base cacheado por documento: identico entre runs (mismos chunks,
        # mismo embedding base); se construye una vez y se reutiliza.
        vectorstore = por_documento(
            "retriever", sesion.run_id,
            (huella_troceado(chunks), config_emb.get("model")),
            lambda: FAISS.from_documents(chunks, modelo))
        print(f"[retriever] Indice listo: {len(chunks)} fragmentos | "
              f"embedding base: {config_emb.get('model')}")
        return {"vectorstore": vectorstore, "chunks": chunks, "modelo": modelo}

    def _tendencias(self, resultados: list[dict],
                    ganador: dict | None = None) -> list[str]:
        import re as _re
        lineas = []
        met = self.metrica_primaria[0]
        valor = {}
        for r in resultados:
            et = (r.get("_config") or {}).get("etiqueta") or ""
            if r.get(met) is not None:
                valor[et] = float(r[met])

        def _serie(patron, extraer):
            grupos = {}
            for et, v in valor.items():
                m = _re.fullmatch(patron, et)
                if m:
                    grupos.setdefault(extraer(m), []).append((et, v))
            return grupos

        # evolucion con k por familia (similarity/bm25) y con w en ensembles
        for familia in ("similarity", "bm25"):
            ks = _serie(rf"{familia}_k(\d+)", lambda m: int(m.group(1)))
            if len(ks) >= 2:
                orden = sorted(ks)
                trazo = " -> ".join(f"k{k}:{ks[k][0][1]:.3f}" for k in orden)
                lineas.append(f"{familia} por k: {trazo}")
        for k in (5, 10):
            ws = _serie(rf"ensemble_k{k}_w([\d.]+)", lambda m: float(m.group(1)))
            if len(ws) >= 2:
                orden = sorted(ws)
                trazo = " -> ".join(f"w{w}:{ws[w][0][1]:.3f}" for w in orden)
                extra = ""
                mejor_w = max(orden, key=lambda w: ws[w][0][1])
                if mejor_w == orden[-1]:
                    extra = "  [BORDE: el mejor w es el maximo probado]"
                elif mejor_w == orden[0]:
                    extra = "  [BORDE: el mejor w es el minimo probado]"
                lineas.append(f"ensemble k{k} por w_bm25: {trazo}{extra}")
        return lineas

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        etiquetas = []
        for e in EXPERIMENTOS_RETRIEVER:
            sufijo = ""
            if "lambda_mult" in e["params"]:
                sufijo = f"_l{e['params']['lambda_mult']}"
            elif "w_bm25" in e["params"]:
                sufijo = f"_w{e['params']['w_bm25']}"
            etiquetas.append({"etiqueta": f"{e['nombre']}_k{e['k']}{sufijo}", **e})
        return etiquetas

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        recuperador = construir_recuperador(
            experimento["nombre"], experimento["params"],
            contexto["vectorstore"], contexto["chunks"], experimento["k"],
        )

        # LATENCIA = solo el tiempo de RECUPERACIÓN. Antes se cronometraba el
        # bucle de evaluación entero, que incluye el puntuado semántico con el
        # árbitro; como la caché del árbitro se comparte entre experimentos, el
        # primero pagaba los embeddings y los siguientes los encontraban ya
        # cacheados: la latencia (métrica de desempate) dependía del ORDEN de
        # la parrilla. Cronometrando dentro del wrapper solo se mide lo que el
        # retriever tarda de verdad.
        tiempos: list[float] = []

        def _recuperar(consulta: str):
            t0 = time.perf_counter()
            docs = recuperador.invoke(consulta)
            tiempos.append(time.perf_counter() - t0)
            return docs

        metricas = evaluar_sobre_preguntas(
            _recuperar,
            sesion.preguntas_test,
            contexto["modelo"],
        )
        metricas["latency"] = round(float(np.mean(tiempos)), 4) if tiempos else 0.0

        # CV de dispersión con centinela "sin señal" (ver metricas.CV_SIN_SENAL):
        # media 0 => no hay estabilidad que premiar; peor valor, no el mejor.
        relevancias = metricas.get("_relevancias", [])
        media = float(np.mean(relevancias)) if relevancias else 0.0
        metricas["cv_variance"] = (round(float(np.std(relevancias)) / media, 4)
                                   if media > 1e-9 else CV_SIN_SENAL)
        return metricas
