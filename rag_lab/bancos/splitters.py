"""
Banco de splitters: ¿qué forma de trocear el documento funciona mejor?

Probamos varias familias de splitter y varios tamaños de fragmento. Para que la
comparación sea justa, todos se evalúan con el MISMO embedding neutro (aquí la
variable es el splitter, no el embedding) y con estas dos reglas de equidad:

  1. TAMAÑOS HOMOGÉNEOS. La etiqueta expresa SIEMPRE caracteres: `token_500`
     significa ~500 caracteres (≈125 tokens), igual que `recursive_500`. Antes
     `token_500` eran 500 TOKENS (~2000 caracteres): mismo número en la
     etiqueta, granularidad 4x distinta — se comparaban peras con manzanas.

  2. RECUPERACIÓN A PRESUPUESTO IGUAL. k no es fijo en número de chunks sino en
     CARACTERES recuperados (PRESUPUESTO_CARACTERES): con k fijo, los chunks
     grandes recuperaban 4x más texto que los pequeños y el recall por unión les
     regalaba ventaja mecánica, no mérito. k = presupuesto / longitud media real.

El COSTE real de un splitter no es el tiempo de troceo (milisegundos) sino el
número de chunks que produce: más chunks = más embeddings que pagar y un índice
más caro para siempre. Por eso `n_chunks` y la longitud media se miden y se
reportan como contexto. El troceo se calcula UNA vez por documento y se cachea
(los runs solo remuestrean preguntas; el troceado no cambia), y su latencia
real se guarda junto a él: así la métrica `latency` es la misma en todos los
runs y no depende del orden de ejecución.

Familias: recursive/token/character (tamaños 500/1000/2000c), `page` (cada
página = un chunk; ganador de los benchmarks de NVIDIA para documentos
paginados, y nuestros PDF lo son) y `semantic` (SemanticChunker, si
langchain-experimental está instalado; corta por saltos semánticos usando el
embedding neutro — caro: su latencia real queda a la vista como objetivo).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ..config import CONFIG
from .cache_documento import por_documento, huella_troceado
from ..corpus import Sesion
from ..evaluacion import evaluar_sobre_preguntas
from ..fabricas import construir_embedding
from ..metricas import CV_SIN_SENAL, coeficiente_variacion
from .base import BancoBase

SPLITTERS = {
    "recursive": RecursiveCharacterTextSplitter,
    "token": TokenTextSplitter,
    "character": CharacterTextSplitter,
}
TAMANOS_FRAGMENTO = [125, 250, 500, 1000, 2000]  # SIEMPRE caracteres (ver docstring)
# 250 añadido tras Annie.pdf (diario aforistico): el lider era el tamaño
# MINIMO de la parrilla (borde) => faltaba probar mas fino. El dedupe por
# huella fusiona los 250 que coincidan con otros troceados.
# 125 añadido tras el dirigido de Annie del 2026-07-17 (primer dirigido de la
# era del banco base unificado): el borde VOLVIO a sonar — token_250 lider y
# minimo de su familia, Spearman n_chunks<->recuperacion_pasaje +0.50/+1.00.
# Misma medicina un escalon mas abajo. Nota: a ~125c el k-por-presupuesto
# toca su tope (40) y el presupuesto efectivo baja de 4000c: el tope ES la
# cota, como documenta el comentario del presupuesto.
FAMILIAS_CON_TAMANO = ("recursive", "token", "character")
PRESUPUESTO_CARACTERES = 4000                # ~ el antiguo k=8 x chunks de 500
CHARS_POR_TOKEN = 4                          # aproximación es/en para TokenTextSplitter


def _tokens_aprox(chars: int) -> int:
    return max(1, round(chars / CHARS_POR_TOKEN))


class _PageSplitter:
    """Identidad: cada página del loader es un chunk. Coste de troceo cero."""
    def split_documents(self, documentos):
        return list(documentos)


def _construir_splitter_banco(nombre: str, chunk_size: int, chunk_overlap: int,
                              modelo: Any = None):
    """Construye el splitter de una config. `modelo` solo lo usa `semantic`."""
    if nombre == "page":
        return _PageSplitter()
    if nombre == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker
        return SemanticChunker(modelo)
    if nombre == "token":
        # la config viene en caracteres; TokenTextSplitter habla en tokens
        return TokenTextSplitter(chunk_size=_tokens_aprox(chunk_size),
                                 chunk_overlap=_tokens_aprox(chunk_overlap))
    return SPLITTERS[nombre](chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _k_por_presupuesto(longitudes: list[int],
                       presupuesto: int = PRESUPUESTO_CARACTERES) -> int:
    """
    k tal que k x longitud_media ~ presupuesto de caracteres, acotado a [2, 20].
    Usa la longitud media REAL de los chunks (no la nominal): vale igual para
    page/semantic, que no tienen tamaño nominal.
    """
    if not longitudes:
        return 2
    media = sum(longitudes) / len(longitudes)
    k = round(presupuesto / max(1.0, media))
    # Tope alto (40->80, 2026-07-17): con chunks ultrafinos (~100c) el
    # presupuesto pide k~36 y el tope viejo (20) dejaba a esas configs con
    # MEDIO presupuesto (visto en Annie: token_250 con 2180c de 4000c). Al dar
    # de alta 125c el bug reaparecia un escalon mas abajo (~55c pide k~73 y el
    # tope de 40 lo dejaba otra vez a medias): mismo arreglo, mismo principio.
    # El presupuesto ES la cota de equidad; el tope solo protege de k absurdos.
    tope = min(80, len(longitudes))
    if k > tope:
        print(f"[splitter] k CAPADO a {tope} (el presupuesto pedia ~{k}): "
              f"compite con menos presupuesto efectivo.")
    return max(2, min(80, k, len(longitudes)))


def _sin_vacios(chunks: list) -> tuple[list, int]:
    """
    Filtra chunks VACIOS (o solo espacios). Un chunk vacio no puede ser
    recuperado con provecho pero SI cuesta: paga embedding, engorda el indice,
    infla n_chunks y — el efecto que lo delato — cambia la huella del troceado:
    `page` conservaba 9 paginas vacias que `character` fundia, y dos troceados
    funcionalmente identicos (metricas iguales a 4 decimales, 4 corridas
    seguidas) tenian huellas distintas y esquivaban el dedupe.
    """
    limpios = [c for c in chunks if (getattr(c, "page_content", "") or "").strip()]
    return limpios, len(chunks) - len(limpios)


def _semantic_disponible() -> bool:
    try:
        from langchain_experimental.text_splitter import SemanticChunker  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _dedupe_por_huella(experimentos: list[dict], documentos) -> list[dict]:
    """
    Elimina configs que producen EXACTAMENTE el mismo troceado (p. ej.
    CharacterTextSplitter ignora chunk_size si los parrafos ya lo superan:
    character_500/1000/2000 pueden ser tres etiquetas del MISMO pipeline).
    Sin esto, los clones empatan a la perfeccion, el liderazgo alterna entre
    ellos al azar entre runs (disputa fantasma) y la Fase B evalua duplicados.
    Se dedupe todo lo BARATO de trocear (familias con tamano y page, que es
    identidad); solo semantic queda fuera, porque trocear con el para comprobar
    la huella costaria minutos. Los chunks vacios se filtran ANTES de la huella
    (ver _sin_vacios): eran lo que hacia a page esquivar el dedupe.
    Guardado: ante cualquier fallo, no deduplica.
    """
    vistos, unicos, clones = {}, [], []
    for exp in experimentos:
        if exp["splitter"] == "semantic":
            # unico excluido: trocear con semantic solo para comprobar la
            # huella costaria minutos de embeddings. page SI entra: es gratis.
            unicos.append(exp)
            continue
        try:
            splitter = _construir_splitter_banco(exp["splitter"], exp["chunk_size"],
                                                 exp["chunk_overlap"])
            trozos, _ = _sin_vacios(splitter.split_documents(documentos))
            huella = huella_troceado(trozos)
        except Exception:  # noqa: BLE001
            unicos.append(exp)
            continue
        if huella in vistos:
            clones.append(f"{exp['etiqueta']}={vistos[huella]}")
        else:
            vistos[huella] = exp["etiqueta"]
            unicos.append(exp)
    if clones:
        print(f"[splitter] Dedupe por huella: {len(experimentos)} -> {len(unicos)} "
              f"configs (troceados identicos: {', '.join(clones)})")
    return unicos


class BancoSplitters(BancoBase):
    nombre = "splitter"
    clave_decision = "splitter_recomendado"
    descripcion = "text splitters (cómo trocear el documento)"
    esquema_json = (
        "{\n"
        '  "splitter_recomendado": "<nombre>",\n'
        '  "chunk_size_recomendado": <int, 0 si el splitter no usa tamaño>,\n'
        '  "chunk_overlap_recomendado": <int, 0 si no aplica>,\n'
        '  "confianza": "<alta|media|baja>",\n'
        '  "justificacion": "<2-3 frases basadas en los datos>",\n'
        '  "alertas": ["<aviso si hay algo preocupante>"]\n'
        "}"
    )
    # Primaria: recall (cuánto del pasaje se recupera; mide si el troceo parte mal
    # la respuesta). Secundarios: rapidez de troceo y homogeneidad de fragmentos.
    # n_chunks NO es objetivo (penalizaria trocear fino per se) pero SI contexto:
    # es el coste real (embeddings + tamaño de indice) y el juez debe verlo.
    metrica_primaria = ("recuperacion_pasaje", "max")
    # Desempate entre empatados estadisticos: CALIDAD primero (mrr = la metrica
    # de coronacion de la Fase B; relevancia), coste despues. Con latencia
    # primero (orden viejo), milisegundos de troceo coronaban configs peores en
    # TODAS las metricas de calidad (visto en Annie: recursive_250 sobre
    # token_500 con -0.035 recall y -0.06 mrr).
    objetivos_secundarios = [("mrr", "max"), ("relevancia_media", "max"),
                             ("latency", "min"), ("cv_variance", "min")]
    metricas_contexto = ("relevancia_media", "mrr", "n_chunks", "len_medio_chunk", "k_usado")

    def _tendencias(self, resultados: list[dict],
                    ganador: dict | None = None) -> list[str]:
        from .base import _corr_rangos
        lineas = []
        con_metrica = [(r["n_chunks"], r[self.metrica_primaria[0]])
                       for r in resultados
                       if r.get("n_chunks") and r.get(self.metrica_primaria[0]) is not None]
        if len(con_metrica) >= 3:
            rho = _corr_rangos([n for n, _ in con_metrica],
                               [v for _, v in con_metrica])
            if rho is not None:
                sentido = ("mas chunks (troceo mas fino) => MEJOR" if rho > 0.35
                           else "mas chunks => PEOR" if rho < -0.35
                           else "sin relacion clara tamaño-metrica")
                lineas.append(f"n_chunks vs {self.metrica_primaria[0]}: "
                              f"Spearman={rho:+.2f} ({sentido})")
        # el chequeo de borde sigue al CORONADO (si el desempate volteo al
        # lider bruto, es el coronado quien va a la final y define el optimo)
        lider = ganador or max((r for r in resultados
                                if r.get(self.metrica_primaria[0]) is not None),
                               key=lambda r: r[self.metrica_primaria[0]],
                               default=None)
        if lider:
            cfg = lider.get("_config") or {}
            fam, cs = cfg.get("splitter"), cfg.get("chunk_size")
            tams = sorted({(r.get("_config") or {}).get("chunk_size")
                           for r in resultados
                           if (r.get("_config") or {}).get("splitter") == fam
                           and (r.get("_config") or {}).get("chunk_size")})
            if cs and len(tams) >= 2 and cs == tams[0]:
                lineas.append(f"BORDE: el lider ({cfg.get('etiqueta')}) es el "
                              f"tamaño MINIMO probado de su familia: el optimo "
                              f"puede estar por debajo (parrilla corta por abajo)")
            elif cs and len(tams) >= 2 and cs == tams[-1]:
                lineas.append(f"BORDE: el lider ({cfg.get('etiqueta')}) es el "
                              f"tamaño MAXIMO probado: el optimo puede estar "
                              f"por encima")
        return lineas

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        experimentos = []
        for nombre in FAMILIAS_CON_TAMANO:
            for size in TAMANOS_FRAGMENTO:
                experimentos.append({
                    "etiqueta": f"{nombre}_{size}",
                    "splitter": nombre,
                    "chunk_size": size,                    # SIEMPRE caracteres
                    "chunk_overlap": int(size * 0.1),
                })
        experimentos.append({"etiqueta": "page", "splitter": "page",
                             "chunk_size": 0, "chunk_overlap": 0})
        if _semantic_disponible():
            experimentos.append({"etiqueta": "semantic", "splitter": "semantic",
                                 "chunk_size": 0, "chunk_overlap": 0})
        else:
            print("[splitter] AVISO: 'semantic' no disponible (falta "
                  "langchain-experimental); se omite de la parrilla.")
        experimentos = _dedupe_por_huella(experimentos, sesion.documentos)
        con_tam = sum(1 for e in experimentos if e["splitter"] in FAMILIAS_CON_TAMANO)
        print(f"[splitter] Parrilla: {len(experimentos)} configs "
              f"({con_tam} con tamaño en CARACTERES + "
              f"{', '.join(e['etiqueta'] for e in experimentos if e['splitter'] not in FAMILIAS_CON_TAMANO) or 'ninguna especial'}) "
              f"| presupuesto de recuperación: {PRESUPUESTO_CARACTERES}c por consulta")
        return experimentos

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        nombre = experimento["splitter"]
        size = experimento["chunk_size"]
        overlap = experimento["chunk_overlap"]
        etiqueta = experimento["etiqueta"]

        # EMBEDDING BASE UNIFICADO (decision 2026-07-16, cierra la asimetria
        # §9 del expediente del pipeline base): cascada coronado->exploratorio,
        # igual que retrievers/metadata/rerankers. Antes: CONFIG.embedding_neutro
        # (text-embedding-3-small, de pago). Razones: (a) el base debe estar
        # hecho de ELEMENTOS TESTEADOS de la parrilla (granite-311m-r2 lo es y
        # no comparte familia con arbitro ni rerankers); (b) "mide sobre lo que
        # despliegas"; (c) la interaccion troceo x embedding existe: medir
        # troceos sobre un modelo que nunca estara en el stack final arriesga
        # coronar el troceo optimo del modelo equivocado; (d) coste: este banco
        # re-embebe el corpus POR CADA troceo. ERA MENOR: los absolutos
        # historicos del banco no son comparables; los duelos internos si
        # (ceteris paribus dentro de cada corrida).
        from .embeddings_banco import embedding_base_exploratorio
        config_emb = sesion.config_embedding or embedding_base_exploratorio()
        modelo = construir_embedding(config_emb)

        # Troceo cacheado POR DOCUMENTO junto con su latencia real: los runs solo
        # remuestrean preguntas, el troceado no cambia. Se paga una vez (clave
        # para 'semantic', que embebe frases al trocear) y la métrica `latency`
        # es idéntica en todos los runs: sin sesgo por orden de ejecución.
        def _trocear():
            t0 = time.perf_counter()
            splitter = _construir_splitter_banco(nombre, size, overlap, modelo)
            chunks, vacios = _sin_vacios(splitter.split_documents(sesion.documentos))
            if vacios:
                print(f"[splitter] {etiqueta}: {vacios} chunk(s) vacios "
                      f"descartados (pagarian embedding sin poder recuperarse).")
            return chunks, round(time.perf_counter() - t0, 4)

        ya_estaba = etiqueta in _cache_troceo_claves(sesion.run_id)
        chunks, latencia = por_documento("splitter_troceo", sesion.run_id,
                                         etiqueta, _trocear)

        if not chunks:
            # Sin fragmentos no hay nada que medir: cv_variance=0 aquí sería el
            # MEJOR valor posible del secundario 'min' y este fracaso ganaría
            # desempates. Centinela = peor dispersión (ver metricas.CV_SIN_SENAL).
            print(f"[splitter] {etiqueta}: 0 chunks -> nada que medir (centinela).")
            return {"relevancia_media": 0.0, "mrr": 0.0, "recuperacion_pasaje": 0.0,
                    "cv_variance": CV_SIN_SENAL, "latency": latencia,
                    "n_chunks": 0, "len_medio_chunk": 0, "k_usado": 0}

        longitudes = [len(c.page_content) for c in chunks]
        len_medio = round(sum(longitudes) / len(longitudes))
        k = _k_por_presupuesto(longitudes)
        k_pedido = max(2, round(PRESUPUESTO_CARACTERES / max(1, len_medio)))
        if k < min(k_pedido, len(chunks)):
            print(f"[splitter] {etiqueta}: k CAPADO a {k} (el presupuesto "
                  f"pedia ~{k_pedido}): compite con menos presupuesto efectivo.")
        detalle_tok = (f" (~{_tokens_aprox(size)} tokens)" if nombre == "token" else "")
        print(f"[splitter] {etiqueta}{detalle_tok}: {len(chunks)} chunks | "
              f"len medio={len_medio}c (min={min(longitudes)}, max={max(longitudes)}) | "
              f"k={k} (~{k * len_medio}c de {PRESUPUESTO_CARACTERES}c de presupuesto) | "
              f"troceo {latencia}s{' [reutilizado]' if ya_estaba else ''}")

        huella = huella_troceado(chunks)
        indice_nuevo = huella not in _cache_indices_claves(sesion.run_id)
        t0 = time.perf_counter()
        vectorstore = por_documento("splitter", sesion.run_id, huella,
                                    lambda: FAISS.from_documents(chunks, modelo))
        if indice_nuevo:
            print(f"[splitter]   índice FAISS construido: {len(chunks)} embeddings "
                  f"en {time.perf_counter() - t0:.1f}s (coste real del troceo fino)")
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        metricas = evaluar_sobre_preguntas(
            lambda consulta: retriever.invoke(consulta),
            sesion.preguntas_test,
            modelo,
        )
        metricas["cv_variance"] = round(coeficiente_variacion(
            [c.page_content for c in chunks]), 4)
        metricas["latency"] = latencia
        metricas["n_chunks"] = len(chunks)
        metricas["len_medio_chunk"] = len_medio
        metricas["k_usado"] = k
        print(f"[splitter]   {etiqueta}: recall={metricas.get('recuperacion_pasaje')} | "
              f"mrr={metricas.get('mrr')} | relevancia={metricas.get('relevancia_media')} "
              f"| cv_len={metricas['cv_variance']}")
        return metricas


def _cache_troceo_claves(run_id: str) -> set:
    """Claves ya cacheadas del troceo (solo para poder loguear 'reutilizado')."""
    from .cache_documento import _CACHES
    c = _CACHES.get("splitter_troceo")
    return set(c["valores"]) if c and c.get("run_id") == run_id else set()


def _cache_indices_claves(run_id: str) -> set:
    from .cache_documento import _CACHES
    c = _CACHES.get("splitter")
    return set(c["valores"]) if c and c.get("run_id") == run_id else set()
