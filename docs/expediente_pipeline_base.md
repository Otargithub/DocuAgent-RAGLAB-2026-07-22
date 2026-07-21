# Expediente: el pipeline BASE de contraste (Fase A, ceteris paribus)

*Documenta qué queda FIJO cuando cada banco varía su componente, con las
características técnicas exactas de cada elemento y su línea de código.
Verificado el 2026-07-16 sobre el zip vigente. Compañero del
`expediente_metrica.md` y de `conjeturas_abiertas.md`.*

**El principio** (plan.py:141-158, docstring literal): componentes "de
comportamiento conocido sobre los que se varía UN banco cada vez SIN heredar
del anterior. Así el ruido de un banco no contamina la medición de otro
(mata la propagación de error)". Importante: los bancos de la Fase A NO se
encadenan (el ganador de splitters no alimenta al banco de embeddings) — el
encadenado ocurre solo en la Fase B, que cruza finalistas.

---

## 1 · La definición canónica: `pipeline_base()` (plan.py:141-158)

| componente | valor base | detalle técnico |
|---|---|---|
| splitter  | `recursive`, chunk_size=1000, chunk_overlap=100 | RecursiveCharacterTextSplitter (LangChain); tamaños en CARACTERES (semántica única del laboratorio, ver §7). Overlap = 10% |
| embedding | `CONFIG.embedding_neutro` = text-embedding-3-small (OpenAI) | (config.py:79-81); sustituible por entorno |
| retriever | `similarity`, params={} | búsqueda vectorial pura sobre FAISS |

**Acta del baseline-fantasma** (comentario plan.py:150-154): el splitter
base está alineado con la etiqueta `recursive_1000` de la parrilla del banco
(overlap 10%) para que el baseline SEA una config medible del banco. Antes
(overlap 200) era una "config fantasma" que no aparecía en ninguna parrilla:
418 vs 449 chunks con el mismo nombre "recursive".

**Matiz de vigencia**: `pipeline_base()` es la referencia doctrinal, pero
cada banco materializa su base en su `preparar_contexto` con dos
desviaciones documentadas: (a) el embedding fijo REAL de los bancos
dependientes ya no es el neutro de OpenAI sino el exploratorio local o el
coronado (ver §3-6); (b) el k del retriever base varía por banco. La tabla
de la sección 8 es la foto exacta por banco.

## 2 · Constantes transversales del contraste

- **k-por-presupuesto**: PRESUPUESTO_CARACTERES = 4000 (splitters.py:63,
  "~el antiguo k=8 × chunks de 500"); k = max(2, 4000/len_medio), tope 40.
  Equidad de envases: cada consulta trae ~la misma cantidad de texto sea
  cual sea el troceo. SOLO el banco de splitters usa k dinámico (es quien
  varía el tamaño); los demás usan k fijos (ver §8).
- **El árbitro de la nota**: intfloat/multilingual-e5-small, huggingface
  (config.py:93-97, sobrescribible con RAG_EMBEDDING_JUEZ), con suelo
  calibrado b=0.7689 cacheado por árbitro. NO es parte del pipeline: es la
  vara. Guardarraíl `relacion_con_arbitro` (idéntico→excluido del banco de
  embeddings; familia→aviso). Antes de cambiarlo: sensibilidad_arbitro.py.
- **Examen congelado** por doc_hash: mismas preguntas para todos los bancos
  y runs (los runs remuestrean el examen con semilla `{doc_type}|run{n}`).
- **Cachés por documento** (`por_documento`, huella_troceado): el troceo,
  el índice FAISS y los embeddings de contenido se construyen UNA vez por
  documento (no por run ni por experimento) cuando los insumos no cambian —
  eficiencia sin tocar la medición.

## 3 · Banco de SPLITTERS — varía el troceo

- **Fijo — embedding**: `CONFIG.embedding_neutro` (splitters.py:270:
  `modelo = construir_embedding(CONFIG.embedding_neutro)`) =
  text-embedding-3-small. ÚNICO banco que aún usa el neutro de OpenAI como
  base (los dependientes migraron al exploratorio local; ver nota §9).
- **Fijo — retriever**: `similarity` puro vía
  `vectorstore.as_retriever(search_kwargs={"k": k})` (splitters.py:319),
  con **k dinámico por presupuesto** (recalculado por config de troceo).
- **Índice**: FAISS.from_documents por huella de troceo, cacheado por
  documento; la latencia del índice se reporta como "coste real del troceo
  fino" (splitters.py:316-318).
- **Varía**: la parrilla de troceos (recursive/token/character con tamaños,
  page, semantic si aplica), deduplicada por huella.

## 4 · Banco de EMBEDDINGS — varía el modelo de embedding

- **Fijo — splitter**: el heredado de la sesión (`sesion.config_splitter`,
  embeddings_banco.py:144); en corrida estándar = el splitter del pipeline
  base (recursive_1000). Se trocea UNA vez y todos los modelos reutilizan
  los mismos chunks.
- **Fijo — retriever**: `similarity` con **k=5 FIJO**
  (embeddings_banco.py:212) — no presupuesto: aquí el troceo no varía, y un
  k común hace comparables los modelos.
- **Varía**: el registro de embeddings (granite-311m-r2, qwen3-0.6b,
  bge-m3, bge-small-en, voyage, text-embedding-3-small/large... según
  credenciales), con el guardarraíl del árbitro y dedupe.

## 5 · Banco de RETRIEVERS — varía la estrategia de recuperación

- **Fijo — splitter**: heredado de la sesión (retrievers.py:80).
- **Fijo — embedding**: `sesion.config_embedding or
  embedding_base_exploratorio()` (retrievers.py:83-84) — es decir: el
  CORONADO del documento si el dirigido lo fijó; si no, el exploratorio
  local (§9).
- **Varía**: similarity k∈{3,5,10}; mmr (lambda_mult, fetch_k=K_CANDIDATOS
  =20, retrievers.py:31,42); bm25 k∈{5,10}; ensemble (w_bm25∈{0.3,0.5,0.7},
  k=10). Dedupe léxico: bajo bm25 puro el embedding no influye.

## 6 · Bancos de ETAPA OPCIONAL (baseline como concursante vivo)

### METADATA — varía el enriquecimiento
- **Fijo — splitter**: heredado; chunks CAPADOS a MAX_FRAGMENTOS=200
  (metadata.py:40,218) — coste acotado del doble embebido.
- **Fijo — embedding**: coronado si está fijado, si no exploratorio
  (metadata.py:228-230), con DECLARACIÓN en el log de cuál se usó
  ([GANADOR]/[coincide]/[⚠ NO coincide]/[exploratorio]) — la observación
  de Xavier hecha código (comentario metadata.py:221-227).
- **Fijo — retriever**: similarity con **K_EVAL=5** (metadata.py:39,300).
- **Varía**: alpha de mezcla contenido/metadatos × config de campos
  (baseline sin campos / positional / structural). Baseline a1.0 = hipótesis
  nula viva. Cachés: e_contenido una vez por documento; e_meta memoizada
  por campos; cada experimento = mezcla numpy + FAISS (ms).

### RERANKER — varía el reordenador
- **Fijo — splitter y embedding**: heredados/exploratorio
  (rerankers.py:160-164).
- **Fijo — retriever**: `sesion.config_retriever or similarity`
  (rerankers.py:174-176) — en el dirigido actual, el reranker fija el STACK
  CORONADO ENTERO (splitter+embedding+retriever) vía `_stack_ganador`.
- **Piscina**: K_CANDIDATOS=20 recuperados UNA vez para todos los
  rerankers; cada uno reordena y se evalúa su top K_FINAL=5
  (rerankers.py:44-45). El k coronado del retriever se ENSANCHA a 20 a
  propósito: el reranker necesita piscina que reordenar.
- **Varía**: no_reranker (baseline vivo, referencia del rerank_gain
  informativo), cross-encoders ms-marco MiniLM L-6/L-12, bge-reranker-base,
  de pago si hay API. La nota se puntúa con el ÁRBITRO fijo, no con el
  embedding heredado (anti-afinidad por construcción).

## 7 · Garantías de coherencia del contraste

- **Una sola semántica de troceo**: `_construir_splitter` DELEGA en el
  builder del banco de splitters (embeddings_banco.py:97-105, acta del bug:
  un finalista token_1000 se reconstruía como 1000 TOKENS ≈ 4000 caracteres
  — la Fase B medía un artefacto). Tamaños SIEMPRE en caracteres.
- **El baseline es medible**: el pipeline base coincide con una etiqueta
  real de la parrilla (recursive_1000), ver acta §1.
- **Sin herencia entre bancos en Fase A**: cada banco parte del base (o del
  coronado fijado explícitamente por el dirigido), nunca del ganador del
  banco anterior de la misma corrida.

## 8 · Tabla-resumen: qué fija cada banco

| banco | splitter fijo | embedding fijo | retriever fijo | k |
|---|---|---|---|---|
| splitter | (varía) | neutro OpenAI 3-small (l.270) | similarity | presupuesto 4000c |
| embedding | heredado (base: recursive_1000) | (varía) | similarity | 5 fijo (l.212) |
| retriever | heredado | coronado→exploratorio granite (l.83-84) | (varía) | según config |
| metadata | heredado, cap 200 chunks | coronado→exploratorio (l.228-230) | similarity | K_EVAL=5 |
| reranker | heredado / CORONADO en dirigido | coronado→exploratorio (l.163-164) | coronado→similarity (l.174) | 20 piscina → 5 eval |

## 9 · Asimetría conocida (candidata a decisión, no un bug)

El banco de SPLITTERS sigue usando como embedding fijo el **neutro de pago**
(text-embedding-3-small, splitters.py:270), mientras los tres bancos
dependientes prefieren coronado→exploratorio local (granite-311m-r2,
embeddings_banco.py:91-95: "medir sobre un modelo de pago que quizá no
despliegues sesga el veredicto y cuesta dinero"). Razones por las que puede
ser deliberado: el banco de splitters compara TROCEOS entre sí (el embedding
es constante compartida y su identidad importa menos que su estabilidad), y
cambiarlo altera los absolutos históricos del banco. Razones para
unificarlo: coste por corrida (embeber todo el corpus por cada troceo con
API de pago) y coherencia con la doctrina "mide sobre lo que despliegas".
PENDIENTE de decisión de Xavier; si se cambia, anotar que los absolutos del
banco de splitters no serán comparables con los previos (los duelos
internos, al ser ceteris paribus, sí conservan sentido).
