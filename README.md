# DocuAgent-RAG (rag_lab)

Laboratorio de *benchmarking* de pipelines RAG sobre documentos PDF: cinco
bancos de pruebas ceteris paribus (splitters, embeddings, metadatos,
retrievers y rerankers), un meta-supervisor LangGraph que sintetiza por
certeza estadística y corona en una fase factorial end-to-end, una métrica
híbrida de relevancia (léxica + semántica) **anclada a juicio humano**, y dos
agentes de análisis (analista e investigador con peaje HITL).

Principios: los LLM generan exámenes y narran, pero **nunca puntúan** (el
árbitro es un embedding fijo con suelo calibrado); todo azar lleva semilla;
los exámenes se congelan por hash; las decisiones se toman con estadística
pareada (bootstrap + mid-p) y quedan re-evaluables offline.

## Estructura

- `rag_lab/` — el paquete: métricas, evaluación, estadística, memoria,
  bancos (`bancos/`), supervisor y Fase B (`supervisor/`), experimento
  dirigido.
- `studio/graphs.py` — los dos agentes DocuAgent para LangGraph Studio.
- `tests/` — tests unitarios + dos *golden tests* que congelan byte a byte
  la síntesis y la coronación de Fase B (`tests/golden/`).
- Notebooks (`*.ipynb`) — entrenamiento, análisis, campañas de anclaje.
- `informe_anclaje_metrica.md` y `docs/` — la documentación técnica de la
  métrica, el pipeline base de contraste y el registro de conjeturas.
- `guia_langgraph_studio.md` — guía detallada de Studio.

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # o: pip install -r requirements.txt
cp .env.example .env             # y rellenar las claves (OpenAI, Anthropic)
```

Verificación (los cinco tests deben salir en verde; los golden no necesitan
red ni claves):

```bash
for t in tests/test_*.py; do echo "== $t"; python3 "$t" || break; done
```

## LangGraph Studio

```bash
pip install "langgraph-cli[inmem]"
langgraph dev                    # abre Studio sobre langgraph.json
```

Grafos disponibles en Studio: `docuagent` (analista, solo lectura) y
`docuagent_investigador` (graduado: puede lanzar experimentos dirigidos con
peaje HITL). El meta-supervisor completo se ejecuta desde los notebooks
(`entrenar.ipynb`, `campana_docuagent.ipynb`); ver la guía para el detalle
de pausas y reanudación.

## Datos

`pdfs/` y `rag_memory/` están en `.gitignore`: el corpus y la memoria de
resultados son datos locales, no código. Para reproducir desde cero basta
un PDF propio en `pdfs/` y una corrida del gimnasio.
