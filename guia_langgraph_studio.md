# Guía práctica: LangGraph Studio para depurar tus grafos

Studio es la UI visual para **ver, interactuar y depurar** grafos LangGraph en
ejecución. Esta guía va al grano y está adaptada a `rag_lab`: los ficheros que
necesita (`langgraph.json`, `studio/graphs.py`, `.env.example`) ya están en la
raíz del proyecto.

## 0 · Qué es Studio HOY (y qué ya no es)

La app de escritorio de Mac está **muerta**. Hoy Studio es dos piezas:

1. **Un servidor local** que corre TU código: `langgraph dev` (in-memory, para
   desarrollo; Python ≥3.11).
2. **Una UI web** alojada en `smith.langchain.com/studio` que se conecta a ese
   servidor local vía `?baseUrl=http://127.0.0.1:2024`. Tu grafo NUNCA sale de
   tu máquina; la web solo lo visualiza.

Hace falta cuenta gratuita de LangSmith (la UI vive ahí).

## 1 · Instalación y arranque (5 minutos)

```bash
pip install -U "langgraph-cli[inmem]"        # el extra inmem es OBLIGATORIO
cp .env.example .env                          # y rellena las claves
cd rag_lab_proyecto                           # donde está langgraph.json
langgraph dev
```

Salida esperada:

```
- 🚀 API:        http://127.0.0.1:2024
- 🎨 Studio UI:  https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs:   http://127.0.0.1:2024/docs
```

Abre la URL de Studio en el navegador. **Safari y Brave bloquean localhost**:
usa `langgraph dev --tunnel` (crea un túnel HTTPS y te da otra baseUrl).

## 2 · Cómo se exporta un grafo (lo que ya está hecho)

`langgraph.json` mapea nombres a objetos compilados:

```json
{ "dependencies": ["."],
  "graphs": { "docuagent": "./studio/graphs.py:docuagent" },
  "env": ".env" }
```

`studio/graphs.py` construye DocuAgent (agente + tools del observatorio) y lo
compila **SIN checkpointer** — regla de oro: el servidor inyecta su propia
persistencia; un `MemorySaver` propio se ignora. Por eso no se reutiliza el
compile de los notebooks.

> **Por qué DocuAgent y no el meta**: las tools de DocuAgent son numpy sobre la
> memoria — cada "Run" en Studio cuesta céntimos y tarda segundos. El meta es
> exportable con el mismo patrón (una factory que devuelva `meta.grafo`), pero
> cada run son MINUTOS de experimentos reales con GPU y APIs: Studio invita a
> darle al botón alegremente, y con el meta eso es dinero. Empieza por
> DocuAgent; exporta el meta solo cuando quieras depurar sus pausas.

## 3 · La UI en 60 segundos

- **Izquierda**: el grafo dibujado (nodos y aristas — verás `agente ⇄ tools`).
- **Centro**: el input. Para DocuAgent, un mensaje: pincha en `messages`, rol
  `human`, escribe la pregunta y **Submit**.
- **Derecha/abajo**: la ejecución en directo — cada nodo se ilumina al correr y
  puedes desplegar su entrada/salida (los `tool_calls` con argumentos, el
  `ToolMessage` con la observación, el `AIMessage` final).
- **Threads** (panel lateral): cada conversación es un hilo con su historial de
  checkpoints. Un hilo = un `thread_id` = memoria multi-turno.

## 4 · Los seis flujos de depuración que importan

### 4.1 Ver el bucle ReAct visualmente
Lanza una pregunta que requiera tools ("¿hay diferencia concluyente entre
bge-m3 y granite-311m-r2?") y observa: `agente` → decide tool → `tools` →
observación → `agente` → respuesta. Es tu `stream_mode="updates"` pero con
dibujos: cada iluminación del nodo es un evento del stream.

### 4.2 Interrupts (HITL) desde la UI
Si el grafo tiene `interrupt()` (las 4 pausas de tu `meta`, o el nodo-peaje del
curso §8), Studio **se detiene visualmente** en el nodo, muestra el payload de
la pausa, y te deja escribir el JSON de reanudación (`{}`, `{"vetar": [...]}`)
en un formulario. Es la forma más cómoda que existe de probar tus pausas:
mucho más ágil que el bucle `Command(resume=...)` a mano.

### 4.3 Editar el estado y re-ejecutar desde un nodo
En cualquier checkpoint del hilo puedes **editar el estado** (p. ej. cambiar el
contenido de un `ToolMessage` para simular otra observación) y re-ejecutar
desde ese punto. Es tu `update_state` + fork del curso §7, con interfaz. Úsalo
para responder "¿qué habría dicho el agente si el duelo hubiera salido
concluyente?" sin tocar datos reales.

### 4.4 Hot reload: romper → arreglar → rerun from node
El servidor recarga tu código al guardar. El flujo estrella de depuración:
corre el grafo hasta que un nodo falle → edita el `.py` en tu editor → dale a
re-ejecutar **desde el nodo que falló** (no desde el principio). Con el meta
esto vale oro: no repites 40 minutos de Fase A por un bug en la síntesis (el
`NameError` del otro día se habría cazado y arreglado en dos minutos así).

### 4.5 Breakpoints de verdad (debugpy + VS Code)
Para inspección de variables paso a paso dentro de un nodo:

```bash
pip install debugpy
langgraph dev --debug-port 5678
```

Y en `launch.json` de VS Code:

```json
{ "name": "Attach to LangGraph", "type": "debugpy", "request": "attach",
  "connect": { "host": "0.0.0.0", "port": 5678 } }
```

Pones un breakpoint en `_agente` o en una tool, lanzas el run desde Studio, y
VS Code se para ahí con todas las variables a la vista.

### 4.6 Arqueología de hilos
En el panel de threads, cada checkpoint del historial es inspeccionable: qué
mensajes había, qué nodo venía después. Es `get_state_history` con interfaz —
útil para el post-mortem de "¿en qué punto exacto se torció este análisis?".

## 5 · El detalle Colab (importante para ti)

Studio necesita el servidor `langgraph dev` corriendo donde está el código, y
**Colab no es su hábitat**: el flujo cómodo es en tu portátil. La buena noticia
es que DocuAgent corre perfecto en local sin GPU (sus tools son numpy) — solo
necesitas la carpeta del proyecto y una copia de `rag_memory/` (bájate el JSON
de Drive). El meta sí necesita el entorno completo.

Alternativa **desde Colab** cuando quieras observabilidad sin Studio: tracing
de LangSmith puro. Dos variables antes de importar nada:

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = userdata.get("LANGSMITH_API_KEY")
```

Cada run del meta/bancos/DocuAgent aparece traceado en smith.langchain.com:
árbol de llamadas, prompts completos, tokens, latencias. No es interactivo como
Studio, pero para auditar qué pasó en una tanda nocturna es incluso mejor.

## 6 · Errores típicos

| Síntoma | Causa / arreglo |
|---|---|
| La UI no conecta (Safari/Brave) | `langgraph dev --tunnel` y usa la baseUrl del túnel |
| `ModuleNotFoundError: rag_lab` | falta `"dependencies": ["."]` en langgraph.json, o no lanzaste `langgraph dev` desde la raíz |
| El grafo no aparece | ruta mal en `graphs` (formato `./fichero.py:variable`) |
| Aviso sobre checkpointer | exportaste un grafo compilado con MemorySaver propio: compílalo sin él |
| `inmem` no soportado | `pip install -U "langgraph-cli[inmem]"` y Python ≥3.11 |
| Claves no cargadas | el `.env` debe estar donde dice `langgraph.json` (`"env": ".env"`) |
| Memoria vacía en DocuAgent | `RAG_MEMORY_DIR` del `.env` no apunta a tu `rag_memory/` local |

## 7 · Chuleta

```bash
langgraph dev                      # servidor + Studio
langgraph dev --tunnel             # túnel HTTPS (Safari, o acceder desde otra máquina)
langgraph dev --debug-port 5678    # breakpoints con debugpy/VS Code
langgraph dev --port 3000          # puerto alternativo (ajusta baseUrl)
```

Ruta de aprendizaje sugerida: exporta DocuAgent (hecho) → juega con 4.1 y 4.3
hasta dominar la UI → añade el grafo HITL del curso (§8) a `langgraph.json` y
practica 4.2 → cuando quieras depurar las pausas reales del meta, expórtalo con
una factory y usa 4.4 para no pagar Fase A completa por cada bug.
