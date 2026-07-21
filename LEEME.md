# rag_lab — laboratorio modular de evaluación RAG

Paquete `rag_lab`: un meta-supervisor (LangGraph) dirige cinco bancos
(splitters, embeddings, metadata, retrievers, rerankers), aprende en una memoria
unificada y se apoya en dos análisis (LDA de temas, UMAP del espacio).

## Estructura
```
rag_lab_proyecto/
  rag_lab/
    __init__.py            config, memory, corpus, fabricas, metricas,
    config.py              evaluacion, estadistica, muestreo, vecinos,
    corpus.py              recuperacion
    memory.py
    fabricas.py
    metricas.py
    evaluacion.py
    estadistica.py
    muestreo.py
    vecinos.py
    recuperacion.py
    bancos/      base.py + splitters/embeddings_banco/metadata/retrievers/rerankers
    analisis/    lda.py, umap_proyeccion.py
    supervisor/  meta.py (MetaSupervisor, Plan)
  requirements.txt
  *.ipynb (instalacion_colab, explorar, entrenar, analizar)
```

## Arranque en Colab (ORDEN IMPORTANTE)
`CONFIG` es una instantánea del entorno EN EL MOMENTO DEL IMPORT. Si importas
`rag_lab` antes de fijar las variables de entorno, las claves saldrán vacías.

```python
# 1) Montar Drive
from google.colab import drive; drive.mount('/content/drive')

# 2) Fijar variables de entorno ANTES de importar rag_lab
import os
os.environ['OPENAI_API_KEY']    = '...'
os.environ['ANTHROPIC_API_KEY'] = '...'
os.environ['RAG_MEMORY_DIR']    = '/content/drive/MyDrive/rag_lab/rag_memory'

# 3) Arreglar sys.path (cwd en Colab es /content, no la raíz del proyecto)
import sys, pathlib
RAIZ = pathlib.Path('/content/drive/MyDrive/rag_lab/rag_lab_proyecto')
sys.path.insert(0, str(RAIZ))

# 4) Ahora sí, importar
from rag_lab import MetaSupervisor, Memoria
meta = MetaSupervisor()
stack = meta.ejecutar('/content/drive/MyDrive/.../documento.pdf')
```

## Deduplicación (no reentrenar lo ya estudiado)
`meta.ejecutar(pdf)` omite un documento si YA fue estudiado y completado antes
(misma huella de contenido SHA-256 y con decisión registrada). Devuelve el stack
previo con `_ya_estudiado=True`. Para reprocesar a propósito: `meta.ejecutar(pdf,
forzar=True)`. Una corrida a medias (sin decisión, p. ej. corte de Colab) NO
bloquea el reproceso: el batch es reanudable.
