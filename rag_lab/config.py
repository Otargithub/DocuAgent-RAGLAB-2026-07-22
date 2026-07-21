"""
Configuración central del laboratorio RAG.

La idea de este módulo es que TODO lo que dependa del entorno (claves de API,
rutas a ficheros, parámetros globales) viva en un solo sitio. Así no repetimos
la carga de variables de entorno en cada banco, y no quedan rutas absolutas
incrustadas por el código.

Diseño: una única instancia `CONFIG` que el resto del paquete importa. Si
mañana cambias la ruta del PDF o la carpeta de memoria, lo cambias aquí y nada
más.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _cargar_variables_entorno() -> None:
    """
    Busca un fichero de variables de entorno y lo carga.

    Probamos, por orden, los nombres más habituales (.env y env.txt) subiendo
    por el árbol de directorios desde el directorio de trabajo actual. En cuanto
    encontramos uno, lo cargamos y paramos. Si no hay ninguno, no pasa nada:
    asumimos que las variables ya están en el entorno del sistema.
    """
    nombres_posibles = (".env", "env.txt")
    directorio = Path.cwd()

    # Subimos como mucho cinco niveles para no recorrer el disco entero.
    for _ in range(5):
        for nombre in nombres_posibles:
            candidato = directorio / nombre
            if candidato.exists():
                load_dotenv(dotenv_path=candidato, override=True)
                return
        directorio = directorio.parent


@dataclass
class Config:
    """Parámetros del laboratorio. Se rellenan desde variables de entorno."""

    # --- Claves de API -------------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""
    hf_token: str = ""

    # --- Rutas ---------------------------------------------------------------
    # El PDF a estudiar y la carpeta donde persiste la memoria. Ambos se pueden
    # sobreescribir por variable de entorno (PDF_PATH, RAG_MEMORY_DIR) o al
    # arrancar el programa, de modo que el código sea portable entre máquinas.
    pdf_path: str = ""
    memoria_dir: Path = field(default_factory=lambda: Path("rag_memory"))

    # --- Modelos por defecto -------------------------------------------------
    # El modelo "juez" es el que usan los supervisores y el meta-supervisor para
    # razonar sobre los resultados. El de clasificación es más barato porque solo
    # etiqueta el documento y genera preguntas.
    modelo_juez: str = "claude-opus-4-5"
    modelo_clasificador: str = "gpt-4o-mini"

    # --- Fiabilidad de la medición ------------------------------------------
    # Cuántas preguntas de test se generan por documento. No es una perilla de
    # experimento (no pertenece al plan del supervisor), sino la base estadística
    # sobre la que se mide cada banco: más preguntas = menos error por métrica.
    n_preguntas_test: int = 10

    # --- Embedding neutro ----------------------------------------------------
    # Cuando un banco necesita un embedding "de control" (por ejemplo, el banco
    # de splitters, que todavía no sabe qué embedding es el mejor), usa este.
    embedding_neutro: dict = field(
        default_factory=lambda: {"model": "text-embedding-3-small", "provider": "openai"}
    )

    # --- Árbitro semántico de la nota ---------------------------------------
    # Modelo FIJO que puntúa la parte semántica de la relevancia (ver metricas.py
    # y evaluacion.py). Debe ser INDEPENDIENTE del conjunto de modelos que se
    # evalúan en el banco de embeddings: si el árbitro (o un pariente cercano de
    # su misma familia) está en la parrilla, se autoevalúa y sesga la comparación.
    # Por eso vive aquí, para poder cambiarlo a algo ajeno a los candidatos.
    # Cambiable por entorno: RAG_EMBEDDING_JUEZ="org/modelo" (HuggingFace).
    # OJO: cambiar de árbitro es cambiar de ERA de métricas (el semántico se
    # puntúa con otro espacio): los MRR dejan de ser comparables con los
    # históricos. Antes de cambiarlo en serio, pasa sensibilidad_arbitro.py.
    embedding_juez: dict = field(
        default_factory=lambda: {
            "model": os.environ.get("RAG_EMBEDDING_JUEZ",
                                    "intfloat/multilingual-e5-small"),
            "provider": "huggingface"}
    )

    # --- Parámetros de la nota de recuperación (metricas.py) -----------------
    # Estas tres SÍ son perillas reales de la medición y ahora se leen de aquí
    # (antes estaban congeladas como constantes en metricas.py).
    #
    #   peso_exactitud (α ∈ [0,1]): cuánto pesa el solapamiento léxico frente a
    #       la similitud semántica. α=1 => nota solo-léxica.
    #   umbral_relevancia ∈ [0,1]: relevancia mínima para que un fragmento cuente
    #       como acierto en el MRR.
    #   base_semantica: SUELO del coseno del árbitro. None (por defecto) =
    #       CALIBRACIÓN EMPÍRICA automática por árbitro (ver
    #       evaluacion.obtener_base_semantica: mediana del coseno entre pares de
    #       textos no relacionados, calculada una vez por modelo árbitro y
    #       cacheada en el proceso). Un número en [0, 0.99) lo FIJA manualmente y
    #       desactiva la calibración; 0.0 reproduce el comportamiento antiguo
    #       (sin suelo). Reescala: sim' = clip((cos - base)/(1 - base), 0, 1).
    peso_exactitud: float = 0.5
    umbral_relevancia: float = 0.5
    base_semantica: float | None = None

    # --- Verificación de las preguntas de test (corpus.verificar_preguntas) --
    # Nivel de exigencia sobre el examen sintético que genera el LLM barato:
    #   "citas"    -> solo capa 1 (la cita existe en el documento; validar_citas).
    #   "barata"   -> + capa 3 (autocontención, regex) y capa 4 (discriminación
    #                 con el árbitro semántico). Sin coste de LLM.
    #   "completa" -> + capa 2 (round-trip con el LLM juez: ¿la cita responde a
    #                 la pregunta y la expected_answer es correcta?). Una llamada
    #                 al juez por tanda de preguntas; es la capa cara y la que
    #                 más confianza aporta (verificador CRUZADO: modelo distinto
    #                 del generador, errores no correlacionados).
    verificacion_preguntas: str = "completa"

    @property
    def fichero_memoria(self) -> Path:
        """Ruta completa al JSON de memoria unificada."""
        return self.memoria_dir / "memoria_unificada.json"

    def validar(self) -> list[str]:
        """
        Devuelve una lista de avisos sobre lo que falta para poder ejecutar.
        No lanza excepciones: simplemente informa, para que el usuario decida.
        """
        avisos = []
        if not self.openai_api_key:
            avisos.append("Falta OPENAI_API_KEY (necesaria para clasificar y para embeddings OpenAI).")
        if not self.anthropic_api_key:
            avisos.append("Falta ANTHROPIC_API_KEY (necesaria para los supervisores).")
        if not self.pdf_path:
            avisos.append("No se ha indicado PDF_PATH.")
        elif not Path(self.pdf_path).exists():
            avisos.append(f"El PDF indicado no existe: {self.pdf_path}")
        return avisos


def construir_config() -> Config:
    """Crea la configuración leyendo el entorno y prepara la carpeta de memoria."""
    _cargar_variables_entorno()

    cfg = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        hf_token=os.getenv("HF_TOKEN", ""),
        pdf_path=os.getenv("PDF_PATH", ""),
        memoria_dir=Path(os.getenv("RAG_MEMORY_DIR", "rag_memory")),
    )

    # Perillas de la nota, sobreescribibles por entorno (útil para barridos de α
    # sin tocar código). Si el valor no parsea, se conserva el de la dataclass.
    def _flotante_entorno(nombre: str, actual: float) -> float:
        crudo = os.getenv(nombre)
        if crudo is None:
            return actual
        try:
            return float(crudo)
        except ValueError:
            print(f"[config] {nombre}={crudo!r} no es un número; uso {actual}.")
            return actual

    cfg.peso_exactitud = min(1.0, max(0.0, _flotante_entorno(
        "RAG_PESO_EXACTITUD", cfg.peso_exactitud)))
    cfg.umbral_relevancia = min(1.0, max(0.0, _flotante_entorno(
        "RAG_UMBRAL_RELEVANCIA", cfg.umbral_relevancia)))

    # RAG_BASE_SEMANTICA: "auto" (o ausente) => None => calibración empírica por
    # árbitro; un número => suelo fijo manual, saturado a [0, 0.99).
    crudo_base = os.getenv("RAG_BASE_SEMANTICA")
    if crudo_base is not None and crudo_base.strip().casefold() != "auto":
        try:
            cfg.base_semantica = min(0.99, max(0.0, float(crudo_base)))
        except ValueError:
            print(f"[config] RAG_BASE_SEMANTICA={crudo_base!r} no es un número "
                  f"ni 'auto'; mantengo calibración automática.")
            cfg.base_semantica = None

    nivel = os.getenv("RAG_VERIFICACION_PREGUNTAS")
    if nivel is not None:
        nivel = nivel.strip().casefold()
        if nivel in ("citas", "barata", "completa"):
            cfg.verificacion_preguntas = nivel
        else:
            print(f"[config] RAG_VERIFICACION_PREGUNTAS={nivel!r} no es válido "
                  f"(citas|barata|completa); uso '{cfg.verificacion_preguntas}'.")

    # Exportamos las claves al entorno del proceso para que las librerías de
    # terceros (langchain_openai, etc.) las encuentren donde esperan encontrarlas.
    os.environ.setdefault("OPENAI_API_KEY", cfg.openai_api_key)
    os.environ.setdefault("ANTHROPIC_API_KEY", cfg.anthropic_api_key)
    os.environ.setdefault("VOYAGE_API_KEY", cfg.voyage_api_key)
    os.environ.setdefault("COHERE_API_KEY", cfg.cohere_api_key)
    os.environ.setdefault("HF_TOKEN", cfg.hf_token)

    cfg.memoria_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# Instancia única que importa el resto del paquete.
CONFIG = construir_config()
