# Conjeturas abiertas — rag_lab

*Registro vivo de hipótesis pendientes de contraste. Regla de la casa: la
conjetura se enuncia ANTES de mirar los datos que la juzgarían; cada una
lleva su experimento discriminador y, cuando existe, su predicción
registrada. Estado a 2026-07-16. Compañero del `expediente_metrica.md`.*

**Disciplina del registro**: (1) predicciones antes que resultados; (2)
estratos y criterios definidos antes de cruzar con las métricas; (3) nunca
agrupar niveles intra-/entre-documentos (Simpson); (4) con n<5 documentos,
toda correlación entre-documentos es orientativa; (5) las conjeturas
refutadas no se borran — pasan a la sección de cerradas con su autopsia.

---

## A · Con experimento ya montado (solo falta correr/datos)

**A1 · rango_similitud ↔ mrr (intra-documento).** Conjetura implícita del
viejo desempate del metadata: "un espacio con rango de similitudes más
abierto recupera mejor". Estado: métrica demovida a contexto; persistencia
`_metricas` instalada; celda intra-doc lista. Falta: re-medición de metadata
sobre Annie con 4 alphas (a0.9 baseline/positional/structural +
a0.7_structural) para generar los primeros puntos. Criterio de readmisión:
rho positivo y CONSISTENTE entre documentos. Predicción (Claude, suave):
no correlacionará de forma consistente — quedará como métrica descriptiva.

**A2 · Acreditación vs descontextualización en el déficit del troceo fino
(Xavier, 2026-07-16).** Enunciado: "embeddings y retrievers sufren más con
cortes finos al puntuar cada fragmento aislado (descontextualización)".
Competidores: efecto PUREZA (fino = señal concentrada) y fallo de
ACREDITACIÓN (el material llega pero ninguna mitad cruza U). Firma
discriminadora: preguntas con recuperacion_pasaje ALTO y mrr = 0 → fallo de
acreditación (la vara, no el retriever); ambas bajas → descontextualización
gana peso. Falta: verificar si las series por pregunta de ambas métricas se
persisten en experimentos (si no, un dirigido corto de splitters las
regenera). Predicción (Claude): dominará la firma de acreditación en los
documentos con citas largas; la descontextualización aparecerá como efecto
menor en fino extremo (token_250) sobre documentos anafóricos (Annie).

**A3 · AUC-agrupado ≈ AUC-pooled (no hay descalibración entre preguntas).**
Si el AUC por pregunta (promediado) fuera MUY superior al pooled, la nota
ordenaría bien localmente pero sus niveles bailarían entre preguntas —
problema para el U global. Celda de 3 líneas ofrecida. Predicción (Claude):
saldrán parecidos (el NDCG 0.903 ya sugiere orden local sano).

**A4 · El resto de la campaña de conjunto rendirá Spearman ≥ Annie.** Annie
es el documento hostil y dio +0.779; los demás tienen mejor AUC por-pares.
Predicción registrada: los cuatro restantes darán ≥ +0.7, con Jünger el más
alto. Refutación interesante: un documento con buen AUC por-pares y mal
Spearman de conjunto revelaría un modo de fallo específico de la unión.

## B · Diseñadas, pendientes de arrancar

**B1 · w óptimo ↔ silueta (la pregunta madura del Investigador).** Patrón
acumulado: silueta 0.018→w0.7 (Annie), 0.039→w0.7 (Jünger), test_13
mixto→w0.3. PREDICCIÓN REGISTRADA para el doc nuevo en|denso (silueta
0.06): coronará ensemble w0.7 o bm25 — sería el 4º punto. Experimento
formal: tabla silueta-vs-w de todos los documentos coronados tras la
repoblación (nivel entre-documentos, orientativo hasta n≥5-6). Techo
actual: el agente no tiene tools transversales para investigarla él mismo.

**B2 · Especialización por facetas (bm25 léxico / ensemble parafrástico).**
Explicaría mecánicamente las disputas concluyentes-pero-inestables del
retriever. Diseño cerrado: anotar la faceta léxica de cada pregunta de los
exámenes congelados (cobertura pregunta-cita, computable sin LLM, criterio
fijado ANTES), duelo pareado POR ESTRATO, interacción = diferencia de
diferencias. Empezar por Jünger. Con ~20 preguntas por estrato solo
interacciones grandes saldrán — suficiente para el diagnóstico. Si
confirma: la respuesta correcta para esos documentos no es un ganador sino
un ROUTER por tipo de query (pieza de pipeline nueva).

**B3 · Reranker sobre stack coronado: gain pequeño pero real, latencia como
juez.** El cross-encoder lee query y candidato JUNTOS: en espacios-sopa
(Annie) tiene margen que el bi-encoder no. Experimento estrenable: dirigido
de reranker (ya fija el stack coronado ENTERO vía _stack_ganador), baseline
no_reranker vivo. Predicción (Claude, suave): rerank_gain positivo pequeño
en Annie; el veredicto final dependerá del coste (20 inferencias/consulta).

**B4 · Overlap 50→150 en token_500 rescata la banda del corte.** Con solape,
las n−1=4 ventanas de cada frontera EXISTEN en ambos fragmentos: deberían
subir a la vez el lex de las mitades, el mrr de las preguntas de cita
partida, y la cobertura de unión. Nacido del anclaje de Annie (falsos
negativos de frontera). Coste: más chunks (~+30%) → más embeddings.
Refutación posible: que el solape duplique gemelos y empeore el ranking.

**B5 · Nivel 2 de correlación con DocuAgent: los VEREDICTOS coinciden.**
Re-jugar los duelos del agente con el MRR humano de los 419 como métrica
(cruce por identidad de fragmento, cobertura como limitante). Predicción
(Claude): concordancia alta en los veredictos de empate (el conservadurismo
se hereda bien); el riesgo residual está en los positivos apretados.

## C · Importables de SMILE (prototipos propuestos, sin empezar)

**C1 · La paráfrasis sintética de la cita hace visible la isla al léxico.**
Generar UNA VEZ por pregunta del examen congelado una paráfrasis de la cita
(mismo estatuto que las preguntas: preprocesado auditable congelado; la
métrica sigue sin LLM en evaluación); lex contra max{cita, paráfrasis}.
Prototipo en test_13 (isla más poblada): re-puntuar los pares del anclaje y
medir cuánto sube la isla y qué hace el AUC. Prerrequisito verificado a
medias: los pares NO persisten el texto — hay que regenerar (identidad por
clave ya demostrada 100/100).

**C2 · El doble umbral graduado (U_parcial, U_pleno) mejora el acuerdo.**
Barrido offline del par por kappa ponderado contra los 0/1/2; MRR con
crédito graduado como consecuencia. Si convence: conversación de CAMBIO DE
ERA (redefine la primaria). La estimación es gratis; la adopción no.

**C3 · AUC(2-vs-1): ¿la nota distingue tus "claramente relevante" de tus
"relevante"?** Columna nueva para la tabla maestra, gratis con los 419.
Mide el grano fino que el binario tira.

## D · Dormidas (esperando masa crítica de datos)

**D1 · α por perfil.** Annie es el caso: sem casi inerte (Sp 0.136), el
léxico carga con todo — un α condicionado al perfil cargaría lex en
documentos tipo Annie. DORMIDA hasta ≥10 documentos anclados (con 5 sería
sobreajuste). El anclaje por documento ya da los datos; falta corpus.

**D2 · Tercera señal para los documentos-techo.** Annie e inmobiliario a
AUC ~0.6 con CUALQUIER α (límite de información, §9). Candidatas:
enriquecimiento a medida (fechas de entrada del diario — la saga metadata
demostró que lo genérico no ayuda), n-grama-por-embedding (SMILE, exigiría
su propio anclaje y guardarraíl de árbitro), señales estructurales.
Se activa cuando el corpus anclado crezca y el techo duela en producción.

**D3 · El suelo b por idioma.** Las sondas de calibración son
presumiblemente españolas; el primer documento inglés (en curso) es el
primer estrés. Si el anclaje de un doc inglés mostrara sem descalibrado,
tocaría sondas por idioma o suelo multilingüe.

**D4 · El hueco partido-Y-parafrástico es raro.** La única población
invisible para ambas ramas de recuperacion_pasaje (unión léxica + max sem
individual). Conjetura: población despreciable en corpus reales. Se
contrasta gratis con las campañas de conjunto: discrepancias donde tu
etiqueta sea 2 y la máquina puntúe bajo en AMBAS ramas.

## E · Cerradas (con autopsia, para no reabrirlas sin datos nuevos)

- **Metadata genérico ayuda en Annie**: REFUTADA 3 veces con rigor creciente
  (base contaminada → varianza cero → corrida limpia sobre granite: empate
  trivial). La estructura genérica no reconoce las entradas del diario.
- **El léxico puro es la mejor forma (vía Spearman)**: ESPEJISMO — empates
  masivos; el AUC lo desmintió (0.747 vs 0.798).
- **Una transformación de la forma de la nota puede subir el AUC**: FALSA
  para toda monótona (§9); el techo es de la información.
- **La campaña de conjunto daría Spearman 0.3–0.5 en Annie** (Claude):
  FALLIDA POR LO BAJO (+0.779) — la agregación de conjunto esquiva las
  patologías del ranking (gemelos no restan gramas; max ignora ruido).
  Fe de erratas registrada.

---

## Material congelado disponible para la fase de conjeturas

(Los activos que hacen baratos los contrastes — verificar la lista con
Xavier, que menciona "análisis congelados" acumulados:)
- 55 exámenes congelados por doc_hash (anotables con facetas sin regenerar).
- 419 pares de anclaje con lex/sem/qi/fi + campaña de conjunto en curso con
  coberturas/sems por fragmento.
- `_metricas` por config en cada veredicto nuevo (desde 2026-07-14).
- Series por pregunta persistidas en experimentos.
- Coronaciones de la era limpia (Annie, Jünger, doc nuevo en curso) con
  stacks completos.
- Scripts offline ya entregados: barridos α/U, tres varas, histogramas por
  documento, sesgo de MRR, tabla maestra, autopsia de conjunto.

**Plan acordado**: terminar el anclaje de recuperacion_pasaje (4 documentos
+ autopsia de Annie) → abrir la fase de conjeturas por la sección A (los
experimentos ya montados), con B1/B2 como platos fuertes.
