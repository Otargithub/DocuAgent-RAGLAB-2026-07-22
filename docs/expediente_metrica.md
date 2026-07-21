# Expediente de la métrica de relevancia — rag_lab

*Documento de traspaso para un hilo dedicado. Estado a 2026-07-16. Todo lo
afirmado aquí está verificado en código, en los 419+10 juicios humanos
persistidos, o en experimentos corridos; las conjeturas van marcadas como
tales. Referencia completa: `informe_anclaje_metrica.md` (§1–§11) y los
transcripts en `/mnt/transcripts/`. El dueño del proyecto es Xavier; espera
precisión matemática, fe de erratas explícita, y español sin regionalismos.*

---

## 1 · Definición exacta (metricas.py)

**Nota por fragmento** (contra la CITA del examen, nunca contra la pregunta):

    nota = α·lex + (1−α)·sem        α = 0.5 (RAG_PESO_EXACTITUD)

- **lex** = cobertura asimétrica de 5-gramas: fracción de las ventanas
  deslizantes de 5 tokens de la *cita* presentes en el fragmento
  (`|∩|/|gramas_cita|`). Tokenización: casefold, sin puntuación, des-guionado
  de PDF. n efectivo reducido si la cita tiene <5 tokens (evita el falso
  negativo estructural). Normalizar por la cita = invarianza al tamaño de
  chunk (la equidad de envases la da el k-por-presupuesto de 4000c). La
  comparación es intersección de CONJUNTOS DE TUPLAS de 5 tokens (orden y
  contigüidad codificados dentro de cada tupla): vocabulario suelto no puntúa.
- **sem** = max(0, (cos − b)/(1 − b)), coseno con el **árbitro fijo**
  e5-small (L2-normalizado, caché por texto), suelo **b = 0.7689** calibrado
  empíricamente (mediana de 66 pares de sondas no relacionadas; el suelo es
  POR árbitro y se recacha al cambiarlo). Sin el suelo, la anisotropía hace
  al sem casi constante (no discrimina e infla el recall).
- **Umbral U = 0.5** (RAG_UMBRAL_RELEVANCIA), estricto: define acierto =
  "localiza el pasaje". DECIDIDO y RATIFICADO dos veces (ver §5).

**Por pregunta** (`evaluar_recuperacion`, sobre los k fragmentos en orden):

- `relevancia` = max(notas) — la calidad del mejor fragmento.
- `mrr` = 1/posición del primer fragmento con nota ≥ U; 0 si ninguno.
- `recuperacion_pasaje` (antes `recall`/`recall_at_k`, renombrado 2026-07-16)
  = α·cobertura(UNIÓN de gramas de todos los fragmentos) + (1−α)·max(sem).
  Asimetría intencionada: reensamblado literal vs mejor equivalente
  individual. NO depende de ningún k ni es el recall clásico de IR.

**Agregación**: medias sobre el examen (28–40 preguntas) + series por
pregunta persistidas → bootstrap percentil 95% (semillas textuales SHA-512),
comparación pareada (cancela dificultad), mid-p en empates, centinela de
varianza cero (determinismo ≠ certeza).

Guardas: sims paralelas o ValueError; sin sims → degradado honesto a
solo-léxico (α=1); troceo vacío → CV_SIN_SENAL.

## 2 · El anclaje humano (la fuente de verdad)

**Campaña por-fragmento (CERRADA)**: 419 pares (pregunta, fragmento) juzgados
por Xavier con escala 0/1/2, en 5 documentos de perfiles distintos, con
lex/sem/qi/fi persistidos por par → cualquier forma de la nota es
re-evaluable offline en segundos. Semilla 20260705; identidad de fragmentos
por hash md5 (regeneración verificada 100/100).

**Tabla maestra (Spearman y AUC por componente, criterio humano ≥ 1):**

    documento     n  | Sp comb  Sp lex  Sp sem | AUC comb  AUC lex  AUC sem
    Annie       100  |  0.138    0.414   0.136 |   0.603    0.652    0.601
    test_13      99  |  0.657    0.538   0.610 |   0.851    0.681    0.837
    test_14     100  |  0.496    0.849   0.496 |   0.907    0.857    0.907
    test_15      70  |  0.332    0.606   0.326 |   0.656    0.660    0.654
    junger       50  |  0.727    0.767   0.731 |   0.970    0.886    0.972
    MEDIA            |  0.470    0.635   0.460 |   0.798    0.747    0.794

**Sesgo del MRR (máquina − humano)**: −0.05, −0.08, 0.00, −0.07, −0.05 —
no-positivo 5/5. El instrumento es CONSERVADOR: cada MRR es cota inferior
del juicio de Xavier. Se descuenta parafrásticos bajo el corte; casi nunca
acredita falsos.

**Campaña de conjunto (EN CURSO, 1/5)**: nivel de PREGUNTA para
`recuperacion_pasaje` ("¿el pasaje es recuperable del conjunto: entero o
reensamblable? 0/1/2"), colección `anclaje_recuperacion` con
coberturas/sems por fragmento persistidos. **Annie: n=10, Spearman
(humano_conjunto, recuperacion_maquina) = +0.779** — mejor de lo predicho
(0.3–0.5); la agregación de conjunto esquiva las patologías de ranking
(gemelos no restan gramas a la unión; max(sem) ignora el ruido). Pendientes:
autopsia de discrepancias de Annie + test_13, test_14, test_15, Jünger.

## 3 · Lecciones metodológicas (cómo se mide la vara)

- **El espejismo del Spearman**: los empates masivos de lex=0 coincidiendo
  con humano=0 inflan la correlación de rangos (α=1.0 parecía arrasar).
  Resuelto con **AUC** (Mann-Whitney, medio punto por empate): inmune al
  orden interno de bloques y al umbral. Regla: para comparar FORMAS de la
  nota, AUC antes que Spearman.
- **AUC pooled vs agrupado**: el auc() actual compara pares de preguntas
  DISTINTAS (504 parejas en Jünger). Es la vara correcta para la coherencia
  GLOBAL de la escala (U es global); el NDCG es la vista por-pregunta.
  Ofrecida y pendiente: variante AUC-agrupado como verificación de
  descalibración entre preguntas.
- **Triangulación de α a tres varas** (corrida): AUC 0.798 (gana 0.5), NDCG
  graduado con 0/1/2 y descuento posicional 0.903 (gana 0.75 por UNA
  milésima; meseta plana 0.25–0.75), fidelidad de MRR estilo Weaviate
  (sesgo −0.050, Sp 0.501) **CONSTANTE en los cinco α** — invarianza exacta.
  Único veto firme: α=1.0 (AUC 0.747).
- **§9, experimento negativo de la forma**: toda transformación MONÓTONA de
  la nota deja el AUC invariante → recalibraciones no lineales descartadas;
  el techo ~0.80 es de la INFORMACIÓN lex+sem, no de su escala.
- **Niveles de correlación**: nunca agrupar intra- y entre-documentos
  (paradoja de Simpson). Spearman por documento, consistencia de signo.

## 4 · Anatomía de la distribución (α=0.5) — las islas

Por construcción, tres poblaciones:
- **Literales** (contienen el pasaje): lex~0.9, sem~0.9 → nota 0.9–1.0.
- **Parafrásticos** (responden reformulando): lex=0.0 EXACTO (ninguna tupla
  de 5 sobrevive a la paráfrasis) → nota = 0.5·sem ≈ **0.15–0.30** (isla
  comprimida: la anisotropía apelotona los cosenos crudos ~0.82–0.90; la
  afín estira uniforme, no descomprime).
- **Irrelevantes**: nota 0–0.05.
- **Intermedios** (lex parcial): citas partidas por frontera de chunk; cada
  frontera destruye n−1=4 ventanas para siempre (ni la unión las recupera).

Entre parafrásticos y literales: el **desierto** (~0.35–0.85 según doc).
Consecuencias: mesetas del F1 al mover U por el desierto, acantilados al
cruzar una isla. U=0.5 vive en el desierto, por encima de la isla
parafrástica: precisión 1.00, recall parafrástico sacrificado.

**Los tres contratos con U** (histogramas por documento): test_14 (léxico):
margen enorme, rebaja inútil, precipicio en ~0.30 (86 irrelevantes vs ~4
relevantes cohabitando). test_13 (variado): margen medio, rebaja cara (isla
parafrástica cohabitada con irrelevantes; el tramo 0.5→0.30 rescata solo la
cornisa). Jünger (parafrástico): margen estrecho, rebaja casi GRATIS
(escalonados sin cohabitar: U=0.30 → precisión 1.00, recall 0.64→0.71;
primer peaje en 0.25: 0.92/0.79). **El precio de mover U lo fija el
SOLAPAMIENTO de cada documento, no el ancho de su desierto.**

**El cruce en U=0.5**: las cinco curvas F1(U) por α convergen (~0.51–0.53) —
con el corte estricto todas las mezclas acreditan el mismo conjunto (los
literales). NO es neutralidad: es que α no trabaja en el corte. α trabaja en
la ORDENACIÓN (AUC/MRR); U implementa la definición. Las perillas no se
pisan (confirmado cuantitativamente por la invarianza de la vara de MRR en
la triangulación).

## 5 · Decisiones de gobernanza (cerradas)

- **U = 0.5 estricto**: RATIFICADO dos veces con el mapa completo delante
  (F1 óptimo global estaría en ~0.25–0.30). Motivo: definición ("localiza el
  pasaje"), precisión 1.00, coste de era. Conservadurismo CON INVENTARIO: se
  sabe qué recall se deja (isla parafrástica de test_13/Jünger) y a cambio
  de qué. La definición amplia sigue re-evaluable offline en segundos.
- **α = 0.5**: validado por AUC, triangulado a tres varas. El seguro de la
  mezcla: nunca la peor forma, ≤0.05 del máximo en todos los perfiles; en
  test_13 SUPERA a ambos componentes (complementariedad real).
- **Valor de opción del sem**: con α=1.0, parafrásticos e irrelevantes
  colapsan en 0.0 exacto → la definición amplia DEJA DE EXISTIR como opción
  (ningún U los separa). El sem mantiene viva la reversibilidad; el corte lo
  ignora en un subconjunto, el laboratorio (ordenación, separación de
  configs, lex parcial, opción futura) no.
- **N_GRAMA = 5**: exigente a propósito (bimodalidad y precisión del lex);
  n menor = barrible offline contra los 419 si la isla parafrástica vuelve
  a la mesa (expectativa templada por §9).
- **Renombrados por honestidad**: precision→relevancia;
  recall/recall_at_k→recuperacion_pasaje (2026-07-16; memoria migrada:
  claves y campos operativos `_primaria/nombre` sí, prosa histórica NO).
  Protocolo: los nombres de métrica viven en 3 sitios — claves, valores
  operativos, prosa — con política migrar/migrar/conservar.

## 6 · Comportamiento por situación (el mapa operativo)

- **Documento léxico-técnico (test_14)**: lex manda (AUC 0.857), U casi
  irrelevante (meseta 0.30–0.95), sem redundante pero inofensivo.
- **Documento parafrástico con buen espacio (Jünger)**: sem redimido (AUC
  0.972, su única victoria), U estricto deja recall pero el MRR lo recupera
  por ordenación; disputa-de-bancos ≠ calidad-de-nota.
- **Documento hostil (Annie)**: AUC 0.603 = techo de la información; sem
  casi INERTE (Sp 0.136) — mejor argumento para un futuro α-por-perfil
  (dormido: con 5 docs anclados sería sobreajuste). Distractores: entradas
  gemelas del diario (misma estructura, fechas distintas) y respuestas
  cortadas por frontera. Metadata genérico NO ayuda (confirmado 3 veces).
- **Inmobiliario (test_15)**: AUC 0.656; fichas repetitivas como
  distractores. Con Annie, el mapa de dónde una TERCERA SEÑAL pagaría.
- **Cita partida por frontera** (ejemplo 30 tokens, corte al medio):
  lex mitades ≈ 0.385/0.462 → mrr probablemente 0; unión ≈ 0.846 (pierde
  solo la banda del corte). recuperacion_pasaje TOLERA el corte reparable
  (a propósito: distingue pecado venial de mortal); el mrr lo castiga; el
  desempate del banco y la Fase B (que corona por MRR) ponen el contrapeso.
  El patrón FINO-GANA-BANCO/grueso-gana-final es consecuencia estructural
  (4 confirmaciones): institucionalizado vía finalista-extra por MRR y
  prior de splitter calificado contra la coronación.
- **Experimento overlap 50→150 pendiente**: el solape haría existir las
  ventanas del corte en ambos fragmentos (ataca la banda perdida).

## 7 · Huecos conocidos y conjeturas abiertas

- **HUECO (encontrado 2026-07-16)**: pasaje partido Y parafrástico a la vez
  es invisible para las dos ramas de recuperacion_pasaje (la unión es solo
  léxica; el sem es del mejor fragmento INDIVIDUAL — no existe "embedding
  de la unión"). Población pequeña pero real.
- **CONJETURA de Xavier (abierta, con diagnóstico diseñado)**: "embeddings y
  retrievers sufren más con cortes finos al puntuar fragmentos aislados
  (descontextualización)". Compite con el efecto PUREZA (fino = señal
  concentrada) y con el sospechoso dominante: el fallo de ACREDITACIÓN
  (material recuperado que no cruza U). Firma que los distingue: preguntas
  con recuperacion_pasaje alto y mrr=0 = fallo de acreditación, no de
  recuperación. Celda pendiente (verificar si las series de ambas métricas
  por pregunta se persisten en experimentos).
- **Recuperacion_pasaje sin anclar del todo**: 1/5 documentos (+0.779).
- **Espectro idioma**: sondas del suelo presumiblemente españolas; primer
  documento inglés en curso.
- **La vara del futuro (I+D, precedente SMILE arXiv 2511.17432)**: (a)
  AUC(2-vs-1) entre relevantes — usa los grados que el binario tira;
  (b) doble umbral (U_parcial, U_pleno) por kappa ponderado + MRR con
  crédito graduado — cambio de era; estimación offline barata primero;
  (c) paráfrasis sintética de la cita (preprocesado congelado, un LLM la
  genera UNA vez como las preguntas; la métrica sigue sin LLM en
  evaluación): lex contra max{cita, paráfrasis} → el léxico vería la isla
  parafrástica. Prototipo propuesto en test_13. (d) n-grama-por-embedding
  (SMILE): candidato a tercera señal; exigiría su propio anclaje y
  guardarraíl de árbitro.

## 8 · Principios innegociables del expediente

1. La métrica NUNCA pasa por LLM en evaluación (los LLM generan exámenes
   auditados y narran; el árbitro e5 es el único que puntúa).
2. El árbitro es fijo, externo a la parrilla, con guardarraíl anti-nepotismo
   (`relacion_con_arbitro`) y suelo calibrado por árbitro.
3. Todo cambio de definición (U, α, forma, N_GRAMA) = CAMBIO DE ERA:
   backups, repoblación, jamás mezclar números inconmensurables.
4. Las decisiones se toman con inventario (qué se pierde, a cambio de qué)
   y quedan re-evaluables offline (los 419+ pares son el activo).
5. Predicciones registradas ANTES de los resultados; fe de erratas explícita
   cuando fallan (la de la campaña de conjunto falló por lo bajo: +0.779 vs
   0.3–0.5 predicho — la agregación de conjunto es más robusta que el
   ranking a las patologías de Annie).
