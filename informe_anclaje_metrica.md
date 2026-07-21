# Anclaje humano de la métrica: informe técnico

*rag_lab · campaña de anclaje de julio 2026 · 5 documentos, 419 juicios humanos*

Este documento explica con detalle completo — matemáticas incluidas — qué es la
métrica del laboratorio, por qué necesitaba calibración, cómo se ancló al
juicio humano, qué se encontró, qué artefactos estadísticos hubo que cazar por
el camino, y qué decisiones quedaron tomadas y abiertas. Es la referencia de
la vara de medir del laboratorio: todo MRR, recall o veredicto que produzca el
sistema significa lo que este documento dice que significa.

---

## 1 · La métrica: qué puntúa exactamente el laboratorio

Cada pregunta del examen congelado lleva una **cita** (`source_context`): el
pasaje literal del documento donde vive la respuesta. Cuando un pipeline
recupera un fragmento `f` para una pregunta, su relevancia se puntúa contra la
cita `c` — nunca contra la pregunta — con una nota combinada:

```
nota(f, c) = α · lex(f, c) + (1 − α) · sem(f, c),        α = 0.5
```

**Componente léxico** — cobertura de n-gramas de la cita:

```
lex(f, c) = |G(f) ∩ G(c)| / |G(c)|
```

donde `G(·)` es el conjunto de n-gramas de palabras del texto, producido por el
tokenizador español del laboratorio (el mismo que usa BM25: minúsculas, tildes
normalizadas — una sola definición de "palabra" en toda la casa). Mide qué
fracción de la cita aparece *literalmente* en el fragmento. Propiedades: vale
1.0 si el fragmento contiene la cita entera; cae rápido con la paráfrasis
(decir lo mismo con otras palabras comparte pocos n-gramas); es exactamente 0
para textos sin secuencias comunes.

**Componente semántico** — similitud de coseno bajo el árbitro, recalibrada:

```
cos(f, c) = ⟨v(f), v(c)⟩            (vectores normalizados del árbitro)
sem(f, c) = max(0, (cos − b) / (1 − b))
```

El árbitro es un modelo de embeddings independiente de los concursantes
(`intfloat/multilingual-e5-small`), y `b` es su **suelo semántico calibrado**
(§2). La recalibración estira el tramo útil `[b, 1]` al `[0, 1]` completo.

**Umbral de relevancia**: un fragmento cuenta como *acierto* si
`nota ≥ U`, con `U = 0.5` vigente. Sobre esa definición binaria se construyen
las métricas de recuperación:

```
MRR      = media_q( 1 / rank_q )     rank_q = posición del primer acierto
                                      (0 si ninguno de los k acierta)
recall@k = media_q( cobertura de la cita lograda entre los k fragmentos )
precisión = media de la nota de los fragmentos recuperados
```

Con presupuesto de caracteres por consulta (k-por-presupuesto, 4000c) para que
splitters con chunks de tamaños distintos compitan en igualdad.

**Incertidumbre**: los intervalos de confianza son bootstrap percentil al 95%
sobre la serie por pregunta (remuestreo con reemplazo, semilla determinista), y
las comparaciones entre configs son **pareadas**: se remuestrea la serie de
diferencias `d_q = m_A(q) − m_B(q)` sobre las mismas preguntas, con

```
prob(A mejor) = P(d̄* > 0) + ½ · P(d̄* = 0)        (mid-p, empates a medias)
CONCLUYENTE   = el IC de d̄ no contiene el 0
```

---

## 2 · La calibración del árbitro: por qué existe el suelo `b`

Los espacios de embeddings multilingües son **anisótropos**: dos textos sin
relación alguna no puntúan coseno ≈ 0 sino un valor alto y estable. Para
e5-small en el corpus del laboratorio, la mediana de coseno entre pares de
sondas *deliberadamente no relacionadas* (12 sondas → 66 pares) es:

```
b = 0.7689
```

Sin recalibrar, "nada que ver" puntuaría 0.77 y "muy relacionado" 0.92 — un
rango útil de 0.15 aplastado en la parte alta de la escala, que además se
mezclaría con un componente léxico que sí usa todo el [0,1]. La recalibración
`(cos − b)/(1 − b)` resuelve eso: el suelo empírico pasa a ser el 0.

Detalles operativos: `b` se calcula **por árbitro** y se cachea
(`_BASES_CALIBRADAS`) — cambiar de árbitro recalibra automáticamente; los
vectores del árbitro se cachean por texto durante un documento y se limpian al
cambiar de documento; y el guardarraíl `relacion_con_arbitro` impide que un
concursante sea juez de sí mismo o de su familia.

**El precio de la recalibración** (hallazgo de esta campaña, §5): al estirar
`[b, 1] → [0, 1]`, las similitudes *moderadas* quedan comprimidas abajo. Un
fragmento temáticamente afín pero no literal, con coseno crudo 0.85, recibe:

```
sem = (0.85 − 0.7689) / (1 − 0.7689) = 0.0811 / 0.2311 ≈ 0.35
```

Esto, combinado con `lex ≈ 0` para paráfrasis, produce la **compresión de
escala** que el anclaje destapó.

---

## 3 · El método de anclaje: medir la vara con la única vara superior

Todas las capas anteriores son máquina-contra-máquina: la calibración ancla el
árbitro a sondas, la verificación ancla el examen al documento, el grounding
ancla al agente a sus tools. Faltaba el eslabón final: ¿la **nota** coincide
con el **juicio humano**? Ese es el anclaje.

**Protocolo** (`anclaje_humano.py`):

1. Se toma el examen congelado del documento y se muestrean `n` preguntas con
   semilla fija (`rng.choice`, determinista → los pares son reconstruibles).
2. El pipeline elegido recupera sus `k` fragmentos por pregunta → `n × k`
   pares (pregunta, fragmento).
3. El humano etiqueta **a ciegas** (sin ver ni la cita ni la nota): 0 = no
   responde, 1 = parcial, 2 = responde. Cada etiqueta se persiste
   atómicamente al instante (la sesión sobrevive a desconexiones).
4. Con las etiquetas se calculan tres contrastes:

**(a) Correlación de orden** — ¿la nota ordena como el humano?

```
ρ = Pearson(R(nota), R(humano))
```

con `R(·)` = rangos **medios** en los empates (imprescindible: `humano ∈
{0,1,2}` y `lex` empata masivamente en 0; los rangos por posición dependen del
orden de la lista y sesgan — corregido durante la campaña, §6).

**(b) Barrido de umbral** — ¿el corte vigente coincide con el criterio humano?
Para cada `U` de 0.05 a 0.95, con `relevante_humano := (humano ≥ criterio)`:

```
precisión(U) = TP/(TP+FP)   recall(U) = TP/(TP+FN)   F1 = 2PR/(P+R)
```

y se reporta el óptimo por F1 **con su meseta** (rango de umbrales empatados —
reportar solo el mínimo era engañoso cuando el F1 es plano).

**(c) MRR humano vs MRR máquina** — la métrica final del laboratorio, por
las dos varas:

```
MRR_humano  : acierto := humano ≥ criterio
MRR_máquina : acierto := nota ≥ U_vigente
sesgo = MRR_máquina − MRR_humano
```

Sesgo positivo = la máquina se apunta aciertos que el humano no le daría
(peligroso); negativo = descarta aciertos reales (conservador).

---

## 4 · Los datos: tres documentos, tres perfiles

| Documento | Perfil | Pipeline auditado | Pares |
|---|---|---|---|
| test_13.pdf | mixto·medio·lex-baja (denso, parafrástico) | recursive_500 · granite-311m-r2 · ensemble_k10_w0.3 (campeón coronado) | 99 |
| test_14.pdf | léxico-técnico | token_500 · granite-311m-r2 · bm25_k10 (líderes Fase A) | 100 |
| doc inmobiliario | es·ligero·lex-media (divulgativo) | recursive_500 · granite-311m-r2 · ensemble_k10_w0.3 | 70 |
| Annie.pdf | es·medio·lex-baja (diario literario) | token_500 · granite-311m-r2 · ensemble_k10_w0.7 (campeón, mrr 0.601) | 100 |
| junger.pdf | es·denso·lex-baja (prosa filosófica) | token_500 · bge-m3 · bm25_k5 (la alternativa económica del empate final) | 50 |

(Hubo una sesión previa sobre un paper académico en inglés que se descartó:
etiquetas invertidas por error de escala, y un fenómeno propio — las
**referencias bibliográficas como distractores adversariales**: entradas que
comparten entidades y vocabulario con la cita (lex y sem altos) sin responder
nada. Queda como hipótesis a confirmar en un cuarto anclaje.)

### Resultados por documento

**test_13** — Spearman: combinada **0.541** > léxico 0.508 > semántico 0.497.
La mezcla supera a cada componente: α trabaja. Barrido: F1 óptimo **0.73 en
U=0.20**; el vigente 0.5 da recall 0.21. Precisión **1.00 para todo U ≥ 0.25**.
MRR humano **1.000** vs máquina 0.950 (sesgo −0.05).

**test_14** — Spearman: léxico **0.947**; combinada = semántico = 0.296
(idénticos: los relevantes puntúan ≈1 en ambas mitades, así que el orden del
resto lo dicta el semántico). Barrido: **meseta perfecta 0.30–0.95** con
F1 = 0.83 (precisión 1.00, recall 0.71) — el vigente 0.5 es óptimo aquí. MRR
humano 0.276 vs máquina 0.196 — el pipeline (dos líderes de banco juntados a
mano) es objetivamente malo: lección independiente sobre el ceteris paribus
(§7).

**inmobiliario** — Spearman: léxico 0.781, semántico 0.193, combinada 0.200.
Barrido: óptimo 0.30 con meseta hasta 0.75+ (F1 0.53); vigente 0.5 da F1 0.44.
MRR humano = MRR máquina = 1.000 (sesgo 0).

**Jünger (prosa filosófica densa)** — el mejor certificado de la campaña,
obtenido en el documento mas dificil (los TRES bancos en disputa por
naturaleza del texto, no por falta de datos): AUC combinada **0.970**
(semantico 0.972 — su primera vez liderando, en el perfil denso; lexico
0.886), Spearman 0.73–0.77, MRR humano 0.900 vs maquina 0.850 (sesgo −0.05,
quinto consecutivo no-positivo). Barrido: optimo 0.25 (F1 0.85); vigente 0.5
=> F1 0.78 con precision 1.00. Leccion conceptual: la DISPUTA de los bancos y
la CALIDAD de la nota son cosas distintas — el laboratorio dudaba sobre que
config es mejor (opciones buenas apretadas); la nota distingue relevante de
irrelevante con un 97%. La silueta UMAP baja (0.039) predice bancos
disputados, no notas malas: mide si los chunks forman grupos entre si, no si
el arbitro puntua bien fragmento-contra-cita.

**Annie (diario literario)** — el anclaje del REGIMEN MEDIO: los tres
anteriores median stacks con MRR humano 1.000 (el techo, donde todo
instrumento parece bueno); Annie corona con 0.601 y pregunta si la nota
distingue aciertos de fallos de verdad. Resultado: MRR humano 0.553 vs
maquina 0.483 (sesgo −0.069, conservador; el 0.601 de la coronacion es
honesto), el umbral vigente 0.5 DENTRO de la meseta optima (0.35–0.65,
primera vez en cuatro documentos)... y la primera GRIETA: precision 0.71 en
la meseta — los primeros falsos positivos altos de la campaña (§7,
auto-similitud), con AUC debil en todas las formas (0.60–0.65).

### La compresión de escala (test_13), con las cuentas

La distribución de la nota entre los pares que el humano acredita es
**bimodal**: una isla alta (0.7–0.95) — fragmentos que *contienen la cita* —
y una isla baja (0.15–0.30) — fragmentos que *responden con otras palabras*.
La isla baja es estructural, no ruido:

```
paráfrasis típica:  lex ≈ 0.05–0.10  (pocos n-gramas comunes con la cita)
                    cos ≈ 0.85       (afín para el árbitro)
                    sem = (0.85−0.7689)/0.2311 ≈ 0.35
                    nota ≈ 0.5·0.07 + 0.5·0.35 ≈ 0.21
```

Entre las islas, un desierto: para puntuar 0.4–0.6 haría falta solapamiento
textual *parcial* con la cita, que es raro. Consecuencia: el umbral 0.5 no
está mal afinado — está **fuera de la distribución** donde vive la relevancia
parafrástica (recall se desploma 0.70 → 0.17 entre U=0.20 y U=0.50). Y la
precisión 1.00 desde 0.25 dice que la información sí está: la nota discrimina;
el corte decide qué definición de "acierto" compra (§8).

---

## 5 · El espejismo del Spearman y su resolución con AUC

Al barrer α contra los 269 juicios (los componentes lex/sem de cada par están
persistidos, así que el barrido es offline, sin re-etiquetar), la tabla de
Spearman mostró algo dramático:

```
forma          test_13   test_14   inmobil.   media
α = 0.5          0.541     0.296      0.199    0.345
α = 0.75         0.543     0.296      0.199    0.346
α = 1.0          0.508     0.947      0.781    0.745   ← ¿el léxico arrasa?
```

La discontinuidad 0.75 → 1.0 (0.296 → 0.947 en test_14, sin transición) era la
señal de alarma: una mezcla continua no salta así. El mecanismo, verificado
con un experimento sintético:

- Con α = 1.0, **todos los pares irrelevantes empatan en lex = 0**. Ese empate
  masivo coincide con el empate del juicio humano (todos 0) → con rangos
  medios, la varianza intra-bloque es cero y la correlación de rangos se
  dispara. El léxico cobra un **bonus por empatar donde el humano empata**.
- Con α < 1, el semántico rompe esos empates con valores continuos — una
  ordenación *interna del bloque de irrelevantes* que operativamente no
  importa (da igual el orden entre fragmentos que nadie acredita) — y el
  Spearman la castiga.

Es decir: la tabla no medía "qué forma discrimina mejor" sino "qué forma
empata donde el humano empata". La pregunta operativa correcta es:

```
AUC = P( nota(par relevante) > nota(par irrelevante) )
    = [ #{(p,n): p>n} + ½·#{(p,n): p=n} ] / (P·N)        (Mann-Whitney)
```

que ignora por construcción el orden interno de los bloques. Y el AUC cuenta
la historia opuesta:

```
forma          test_13   test_14   inmobil.   media
α = 0.0          0.837     0.907      0.654    0.799
α = 0.5          0.851     0.907      0.656    0.805   ← vigente
α = 0.75         0.850     0.907      0.656    0.804
α = 1.0          0.681     0.857      0.660    0.733   ← el peor en 3 de 3
max(lex,sem)     0.840     0.907      0.656    0.801
```

El léxico puro es la **peor** forma en los tres documentos: deja a los
relevantes-parafrásticos empatados a 0 con los irrelevantes (medio punto de
AUC por empate — no puede separarlos ni por azar). El semántico, villano del
Spearman, es quien carga con la separación donde el léxico enmudece. Y la
meseta 0.0–0.75 muestra que la mezcla es robusta al reparto exacto, con el
0.5 vigente en el máximo o rozándolo en los tres perfiles.

**Decisión tomada: α = 0.5 se queda.** La idea del α-por-perfil, que la tabla
de Spearman parecía respaldar, murió con evidencia — la mejor muerte posible.

Moralejas estadísticas del episodio, en frío:

1. Un salto discontinuo en un barrido continuo es un artefacto hasta que se
   demuestre lo contrario.
2. Spearman con empates masivos responde una pregunta distinta de la que
   parece; con datos por bloques (relevante/irrelevante), la vara operativa
   es AUC.
3. Los rangos medios en los empates no son un refinamiento opcional: sin
   ellos, el número depende del orden de la lista (el `argsort(argsort(·))`
   original se corrigió durante la campaña).

---

## 6 · El retrato del instrumento (lo que el anclaje certifica)

Tres documentos, tres perfiles, y estas propiedades salieron **tres de tres**:

- **Conservador en la dirección del MRR, cinco de cinco**: sesgo nunca
  positivo (−0.05, −0.08, 0.00, −0.07, −0.05). Los MRR del laboratorio son cotas
  inferiores del rendimiento según juicio humano.
- **"Precisión 1.00 por encima de U≈0.25" resultó ley de tres documentos, no
  universal**: en Annie (diario auto-similar) la precisión en la meseta cae a
  0.71 — los primeros falsos positivos altos de la campaña. La causa no es la
  nota sino la estructura del documento (§7): en textos auto-similares
  existen fragmentos gemelos de la cita que no responden. El certificado
  enmendado: la nota no regala relevancia *salvo cuando el documento contiene
  casi-duplicados de la cita*.
- **La mezcla α=0.5 separa bien** (AUC 0.60–0.97) y es robusta: nunca la
  peor forma, casi siempre rozando la mejor, sin saber de antemano que perfil
  toca (el lexico lidero en test_14/inmobiliario/Annie; el semantico en
  Jünger; la mezcla nunca paga mas de ~0.05 de AUC por la robustez).
- **La definición vigente de "acierto" es estricta**: `nota ≥ 0.5` ≈ "el
  fragmento contiene el pasaje fuente". El juicio humano usa una definición
  más amplia ("responde, aunque sea con otras palabras"). Ambas son
  coherentes; no son la misma.
- **Validación externa del campeón de test_13**: MRR humano = 1.000 — el
  primer fragmento del stack coronado responde siempre según el humano. La
  coronación del gimnasio queda respaldada por la única vara que no se puede
  amañar.
- **Punto débil localizado**: el documento divulgativo (AUC ≈ 0.656 en todas
  las formas de la nota — no es cuestión de α). Autopsia pendiente de sus
  desacuerdos.

---

## 7 · Hallazgos colaterales

- **El ceteris paribus no compone**: token_500 + bm25_k10 (líderes de sus
  bancos en test_14) dan MRR humano 0.276 — dos óptimos locales no hacen un
  óptimo global. Es exactamente la razón de existir de la Fase B factorial,
  ahora con confirmación por juicio humano.
- **Documentos AUTO-SIMILARES como adversarios** (confirmandose en Annie):
  un diario tiene decenas de entradas con las mismas escenas y vocabulario —
  fragmentos de OTRA entrada comparten lexico y semantica con la cita (nota
  alta) sin responder la pregunta. Es la version literaria de la hipotesis de
  las bibliografias academicas (lex+sem altos, relevancia cero). La cadena
  causal llega hasta el examen: "¿que le dice la madre acostada en la cama?"
  es inherentemente ambigua en un documento donde esa escena ocurre veinte
  veces, y la capa de discriminacion de la verificacion (cita vs otras
  paginas) pierde fuerza justo cuando las paginas se parecen entre si.
  Arreglos candidatos (pendientes): endurecer la discriminacion exigiendo
  MARGEN (no solo mayor afinidad), y/o anclar las preguntas de documentos
  repetitivos con referencias especificas (fechas de entrada).
- **Reproducibilidad**: pares deterministas por semilla + persistencia
  atómica por etiqueta + componentes guardados por par = el banco de juicios
  humanos es un activo reutilizable (el barrido de α costó cero etiquetas
  nuevas).

---

## 8 · Decisiones: tomadas y pendientes

**Cerradas**
- α = 0.5 (validado por AUC en tres perfiles; robusto en 0.0–0.75).
- Conservadurismo del instrumento: se conserva deliberadamente.
- Spearman con rangos medios + AUC en el informe del anclaje (código).
- Meseta en el barrido de umbral (código).

**Abiertas**
- **El umbral U** — la única perilla con desacuerdo real entre perfiles:
  óptimos 0.20 / 0.30–0.95 / 0.30–0.75; el vigente 0.5 solo es óptimo en el
  perfil léxico. Opciones sobre la mesa: (a) mantener 0.5 y documentar la
  definición estricta ("localiza el pasaje"); (b) U ≈ 0.25 global — F1 del
  mal-menor (~0.61/0.61/0.46), definición alineada con el juicio humano, a
  costa de perder la garantía de precisión 1.00. Cualquier cambio = **nueva
  era de métricas**, a ejecutar junto a la limpieza de memoria pendiente, en
  un solo movimiento.
  - La opción "(c) recalibración menos agresiva del semántico (cóncava o
    suelo más bajo)" queda **DESCARTADA con evidencia** (barrido offline sobre
    los 419 juicios, §9): toda transformación monótona de la nota deja el AUC
    invariante (el AUC solo depende del orden, que una monótona preserva); lo
    único que movía era el recall@0.5 (+0.012), es decir, desplazaba
    levemente el umbral efectivo. La forma de la nota es redundante con el
    umbral, y el umbral es más simple y reversible. El techo de AUC (~0.80
    medio) es de la INFORMACIÓN en lex+sem, no de su reescalado: subirlo pide
    cambiar los ingredientes (árbitro, tokenizador, tercera señal), no la
    forma.
- Autopsia de los dos documentos de separación débil: inmobiliario
  (AUC 0.656) y Annie (AUC 0.603, falsos positivos por auto-similitud —
  autopsia con los pares rel_auto≥0.5 ∧ humano=0).
- Quinto anclaje eventual: el paper académico re-etiquetado (confirma la
  variante bibliográfica del mismo fenómeno).
- Endurecimiento de la verificación para documentos auto-similares (margen en
  la capa de discriminación; preguntas ancladas a la entrada).

---

## 9 · Experimento: ¿mejora la forma de la nota la separación? (no)

Con `lex` y `sem` guardados por par (419 juicios), se barrió offline la FORMA
del componente semántico —des-transformando a coseno crudo
(`cos = sem·(1−b)+b`) y reaplicando— para seis variantes: afín actual, cóncavas
γ ∈ {0.7, 0.5, 0.35}, y suelos b ∈ {0.70, 0.65}. α fijo en 0.5.

```
forma              AUC_med   prec@.5   rec@.5
afín (γ=1, actual)   0.798     0.943    0.436
cóncava γ=0.7        0.798     0.943    0.440
cóncava γ=0.5        0.797     0.943    0.448
cóncava γ=0.35       0.798     0.943    0.448
suelo b=0.70         0.797     0.943    0.440
suelo b=0.65         0.797     0.943    0.440
```

**Resultado: la forma no cambia nada.** AUC y precisión constantes; recall se
mueve +0.012 como mucho. Es un teorema con datos: toda transformación MONÓTONA
preserva el orden de las notas, y el AUC solo depende del orden — luego ninguna
puede mejorar la separación. Lo poco que movía (recall@0.5) es un
desplazamiento del umbral efectivo, no una mejora de discriminación: la palanca
"forma" es redundante con la palanca "umbral", que es más simple y reversible.
El techo de ~0.80 de AUC medio es de la INFORMACIÓN que lex+sem tienen sobre el
juicio humano, no de su reescalado; subirlo exige cambiar ingredientes, no
forma. Consecuencia: opción (c) descartada, la sentada de decisión se reduce a
umbral + primaria del splitter.

---

## 10 · Tabla maestra: Spearman y AUC por documento y componente

La foto completa del instrumento contra el juicio humano (419 pares, criterio
humano ≥ 1; nota combinada con α=0.5). Regenerable en cualquier momento desde
`anclaje_humano` con una celda offline.

```
documento          n  | Sp comb  Sp lex  Sp sem | AUC comb  AUC lex  AUC sem
---------------------------------------------------------------------------
Annie.pdf        100  |  0.138    0.414   0.136 |   0.603    0.652    0.601
test_13.pdf       99  |  0.657    0.538   0.610 |   0.851    0.681    0.837
test_14.pdf      100  |  0.496    0.849   0.496 |   0.907    0.857    0.907
test_15.pdf       70  |  0.332    0.606   0.326 |   0.656    0.660    0.654
junger.pdf        50  |  0.727    0.767   0.731 |   0.970    0.886    0.972
---------------------------------------------------------------------------
MEDIA                 |  0.470    0.635   0.460 |   0.798    0.747    0.794
```

Lecturas que esta tabla deja cerradas:

- **El espejismo del Spearman, cuantificado en toda la campaña**: el lexico
  gana el Spearman en 4 de 5 documentos (media 0.635 vs 0.470 de la
  combinada) pero pierde en AUC (0.747 vs 0.798). El caso flagrante es
  test_13: Sp lex 0.538 con AUC lex 0.681 — el bonus de empates (lex=0
  masivo coincidiendo con humano=0) infla el orden sin separar nada. Elegir
  α por Spearman habria dado α=1.0; el AUC lo desmiente.
- **α=0.5 revalidado**: AUC comb ≥ max(lex, sem) − 0.05 en TODOS los
  documentos; en test_13 la mezcla SUPERA a ambos componentes (0.851 vs
  0.681/0.837) — lex y sem aportan informacion complementaria. El seguro de
  la mezcla cuesta ≤0.05 de AUC sin saber el perfil de antemano.
- **Annie, el caso extremo**: Sp comb 0.138 (casi nulo) y el semantico
  aportando practicamente nada (Sp 0.136, AUC 0.601 ~ azar+0.1): en Annie la
  mitad semantica de la nota es casi peso muerto y hasta el orden fino lo
  lleva el lexico (Sp lex 0.414). Mejor argumento empirico registrado para
  un eventual α POR PERFIL — expediente dormido hasta tener mas documentos
  anclados (con 5, seria sobreajuste).
- **Jerarquia de dificultad**: Jünger 0.970 > test_14 0.907 > test_13 0.851 >
  inmobiliario 0.656 ≈ Annie 0.603. Los dos flojos comparten diagnostico
  (distractores estructurales que ni lex ni sem ven) y son el mapa de donde
  una tercera señal tendria mas que ganar.

---

## 11 · Síntesis interpretativa: U, α y el carácter del instrumento

*(Lectura conjunta de los barridos de la §10 y de las curvas precisión-recall
por mezcla; redactada a partir del análisis de Xavier, con dos precisiones.)*

Con la campaña de anclaje delante, la decisión sobre U y α es por fin una
decisión informada sobre **cómo de conservador es el laboratorio**. Con U se
define qué cuenta como relevante: mantener U=0.5 (decidido) es aceptar el
sacrificio de recall sobre los parafrásticos a cambio de precisión 1.00 —
U=0.3 haría el instrumento más sensible y menos conservador, al precio de
perder esa garantía. Esa disyuntiva depende del propósito, no de los datos:
es normativa.

α, en cambio, es distributivo: dado que la nota no se distribuye igual para
fragmentos léxicos y parafrásticos, el reparto lex/sem decide **dónde caen
los puntos ciegos** del instrumento. Y la interacción entre conservadurismo
(U) y reparto (α) es asimétrica, pero conviene describirla con exactitud:

- Con α alto (refuerzo léxico), los parafrásticos (lex = 0 exacto) se
  pierden **en la puerta** — ya con U minúsculo quedan bajo el corte. La
  curva de recall parece "caer suave" a partir de ahí no porque pierda poco,
  sino porque **ya no le queda nada que perder**: su sensibilidad tiene un
  techo estructural (la fracción de relevantes con solapamiento léxico) que
  ningún U puede superar.
- Con α bajo (refuerzo semántico), los parafrásticos se capturan a U bajo y
  se pierden **de golpe** cuando U cruza su isla comprimida (~0.25–0.50): el
  acantilado. El peaje del conservadurismo se paga en ese tramo.
- La comparación honesta no es "suave vs acusada" sino **"pérdida total por
  adelantado vs pérdida escalonada por el camino"**. A U alto ambas
  trayectorias convergen: solo los literales.

Dos precisiones que corrigen la moraleja ingenua:

1. **El peaje no lo causa "lo semántico" sino su compresión** — la
   anisotropía aparca a los parafrásticos en una isla estrecha y baja (es la
   distribución del coseno crudo, no la fórmula; la afín estira uniforme).
   Un semántico sin compresión no tendría acantilado.
2. **Con U conservador, bajar el peso semántico no recupera sensibilidad**
   (los parafrásticos son inalcanzables para el léxico a cualquier U: el
   techo no sube quitándole peso al sem) **y sí degrada la ordenación** — el
   AUC de test_13 cae de 0.851 a 0.681 con α=1.0, y el MRR intra-documento
   con él. Con U estricto, α apenas mueve el recall; sigue moviendo de lleno
   la ordenación, que es lo que el laboratorio consume para coronar.

La máxima resultante: **un instrumento conservador no cobra los beneficios
de sensibilidad del semántico (solo se cobran a U bajo), pero renunciar al
semántico le costaría la ordenación en los documentos parafrásticos. La
pareja coherente es la vigente: α equilibrado (0.5) para ordenar bien en
todos los perfiles, U donde la definición lo exija (0.5, estricto), y el
punto ciego parafrástico asumido con inventario — no negado.**

Nota de planos, para no mezclar nieblas: el juicio del instrumento "se
nubla" según el tipo de documento por dos mecanismos distintos. En recall se
nubla por el **corte** (islas bajo U — gobernable con las perillas); en
ordenación se nubla por la **información** (el techo de Annie ~0.60 con
cualquier α — solo gobernable con nuevas señales). Las perillas reparten la
primera niebla; la segunda es propiedad del material.

**Coda: la lectura del histograma de las tres anatomías** *(distribución de
la nota con α=0.5, apilada por juicio humano, para test_13 / test_14 /
Jünger — la figura de las tres islas)*. En un texto variado como test_13,
mover el umbral entre ~0.35 y 0.5 no varía prácticamente el recall (es el
desierto entre islas). Hacer el instrumento menos conservador lo hará menos
preciso — aceptará la franja donde relevantes e irrelevantes cohabitan, sin
que su capacidad de distinguirlos cambie: el orden de las notas es el mismo,
solo se corta más abajo (U gobierna el corte; la ordenación solo la
gobiernan α y la información de lex+sem — el AUC es invariante a U). Hacerlo
más conservador irá perdiendo recall gradualmente, erosionando la isla
literal. Y el precio de bajar U lo fija el **solapamiento**, que es
propiedad del documento: en test_13 la isla parafrástica cohabita con los
irrelevantes (bajar U paga caro en precisión); en Jünger apenas se solapan
(AUC 0.970 — la misma bajada saldría casi gratis). Corolario alentador:
"aceptar más grises sabiendo ordenarlos" significa que el MRR (que consume
el orden) sufre menos que la precisión al relajar U. DocuAgent-RAG hereda
estas sensibilidades a través de los MRR que lee; su conservadurismo propio
(contrato: empates, varianza cero, no perseguir) es otra capa, gobernable
por separado.

**Los tres contratos con U** *(lectura de los tres paneles del histograma,
uno por carácter de documento)*:

- **test_14 (técnico/léxico): margen enorme, rebaja inútil.** Anatomía
  binaria — literales arriba, todo lo demás abajo, desierto 0.35–0.85. U
  puede relajarse muchísimo sin coste alguno, pero tampoco compra nada (no
  hay relevantes que rescatar en el desierto); el riesgo no es una pérdida
  gradual sino un precipicio localizado: cruzar U por debajo de ~0.30 hunde
  la precisión de golpe (los pocos relevantes no-literales cohabitan con la
  masa entera de irrelevantes, 86 contra ~4). El perfil más tolerante al
  umbral — mientras no se cruce la frontera.
- **test_13 (variado): margen medio, rebaja cara.** La isla parafrástica
  cohabita con los irrelevantes en los mismos bins: cada paso de relajación
  compra recall pagando precisión en proporción — la pendiente de peaje.
- **Jünger (parafrástico): margen estrecho, rebaja casi gratis.** La misma
  idea se reformula una y otra vez y el margen muerto de U es el más pequeño
  del corpus — pero los relevantes reformulados viven ESCALONADOS justo
  encima de los irrelevantes, casi sin cohabitar (AUC 0.970): hacerlo menos
  conservador recupera sensibilidad sin pagar peaje. Y no es retórica: el
  barrido del anclaje de Jünger da, en U=0.30, precisión 1.00 con recall
  0.71 (vs 0.64 del vigente) — la rebaja 0.5→0.30 es gratis en sentido
  estricto; el primer peaje aparece en 0.25 (precisión 0.92, recall 0.79).

Moraleja de los tres paneles: **el contrato de cada documento con U no lo
fija el ancho de su desierto sino su SOLAPAMIENTO** — margen enorme y rebaja
inútil (test_14), margen medio y rebaja cara (test_13), margen estrecho y
rebaja casi gratis (Jünger). Tres anatomías, tres contratos, una sola causa.
Y el corolario en su caso extremo: la decisión conservadora (U=0.5) fue mas
barata precisamente donde el instrumento es mejor — en Jünger, lo que el
corte sacrifica en recall, el MRR lo recupera por ordenacion.

**El cruce en U=0.5 (hallazgo del barrido F1 por mezclas)**: en el entorno
del umbral vigente, la elección de α es prácticamente indiferente para el F1
— las cinco curvas convergen (~0.51–0.53), porque con el corte estricto
todas las mezclas acreditan esencialmente el mismo conjunto (los literales:
los parafrásticos quedan debajo para cualquier α, los literales encima para
cualquier α). Esto NO significa que U=0.5 equilibre el instrumento entre
tipos de documento — al contrario: es el corte cuya severidad pagan
asimétricamente los parafrásticos, con cualquier mezcla. Lo que el cruce
demuestra es **dónde trabaja cada perilla**: en el régimen estricto, α no
actúa sobre el corte (recall/precisión) sino sobre la ordenación (AUC/MRR) —
y es ahí, en la tabla maestra (§10), donde α=0.5 se gana el título de
equilibrado. La pareja vigente (α=0.5, U=0.5) no es equilibrada porque las
curvas se crucen ahí, sino porque α equilibra la ordenación entre perfiles y
U implementa la definición estricta cuyo precio asimétrico está inventariado
y aceptado. El cruce es el diagrama de por qué las dos perillas no se pisan
— no un certificado de neutralidad.

---

## Apéndice · Fórmulas de referencia rápida

```
nota      = α·lex + (1−α)·sem                       α = 0.5
lex       = |G(f) ∩ G(c)| / |G(c)|                  n-gramas, tokenizador ES
sem       = max(0, (cos(f,c) − b)/(1 − b))          b = 0.7689 (e5-small)
acierto   = nota ≥ U                                 U = 0.5 vigente
MRR       = media_q(1/rank del primer acierto)
Spearman  = Pearson(rangos medios)
AUC       = [#(p>n) + ½·#(p=n)] / (P·N)
IC 95%    = bootstrap percentil sobre la serie por pregunta
prob(A>B) = P(d̄*>0) + ½·P(d̄*=0)   (pareado, mid-p)
CONCLUYENTE = 0 ∉ IC(d̄)
```
