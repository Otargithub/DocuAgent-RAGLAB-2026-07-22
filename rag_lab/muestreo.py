"""
Remuestreo de preguntas de test para medir la ESTABILIDAD de verdad.

El problema
-----------
Repetir un banco n_runs veces sobre exactamente los mismos datos, las mismas
preguntas y un juez determinista (temperatura 0) produce siempre el mismo
ganador. El "acuerdo entre runs" sale ~100% por construcción: mide el
determinismo del modelo, no la robustez de la decisión.

La solución
-----------
Cada run (a partir del segundo) evalúa sobre un REMUESTREO bootstrap de las
preguntas de test: se muestrea con reemplazo el mismo número de preguntas. Si el
ganador se mantiene pese a cambiar la muestra de preguntas, es robusto; si cambia
según qué preguntas tocaron, el banco sale (correctamente) inestable.

El remuestreo es determinista por (doc_type, run): mismo documento y mismo run
-> misma muestra, así que los resultados son reproducibles entre ejecuciones y
entre procesos (random.Random siembra con cadenas vía SHA-512).

El run 1 usa SIEMPRE el conjunto completo: es la evaluación canónica. Los runs
2..n son las sondas de robustez alrededor de ella.
"""

from __future__ import annotations

import random


def resamplear_preguntas(preguntas_test: list[dict], doc_type: str, run: int) -> list[dict]:
    """
    Devuelve un bootstrap (muestreo con reemplazo, mismo tamaño) de las preguntas.

    - run <= 1  o  menos de 2 preguntas -> se devuelve el conjunto completo tal cual.
    - en otro caso -> bootstrap determinista por (doc_type, run).
    """
    if run <= 1 or len(preguntas_test) < 2:
        return list(preguntas_test)
    rng = random.Random(f"{doc_type}|run{run}")
    return [rng.choice(preguntas_test) for _ in range(len(preguntas_test))]
