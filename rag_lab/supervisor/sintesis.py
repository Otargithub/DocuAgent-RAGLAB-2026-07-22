"""
SINTESIS — ensamblado determinista de la respuesta final.

El stack recomendado es el lider evaluado por banco (los empates los dirime la
Fase B) y el alternativo economico es el mas barato dentro del empate. Aqui no
decide ningun LLM: solo evidencia medida y coste declarado.
"""
from __future__ import annotations


def _coste_config(cfg: dict) -> float:
    """Coste API por uso; los modelos locales/HF cuestan 0."""
    cfg = cfg or {}
    return float(cfg.get("cost_per_1k_tokens") or cfg.get("cost_per_1k_queries") or 0.0)


def stack_economico_det(certezas: dict) -> dict:
    """
    Alternativa economica DETERMINISTA: por banco, dentro del grupo de empate
    (finalistas = estadisticamente indistinguibles), el de MENOR coste;
    desempate por valor primario. Sin eleccion del LLM: si empatan en calidad,
    el barato es objetivamente preferible.
    """
    eco = {}
    for banco, cer in (certezas or {}).items():
        finalistas = cer.get("finalistas") or []
        if not finalistas:
            continue
        elegido = min(finalistas,
                      key=lambda f: (_coste_config(f.get("config")),
                                     -(f.get("valor") or 0.0)))
        eco[banco] = {"etiqueta": elegido.get("etiqueta"),
                      "config": elegido.get("config"),
                      "coste": _coste_config(elegido.get("config")),
                      "en_disputa": cer.get("es_empate")}
    return eco


