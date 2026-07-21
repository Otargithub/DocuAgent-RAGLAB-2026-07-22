"""
La capa de INFORMES del meta-supervisor: prosa por pantalla, cero decisiones.

Extraida de meta.py en la fase 4 del refactor (2026-07-16). Vive aparte por
la doctrina de la casa: la prosa no decide — este modulo no importa ni
estadistica ni memoria (recibe todo masticado) y ningun veredicto depende de
lo que aqui se formatee.
"""
from __future__ import annotations

from pathlib import Path


def diagnostico_a_texto(diag: dict, tasa_acierto: dict | None) -> str:
    """Los priors por perfil, en texto (pista, no certeza)."""
    partes = [f"\n[meta] Perfil del documento: {diag.get('descriptor', '?')}"]
    partes.append("[meta] Priors por perfil (documentos parecidos; pista, no certeza):")
    for nombre, pr in diag.get("bancos", {}).items():
        if not pr.get("disponible"):
            partes.append(f"  {nombre:10}: sin vecinos comparables")
            continue
        partes.append(
            f"  {nombre:10}: "
            f"{'[HISTORIAL PROPIO, no transferencia] ' if pr.get('autorreferencial') else ''}"
            f"tipico={pr.get('ganador')} "
            f"(acuerdo {pr.get('acuerdo'):.0%} de {pr.get('n_vecinos')} vecinos, "
            f"sim media {pr.get('sim_media')}) | resuelto alli "
            f"{pr.get('frac_resuelto', 0):.0%} | IC medio={pr.get('ic_medio')} "
            f"| motivo tipico={pr.get('motivo_dominante')} "
            f"| valor esperado~{pr.get('valor_esperado')}")
    if tasa_acierto:
        resumen = ", ".join(
            f"{b}:{v['acierto']:.0%} exacto/{v['acierto_familia']:.0%} familia (n={v['n']})"
            for b, v in sorted(tasa_acierto.items()))
        partes.append(f"  [tasa de acierto historica de estos priors: {resumen}]")
    return "\n".join(partes)


def informe_sintesis(sesion, plan: dict, lineas: list[str], stack: dict,
                     runs_ejecutados: int | None = None,
                     runs_ampliacion: int = 0) -> str:
    """El informe final del meta-supervisor (certeza, stack, economico, prosa)."""
    bordes = "=" * 70
    partes = [f"\n{bordes}", "  INFORME DEL META-SUPERVISOR", bordes,
              f"Documento     : {Path(sesion.pdf_path).name} [{sesion.doc_type}]",
              f"Objetivo      : {plan['objetivo']}",
              f"Runs: {runs_ejecutados or plan['n_runs']} ejecutados "
              f"({(runs_ejecutados or plan['n_runs']) - runs_ampliacion} "
              f"de remuestreo sobre max {plan.get('presupuesto', {}).get('max_runs', plan['n_runs'])} "
              f"+ {runs_ampliacion} por ampliacion del examen)",
              "\nCERTEZA POR BANCO:", *lineas,
              "\nSTACK RECOMENDADO:"]
    for comp, cfg in stack.get("stack_recomendado", {}).items():
        # Un unico ESTADO legible en vez de los flags crudos (resuelto y
        # en_disputa pueden ser True A LA VEZ: banco estable+concluyente
        # CUYO grupo de finalistas aun empata por IC -> lo dirime la final).
        if cfg.get("via") == "fase_b":
            estado = "coronado en Fase B"
        elif cfg.get("via") == "fijado_por_usuario":
            estado = "fijado por el usuario"
        elif cfg.get("resuelto") and cfg.get("en_disputa"):
            estado = "resuelto (empate de IC -> Fase B)"
        elif cfg.get("resuelto"):
            estado = "resuelto"
        else:
            estado = "en disputa"
        partes.append(f"  {comp:10}: {cfg.get('etiqueta')}  [{estado}]")
        partes.append(f"              config: {cfg.get('config')}")
    eco = stack.get("stack_alternativo_economico") or {}
    if eco:
        partes.append("\nSTACK ALTERNATIVO ECONOMICO (mas barato dentro del empate):")
        for comp, cfg in eco.items():
            partes.append(f"  {comp:10}: {cfg.get('etiqueta')} (coste={cfg.get('coste')})")
    partes.append(f"\nConfianza global: {stack.get('confianza_global', '?')}")
    partes.append(f"\nJustificacion: {stack.get('justificacion', '')}")
    for aviso in stack.get("alertas", []):
        partes.append(f"  AVISO: {aviso}")
    partes.append(bordes)
    return "\n".join(partes)
