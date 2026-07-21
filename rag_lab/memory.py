"""
MEMORIA v2 — un almacen plano, tres vistas, cero redundancia.

Registro de eventos de solo-anadir con CUATRO listas planas:

  sesiones      {run_id, pdf_nombre, doc_hash, perfil, n_preguntas}
  experimentos  {run_id, banco, run, veredicto}
  decisiones    {run_id, stack}
  predicciones  {run_id, banco, esperado, base, real, acierto}

Sin `doc_type` (el LLM no clasifica): el documento se caracteriza por su PERFIL
numerico determinista (ver perfil.py), y LDA/UMAP viven DENTRO del perfil.

Las tres vistas de lectura:
  1. certeza(run_id, banco, clave)   que se de ESTE documento (validada por tests)
  2. priors(perfil, banco, clave)    que funciono en documentos PARECIDOS
  3. resumen()/_leer()               el crudo, para analisis humano

Todo prior emitido se registra como PREDICCION y se resuelve contra el ganador
real al coronar: el aprendizaje es comprobable (tasa_acierto), no decorativo.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CONFIG
from .perfil import similitud


def _familia(etiqueta) -> str:
    """
    Familia/efecto de una etiqueta: 'recursive_1000'->'recursive',
    'ensemble_k10_w0.7'->'ensemble', 'a0.8_baseline'->'baseline' (en metadata el
    efecto es el mecanismo, no el alpha), 'bge-m3'/'bge-small...'->'bge',
    'ms-marco-...'->'ms'. Medir el acierto de un prior SOLO a etiqueta exacta lo
    penaliza injustamente: los ganadores end-to-end demostraron ser estables a
    nivel de familia, no de etiqueta (y el dedupe puede renombrar etiquetas).
    """
    e = str(etiqueta or "")
    if "_" in e:
        cabeza, cola = e.split("_", 1)
        if re.fullmatch(r"a\d+(?:\.\d+)?", cabeza):   # metadata: aX.Y_config
            return cola.split("_")[0]
        return cabeza
    return e.split("-")[0] or e


class Memoria:
    """Acceso a la memoria persistida en un unico fichero JSON (manana, tablas)."""

    def __init__(self, fichero: Path | None = None):
        self.fichero = Path(fichero) if fichero else CONFIG.fichero_memoria
        self.fichero.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Lectura / escritura del fichero completo
    # ------------------------------------------------------------------ #
    def _leer(self) -> dict[str, list]:
        """Devuelve el diario completo. Si no existe, devuelve uno vacío."""
        if self.fichero.exists():
            with open(self.fichero, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"sesiones": [], "experimentos": [], "decisiones": [], "predicciones": []}

    def _escribir(self, datos: dict) -> None:
        """Escritura atómica: escribimos a un temporal y renombramos."""
        temporal = self.fichero.with_suffix(".tmp")
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        temporal.replace(self.fichero)

    def _anadir(self, coleccion: str, evento: dict) -> None:
        """Añade un evento con marca de tiempo a la colección indicada."""
        datos = self._leer()
        evento = {"guardado_en": datetime.now().isoformat(timespec="seconds"), **evento}
        datos.setdefault(coleccion, []).append(evento)
        self._escribir(datos)

    # ------------------------------------------------------------------ #
    # Escritura
    # ------------------------------------------------------------------ #
    def congelar_preguntas(self, doc_hash: str, preguntas: list[dict],
                           info: dict | None = None) -> None:
        """
        CONGELA el examen de un documento: las preguntas verificadas quedan
        asociadas al hash de contenido, y las corridas futuras del MISMO
        documento (aunque cambie la ruta o la máquina) miden contra el mismo
        examen. Sin esto, cada corrida forzada regenera preguntas y las
        comparaciones entre corridas llevan el ruido de exámenes distintos.
        Se guarda un evento nuevo por congelación: `preguntas_congeladas`
        devuelve siempre el más reciente (re-congelar = actualizar).
        """
        if not doc_hash or not preguntas:
            return
        self._anadir("examenes", {"doc_hash": doc_hash, "preguntas": preguntas,
                                  "n": len(preguntas),
                                  "verificacion": info or {}})

    def preguntas_congeladas(self, doc_hash: str) -> list[dict] | None:
        """Examen congelado más reciente para este documento, o None."""
        if not doc_hash:
            return None
        for ev in reversed(self._leer().get("examenes", [])):
            if ev.get("doc_hash") == doc_hash and ev.get("preguntas"):
                return list(ev["preguntas"])
        return None

    def registrar_sesion(self, run_id: str, pdf_nombre: str, doc_hash: str,
                         perfil: dict, n_preguntas: int) -> None:
        perfil = dict(perfil or {})
        if perfil and "descriptor" not in perfil:
            # el descriptor textual ("mixto|medio|lex-baja") es lo que agrupan
            # los analisis transversales; sin el, todo caia en el perfil '?'
            try:
                from .perfil import descriptor
                perfil["descriptor"] = descriptor(perfil)
            except Exception:  # noqa: BLE001
                pass
        self._anadir("sesiones", {"run_id": run_id, "pdf_nombre": pdf_nombre,
                                  "doc_hash": doc_hash, "perfil": perfil,
                                  "n_preguntas": n_preguntas})

    def actualizar_perfil(self, run_id: str, cambios: dict) -> None:
        """Completa el perfil de una sesion (p.ej. dispersion_lda, silueta_umap)."""
        datos = self._leer()
        for s in datos.get("sesiones", []):
            if s.get("run_id") == run_id:
                s.setdefault("perfil", {}).update(cambios or {})
        self._escribir(datos)

    def registrar_experimento(self, run_id: str, banco: str, veredicto: dict,
                              run: int | None = None) -> None:
        self._anadir("experimentos", {"run_id": run_id, "banco": banco,
                                      "run": run, "veredicto": veredicto})

    def registrar_decision(self, run_id: str, stack: dict) -> None:
        self._anadir("decisiones", {"run_id": run_id, "stack": stack})

    def registrar_prediccion(self, run_id: str, banco: str, esperado: str,
                             base: dict | None = None,
                             autorreferencial: bool = False) -> None:
        """Un prior emitido = una prediccion pendiente de resolver."""
        self._anadir("predicciones", {"run_id": run_id, "banco": banco,
                                      "esperado": esperado, "base": base or {},
                                      "autorreferencial": bool(autorreferencial),
                                      "real": None, "acierto": None})

    def resolver_predicciones(self, run_id: str, ganadores: dict) -> dict:
        """Cierra el bucle: compara lo predicho con el ganador real por banco."""
        datos = self._leer()
        resueltas, aciertos, aciertos_familia = 0, 0, 0
        for p in datos.get("predicciones", []):
            if p.get("run_id") != run_id or p.get("real") is not None:
                continue
            real = ganadores.get(p.get("banco"))
            if real is None:
                continue
            p["real"] = real
            p["acierto"] = bool(str(p.get("esperado")) == str(real))
            resueltas += 1
            aciertos += int(p["acierto"])
            aciertos_familia += int(_familia(p.get("esperado")) == _familia(real))
        self._escribir(datos)
        return {"resueltas": resueltas, "aciertos": aciertos,
                "aciertos_familia": aciertos_familia}

    # ------------------------------------------------------------------ #
    # Vista 1: certeza de ESTE documento (logica validada; sin cambios)
    # ------------------------------------------------------------------ #
    def experimentos_de_documento(self, run_id: str, banco: str | None = None) -> list[dict]:
        """Experimentos de UN documento concreto (este run_id), opcionalmente de un banco.

        Ordenados por índice de run (el canónico = run 1 primero); si algún
        registro antiguo no tiene `run`, se respeta el orden de inserción, que
        para un diario de solo-añadir también deja el run 1 al principio.
        """
        datos = self._leer()
        salida = [e for e in datos["experimentos"] if e["run_id"] == run_id]
        if banco:
            salida = [e for e in salida if e["banco"] == banco]
        salida.sort(key=lambda e: (e.get("run") is None, e.get("run") or 0))
        return salida

    def certeza(self, run_id: str, banco: str, clave_decision: str) -> dict:
        """
        Cuán resuelta está la decisión de un banco PARA ESTE DOCUMENTO (run_id).

        A diferencia de `consistencia_entre_documentos`, mira solo los runs de
        este documento (arregla el viejo defecto de mezclar todos los documentos
        de un `doc_type`). Combina DOS señales independientes y complementarias:

          - `acuerdo`     : fracción de runs de ESTE documento cuyo ganador
                            coincide (robustez frente al remuestreo de preguntas
                            de test). Mide estabilidad.
          - `concluyente` : si en el run canónico el margen pareado del líder
                            sobre su rival NO cruza 0 (la calidad los distingue).
                            Mide separación estadística.

        Un banco solo está `resuelto` si AMBAS se cumplen. Y cuando no lo está,
        `motivo` dice POR QUÉ, que es lo que el planificador necesita para elegir
        la palanca correcta (más runs vs. más preguntas de test vs. desempatar
        por secundarios). Devuelve `disponible=False` si no hay datos.
        """
        exps = self.experimentos_de_documento(run_id, banco)
        if not exps:
            return {"disponible": False, "banco": banco, "motivo": "sin_datos",
                    "resuelto": False, "n_runs": 0}

        valores = [e["veredicto"].get(clave_decision) for e in exps]
        valores = [v for v in valores if v]
        n = len(valores)
        if n == 0:
            return {"disponible": False, "banco": banco, "motivo": "sin_ganador",
                    "resuelto": False, "n_runs": 0}

        conteo = Counter(valores)

        # Run canónico = el de MÁS preguntas (evidencia más ancha => IC más
        # estrecho). Empate por orden de run (el primero). Esto es lo que hace
        # que escalar preguntas (Fase A.2) estreche el IC que se lee.
        def _n_preg(e: dict) -> int:
            return ((e.get("veredicto", {}) or {}).get("_primaria") or {}).get("n", 0) or 0
        canonico = max(exps, key=lambda e: (_n_preg(e), -(e.get("run") or 0)))["veredicto"]

        # GANADOR CANÓNICO: el líder evaluado del run canónico — la MISMA verdad
        # que usan los finalistas y el stack. La moda entre runs era arbitraria
        # en empates totales (salía el del run 1 por orden de inserción) y
        # contaminaba el informe y la tasa de acierto de los priors. El acuerdo
        # pasa a medir cuántos runs coinciden con esa verdad canónica.
        ganador = canonico.get(clave_decision) or conteo.most_common(1)[0][0]
        acuerdo = round(conteo.get(ganador, 0) / n, 3)
        prim = canonico.get("_primaria") or {}
        comp = canonico.get("_comparacion_lider") or {}
        finalistas = canonico.get("_finalistas") or []
        concluyente = comp.get("concluyente") if comp else None

        ic = None
        if prim:
            ic = {"media": prim.get("valor"), "ic_bajo": prim.get("ic_bajo"),
                  "ic_alto": prim.get("ic_alto"),
                  "amplitud": (round(prim["ic_alto"] - prim["ic_bajo"], 4)
                               if prim.get("ic_alto") is not None
                               and prim.get("ic_bajo") is not None else None),
                  "n": prim.get("n")}

        # ¿Resuelto? y, si no, ¿por qué? (la palanca correcta para T3/T4)
        if n < 2:
            resuelto, motivo = False, "pocos_runs"          # -> más runs
        elif acuerdo < 0.75:
            resuelto, motivo = False, "desacuerdo_entre_runs"  # -> más runs
        elif concluyente is False:
            resuelto, motivo = False, "margen_no_concluyente"  # -> más preguntas de test
        elif concluyente is None:
            resuelto, motivo = False, "sin_comparacion"        # -> 1 sola config o sin stats
        else:
            resuelto, motivo = True, None

        return {
            "disponible": True,
            "banco": banco,
            "ganador": ganador,
            "n_runs": n,
            "acuerdo": acuerdo,
            "distribucion": dict(conteo),
            "metrica_primaria": prim.get("nombre"),
            "ic_primaria": ic,
            "concluyente": concluyente,
            "margen_vs_2": comp.get("dif_media"),
            "rival": comp.get("rival"),
            "finalistas": finalistas,
            "n_finalistas": len(finalistas),
            "es_empate": len(finalistas) > 1,
            "resuelto": resuelto,
            "motivo": motivo,
        }

    # ------------------------------------------------------------------ #
    # Vista 2: priors por perfil (que funciono en documentos PARECIDOS)
    # ------------------------------------------------------------------ #
    def perfil_de(self, run_id: str) -> dict:
        for s in reversed(self._leer().get("sesiones", [])):
            if s.get("run_id") == run_id:
                return s.get("perfil") or {}
        return {}

    def priors(self, perfil: dict, banco: str, clave_decision: str,
               k: int = 3, excluir_run_id: str | None = None,
               excluir_doc_hash: str | None = None,
               min_similitud: float = 0.5) -> dict:
        """
        Los k documentos mas parecidos por PERFIL (no por etiqueta) y que dicen
        sus certezas sobre este banco. Fusiona en una vista lo que antes hacian
        estabilidad/consistencia/tendencia/vecinos. Devuelve el ganador tipico,
        el acuerdo entre vecinos, cuanto costo resolver alli (frac_resuelto,
        ic_medio, motivo_dominante) y el rendimiento esperado.
        """
        sesiones = [s for s in self._leer().get("sesiones", [])
                    if s.get("run_id") != excluir_run_id and s.get("perfil")]
        # ANTI-AUTORREFERENCIA: las sesiones previas del MISMO documento entran
        # con similitud 1.0 y acaparan el vecindario, convirtiendo el prior en
        # "en ESTE documento gano X las ultimas veces" (apuesta casi segura que
        # inflaba la tasa de acierto). El prior debe ser TRANSFERENCIA entre
        # documentos: se excluye el propio doc_hash; solo si no queda ningun
        # vecino ajeno se recurre al historial propio, MARCADO como tal.
        ajenos, propios = [], []
        for s in sesiones:
            sim = similitud(perfil, s["perfil"])
            if sim < min_similitud:
                continue
            destino = (propios if excluir_doc_hash
                       and s.get("doc_hash") == excluir_doc_hash else ajenos)
            destino.append((sim, s))
        autorreferencial = False
        candidatos = ajenos
        if not candidatos and propios:
            candidatos, autorreferencial = propios, True
        if not candidatos:
            return {"disponible": False, "banco": banco, "n_vecinos": 0}
        candidatos.sort(key=lambda x: -x[0])
        vecinos = candidatos[:k]

        detalles, ganadores, motivos, amplitudes, resueltos = [], [], [], [], 0
        for sim, s in vecinos:
            c = self.certeza(s["run_id"], banco, clave_decision)
            if not c.get("disponible"):
                continue
            ganadores.append(c.get("ganador"))
            if c.get("motivo"):
                motivos.append(c["motivo"])
            amp = (c.get("ic_primaria") or {}).get("amplitud")
            if amp is not None:
                amplitudes.append(amp)
            resueltos += int(bool(c.get("resuelto")))
            detalles.append({"run_id": s["run_id"], "similitud": sim,
                             "ganador": c.get("ganador"),
                             "valor": (c.get("ic_primaria") or {}).get("media"),
                             "resuelto": c.get("resuelto")})
        if not ganadores:
            return {"disponible": False, "banco": banco, "n_vecinos": len(vecinos)}

        ganador, veces = Counter(ganadores).most_common(1)[0]
        valores_ganador = [d["valor"] for d in detalles
                           if d["ganador"] == ganador and d["valor"] is not None]
        return {
            "disponible": True,
            "autorreferencial": autorreferencial,
            "banco": banco,
            "n_vecinos": len(detalles),
            "sim_media": round(sum(d["similitud"] for d in detalles) / len(detalles), 3),
            "ganador": ganador,
            "acuerdo": round(veces / len(ganadores), 3),
            "frac_resuelto": round(resueltos / len(detalles), 3),
            "ic_medio": (round(sum(amplitudes) / len(amplitudes), 4)
                         if amplitudes else None),
            "motivo_dominante": (Counter(motivos).most_common(1)[0][0]
                                 if motivos else None),
            "valor_esperado": (round(sum(valores_ganador) / len(valores_ganador), 3)
                               if valores_ganador else None),
            "vecinos": detalles,
        }

    def tasa_acierto(self) -> dict:
        """Cuanto aciertan los priors, por banco. El test del aprendizaje."""
        por_banco: dict[str, list[tuple[bool, bool]]] = {}
        autorref = 0
        for p in self._leer().get("predicciones", []):
            if p.get("acierto") is None:
                continue
            if p.get("autorreferencial"):
                # predecir el pasado del PROPIO documento no es aprendizaje:
                # fuera de la tasa (se cuentan aparte, por transparencia)
                autorref += 1
                continue
            # la familia se DERIVA de esperado/real: retroactivo, sin migracion
            fam = _familia(p.get("esperado")) == _familia(p.get("real"))
            por_banco.setdefault(p["banco"], []).append((bool(p["acierto"]), fam))
        out = {b: {"n": len(v),
                   "acierto": round(sum(e for e, _ in v) / len(v), 3),
                   "acierto_familia": round(sum(f for _, f in v) / len(v), 3)}
               for b, v in por_banco.items()}
        if autorref:
            out["_autorreferenciales_excluidas"] = autorref
        return out

    # ------------------------------------------------------------------ #
    # Vista 3: el crudo
    # ------------------------------------------------------------------ #
    def documento_estudiado(self, doc_hash: str | None = None,
                            pdf_path: str | None = None) -> dict | None:
        """
        ¿Este documento ya fue estudiado Y COMPLETADO? Sirve para que DocuAgent no
        vuelva a entrenar con un documento que ya procesó.

        Cuenta como estudiado solo si existe una sesión que lo identifica Y ADEMÁS
        se registró una DECISIÓN final para ese run_id. Una corrida a medias (con
        sesión pero sin decisión, p. ej. un corte de Colab) NO bloquea el
        reproceso: el batch sigue siendo reanudable.

        La identificación es por `doc_hash` (hash de contenido, robusto a la ruta
        y a la máquina); si no hay hash disponible en alguno de los lados, cae a
        comparar el NOMBRE del fichero. Si no hay forma de identificar, devuelve
        None (conservador: ante la duda, no se da por estudiado).

        Devuelve {run_id, pdf_nombre, estudiado_en, stack} de la corrida
        completada más reciente, o None.
        """
        datos = self._leer()
        nombre = Path(pdf_path).name if pdf_path else None
        decididos = {d["run_id"]: d for d in datos["decisiones"]}

        coincidencias = []
        for s in datos["sesiones"]:
            if doc_hash and s.get("doc_hash"):
                igual = s["doc_hash"] == doc_hash
            elif nombre and s.get("pdf_nombre"):
                igual = s["pdf_nombre"] == nombre
            else:
                igual = False
            if igual and s["run_id"] in decididos:
                dec = decididos[s["run_id"]]
                coincidencias.append({
                    "run_id": s["run_id"],
                    "pdf_nombre": s.get("pdf_nombre"),
                    "estudiado_en": dec.get("guardado_en"),
                    "stack": dec.get("stack", {}),
                })
        return coincidencias[-1] if coincidencias else None

    def resumen(self) -> dict[str, Any]:
        d = self._leer()
        return {"sesiones": len(d.get("sesiones", [])),
                "experimentos": len(d.get("experimentos", [])),
                "decisiones": len(d.get("decisiones", [])),
                "predicciones": len(d.get("predicciones", []))}
