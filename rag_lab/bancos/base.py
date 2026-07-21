"""
Banco base: el esqueleto comun de los cuatro bancos de pruebas.

Cada banco hereda de aqui y solo define lo que cambia (que experimentos probar
y como medirlos). El flujo es siempre el mismo:

    preparar -> experimentar -> rankear -> juzgar

Detalle de diseno importante: el estado de LangGraph solo contiene tipos
serializables (listas y diccionarios con numeros y cadenas). Los objetos
pesados que NO son serializables (el documento cargado, el indice FAISS, el
modelo de embeddings) se guardan como variables de instancia del banco
(_sesion_actual, _contexto_actual) y jamas entran en el estado del grafo.
Esto evita el error "Type is not msgpack serializable: FAISS" que aparece en
versiones recientes de LangGraph aunque no se use checkpointer explicito.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..corpus import Sesion
from ..fabricas import construir_llm_juez
from ..memory import Memoria
from ..muestreo import resamplear_preguntas
from ..estadistica import bootstrap_ic, comparacion_pareada, spearman


def _corr_rangos(xs: list[float], ys: list[float]) -> float | None:
    """Spearman para tendencias de log (n pequeño). Delegado al canónico de
    estadistica.spearman (fase 1); conserva su contrato: None si n<3 o si
    alguna serie es constante (el canónico devuelve NaN en ese caso)."""
    import math
    if len(xs) < 3:
        return None
    r = spearman(xs, ys)
    return None if math.isnan(r) else r


def _finalista_por_metrica_final(resultados: list[dict], finalistas: list[dict],
                                 metrica_primaria: str,
                                 metrica_final: str = "mrr") -> dict | None:
    """
    TENSION PRIMARIA/CORONACION (comprobada empiricamente en test_13): un banco
    puede resolverse limpiamente por SU primaria (p. ej. el splitter por recall:
    ¿parte el troceo la respuesta?) y aun asi congelar una eleccion suboptima
    para la metrica con la que la Fase B CORONA (MRR): recursive_500 gano el
    banco por recall y en la final perdia 0.079 de MRR (concluyente, prob 0.995)
    contra recursive_1000, que la final nunca vio porque el banco iba resuelto.

    Regla: si la primaria del banco NO es la metrica de coronacion, el banco
    aporta a la final un finalista EXTRA: su mejor config por la metrica de
    coronacion (si no esta ya entre los finalistas). La final decide.
    """
    if metrica_primaria == metrica_final or not resultados:
        return None
    con_metrica = [r for r in resultados
                   if "error" not in r and r.get(metrica_final) is not None]
    if not con_metrica:
        return None
    mejor = max(con_metrica, key=lambda r: float(r[metrica_final]))
    etiqueta = (mejor.get("_config") or {}).get("etiqueta") or mejor.get("etiqueta")
    ya = {f.get("etiqueta") for f in (finalistas or [])}
    if not etiqueta or etiqueta in ya:
        return None
    return {"etiqueta": etiqueta, "es_lider": False,
            "valor": float(mejor[metrica_final]),
            "config": dict(mejor.get("_config") or {}),
            "via_metrica_final": True}


def _filtrar_por_restricciones(experimentos: list[dict], restricciones: dict | None,
                               banco: str) -> list[dict]:
    """
    Restringe la parrilla del banco a lo pedido en el plan. Un experimento pasa
    si su etiqueta esta en `etiquetas`, o si su familia (splitter/nombre/config/
    provider, o el prefijo de la etiqueta) esta en `familias`. Si el filtro deja
    la parrilla VACIA (nombre mal escrito, registro desactualizado), se lanza
    ValueError: fallar rapido y sin gastar es mejor que correr en silencio una
    parrilla que el usuario no pidio.
    """
    r = (restricciones or {}).get(banco) or {}
    familias = {str(x) for x in (r.get("familias") or [])}
    etiquetas = {str(x) for x in (r.get("etiquetas") or [])}
    if not familias and not etiquetas:
        return experimentos

    def _pasa(e: dict) -> bool:
        et = str(e.get("etiqueta", ""))
        if etiquetas and et in etiquetas:
            return True
        if familias:
            rasgos = {str(e.get(k)) for k in ("splitter", "nombre", "config", "provider")
                      if e.get(k) is not None}
            if rasgos & familias or any(et.startswith(f) for f in familias):
                return True
        return False

    filtrados = [e for e in experimentos if _pasa(e)]
    if not filtrados:
        # FALLO RÁPIDO, no fallback silencioso. Antes se ignoraba la restricción
        # y se corría la parrilla COMPLETA: gastaba tiempo y API en modelos que
        # el usuario había excluido explícitamente, y el experimento ejecutado
        # ya no era el que el plan describía. La causa típica es una etiqueta
        # mal escrita o un registro desactualizado (zip viejo en Colab sin
        # re-extraer / kernel sin reiniciar).
        disponibles = sorted(str(e.get("etiqueta", "")) for e in experimentos)
        raise ValueError(
            f"[{banco}] La restriccion del plan no casa con ningun experimento "
            f"(familias={sorted(familias)}, etiquetas={sorted(etiquetas)}). "
            f"Etiquetas disponibles en el registro cargado: {disponibles}. "
            f"Si esperabas otras, el codigo en ejecucion es viejo: re-extrae el "
            f"zip y reinicia el kernel. No ejecuto nada para no gastar en una "
            f"parrilla que no pediste.")
    print(f"[{banco}] Restriccion del plan: {len(experimentos)} -> "
          f"{len(filtrados)} experimentos.")
    return filtrados


def extraer_json(texto: str) -> dict | None:
    """
    Intenta sacar un objeto JSON de la respuesta de un LLM.

    Quita cercos de markdown y, si el parseo directo falla, busca el primer
    bloque entre llaves. Devuelve None si no hay forma.
    """
    limpio = re.sub(r"```json|```", "", texto).strip()
    try:
        return json.loads(limpio)
    except json.JSONDecodeError:
        pass
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio != -1 and fin != -1 and fin > inicio:
        try:
            return json.loads(limpio[inicio:fin + 1])
        except json.JSONDecodeError:
            return None
    return None


class EstadoBanco(TypedDict, total=False):
    """
    Estado que circula entre los nodos del banco.

    Solo contiene tipos JSON-seguros: listas de dicts con numeros y cadenas.
    La sesion y el contexto (FAISS, modelo) viven fuera del estado.
    """
    experimentos: list[dict]
    resultados: list[dict]
    ranking_texto: str
    veredicto: dict
    confianza_estadistica: dict


class BancoBase:
    """
    Clase base. No se usa directamente; cada banco hereda de ella.

    Atributos que la subclase DEBE definir:
        nombre          : etiqueta corta del banco ("splitter", "embedding"...).
        clave_decision  : campo del veredicto que identifica al ganador.
        descripcion     : que papel juega este banco.
        esquema_json    : forma del JSON que debe emitir el supervisor.
        metrica_primaria: (nombre, "max"|"min"). La metrica de CALIDAD con la que
                          se decide y se compara. Su IC y su valor se persisten en
                          el veredicto para que el meta-supervisor compare vecinos.
        objetivos_secundarios: [(metrica, "max"|"min"), ...] en ORDEN de prioridad.
                          No son pesos: solo desempatan cuando la primaria empata
                          estadisticamente (coste, latencia, dispersion...).
    """

    nombre: str = "base"
    clave_decision: str = "recomendado"
    descripcion: str = ""
    esquema_json: str = "{}"
    metrica_primaria: tuple[str, str] = ("relevancia_media", "max")
    objetivos_secundarios: list[tuple[str, str]] = []
    # Metricas que se MUESTRAN al supervisor como contexto aunque no decidan.
    metricas_contexto: tuple[str, ...] = ()

    # Mapa metrica agregada -> lista por pregunta (para IC y comparacion pareada).
    _LISTA_POR_PREGUNTA = {
        "relevancia_media": "_relevancias",
        "mrr": "_mrrs",
        "recuperacion_pasaje": "_recuperaciones",
    }

    def __init__(self, memoria: Memoria):
        self.memoria = memoria
        # Objetos pesados no serializables: se asignan en cada ejecucion,
        # nunca en el estado del grafo.
        self._sesion_actual: Sesion | None = None
        self._contexto_actual: dict = {}
        self._run_actual: int = 1  # run en curso; gobierna el anclaje en el juez
        self.grafo = self._construir_grafo()

    # ------------------------------------------------------------------ #
    # Piezas que la subclase personaliza
    # ------------------------------------------------------------------ #
    def preparar_contexto(self, sesion: Sesion) -> dict:
        """Construye lo que se reutiliza entre experimentos (por defecto, nada)."""
        return {}

    def generar_experimentos(self, sesion: Sesion) -> list[dict]:
        """Devuelve la lista de configuraciones a probar. La subclase la implementa."""
        raise NotImplementedError

    def ejecutar_experimento(self, experimento: dict, sesion: Sesion,
                            contexto: dict) -> dict:
        """Ejecuta UNA configuracion y devuelve un dict de metricas."""
        raise NotImplementedError

    def construir_fallback(self, ganador: dict) -> dict:
        """Veredicto de emergencia si el supervisor no devuelve JSON valido."""
        return {self.clave_decision: ganador.get("etiqueta"), "confianza": "baja",
                "justificacion": "El supervisor no devolvio JSON valido; se usa el lider."}

    # ------------------------------------------------------------------ #
    # Los cuatro nodos del grafo
    # ------------------------------------------------------------------ #
    def _nodo_preparar(self, estado: EstadoBanco) -> dict:
        """
        Prepara el contexto y genera los experimentos.

        Guarda sesion y contexto en variables de instancia (fuera del estado
        del grafo) para evitar problemas de serializacion con FAISS.
        """
        sesion = self._sesion_actual
        print(f"[{self.nombre}] Preparando experimentos...")
        self._contexto_actual = self.preparar_contexto(sesion)
        experimentos = self.generar_experimentos(sesion)
        experimentos = _filtrar_por_restricciones(
            experimentos, getattr(sesion, "restricciones", None), self.nombre)
        print(f"[{self.nombre}] {len(experimentos)} experimentos por ejecutar")
        return {"experimentos": experimentos}

    def _nodo_experimentar(self, estado: EstadoBanco) -> dict:
        """
        Ejecuta los experimentos en bucle secuencial.

        Lee sesion y contexto desde variables de instancia, no desde el estado.
        Si un experimento falla lo registra como error y sigue adelante.
        """
        import time as _t
        sesion = self._sesion_actual
        contexto = self._contexto_actual
        resultados = []
        n_total = len(estado["experimentos"])
        for i, exp in enumerate(estado["experimentos"], start=1):
            etiqueta = exp.get("etiqueta", f"exp{i}")
            _t0 = _t.time()
            try:
                resultado = self.ejecutar_experimento(exp, sesion, contexto)
                print(f"[{self.nombre}]   {i}/{n_total} {etiqueta} "
                      f"({_t.time() - _t0:.1f}s)")
                resultado["etiqueta"] = etiqueta
                resultado["_config"] = exp   # config ejecutable (para finalistas/Fase B)
                resultados.append(resultado)
            except Exception as error:  # noqa: BLE001
                print(f"[{self.nombre}] Experimento '{etiqueta}' fallo: {error}")
                resultados.append({"etiqueta": etiqueta, "error": str(error)})
        return {"resultados": resultados}

    def _nodo_rankear(self, estado: EstadoBanco) -> dict:
        """
        Decide sin pesos: ordena por la metrica PRIMARIA, forma el grupo de empate
        ESTADISTICO (rivales cuya desventaja no es concluyente segun la comparacion
        pareada) y desempata por los objetivos SECUNDARIOS. Marca tambien el frente
        de Pareto para que el supervisor vea las opciones racionales.
        """
        validos = [r for r in estado["resultados"] if "error" not in r]
        if not validos:
            return {"ranking_texto": "Sin resultados validos.", "resultados": [],
                    "confianza_estadistica": {"disponible": False}}

        ids_frente = {id(r) for r in self._frente_pareto(validos)}
        for r in validos:
            r["pareto"] = id(r) in ids_frente

        ordenados, ganador, ic, comp, finalistas = self._decidir(validos)
        ganador["es_lider"] = True
        confianza = {
            "disponible": bool(ic),
            "lider": ganador.get("etiqueta"),
            "metrica_primaria": self.metrica_primaria[0],
            "ic_primaria": ic,
            "comparacion": comp,
            "finalistas": finalistas,
            "n_finalistas": len(finalistas),
            "n_frente": len(ids_frente),
        }
        ranking_texto = self._formatear_ranking(ordenados, confianza)
        for linea in self._tendencias([r for r in estado.get("resultados", [])
                                       if "error" not in r], ganador=ganador):
            print(f"[{self.nombre}] tendencia: {linea}")
        mp = self.metrica_primaria[0]
        print(f"[{self.nombre}] Lider: {ganador.get('etiqueta')} "
              f"({mp}={ganador.get(mp, 0):.3f}) | {len(ids_frente)} en Pareto")
        return {"resultados": ordenados, "ranking_texto": ranking_texto,
                "confianza_estadistica": confianza}

    # ------------------------------------------------------------------ #
    # Decision sin pesos: primaria + empate pareado + secundarios
    # ------------------------------------------------------------------ #
    def _clave_orden(self, r: dict, metrica: str, sentido: str) -> float:
        """Clave ASCENDENTE donde el primero es el mejor (segun el sentido)."""
        valor = r.get(metrica, 0.0)
        return -valor if sentido == "max" else valor

    def _serie_primaria(self, r: dict) -> list | None:
        """Valores por pregunta de la metrica primaria (para IC y comparacion)."""
        clave = self._LISTA_POR_PREGUNTA.get(self.metrica_primaria[0])
        return r.get(clave) if clave else None

    def _domina(self, a: dict, b: dict) -> bool:
        """a domina a b: a >= b en TODOS los objetivos y > en al menos uno."""
        objetivos = [self.metrica_primaria, *self.objetivos_secundarios]
        mejor_en_alguno = False
        for metrica, sentido in objetivos:
            va, vb = a.get(metrica, 0.0), b.get(metrica, 0.0)
            if sentido == "max":
                if va < vb:
                    return False
                if va > vb:
                    mejor_en_alguno = True
            else:
                if va > vb:
                    return False
                if va < vb:
                    mejor_en_alguno = True
        return mejor_en_alguno

    def _frente_pareto(self, validos: list[dict]) -> list[dict]:
        """Configuraciones no dominadas por ninguna otra."""
        return [r for r in validos
                if not any(self._domina(o, r) for o in validos if o is not r)]

    def _decidir(self, validos: list[dict]) -> tuple[list[dict], dict, dict | None, dict | None, list[dict]]:
        """
        Devuelve (ordenados, ganador, ic_primaria_ganador, comparacion_vs_rival,
        finalistas).

        1) Ordena por la primaria (estimacion puntual) -> lider provisional L0.
        2) Empate estadistico: cada rival cuya desventaja frente a L0 en la
           primaria NO sea `concluyente` (comparacion pareada) entra en el grupo.
        3) Dentro del grupo, gana el mejor en los secundarios (en orden).
        4) IC del ganador, comparacion pareada ganador vs primer rival fuera del
           empate, y el GRUPO DE FINALISTAS completo (todos los empatados con su
           IC), que alimenta la carrera factorial de la Fase B.
        """
        mp, sp = self.metrica_primaria
        sem = f"{getattr(self._sesion_actual, 'doc_type', '?')}|{self.nombre}"

        por_primaria = sorted(validos, key=lambda r: self._clave_orden(r, mp, sp))
        lider0 = por_primaria[0]
        serie0 = self._serie_primaria(lider0)

        empate = [lider0]
        for r in por_primaria[1:]:
            serie_r = self._serie_primaria(r)
            if serie0 and serie_r:
                comp = comparacion_pareada(serie0, serie_r,
                                           semilla=f"{sem}|{r.get('etiqueta')}")
                if not comp.get("concluyente"):
                    empate.append(r)
            # Sin series por pregunta no podemos afirmar empate: el rival queda
            # fuera (decision conservadora: el lider de la primaria manda).

        empate.sort(key=lambda r: tuple(
            self._clave_orden(r, m, s) for m, s in self.objetivos_secundarios) or (0.0,))
        ganador = empate[0]
        if len(empate) > 1 and ganador is not lider0:
            # El desempate por secundarias cambio al ganador: DECIRLO. Un
            # veredicto que contradice a la primaria sin explicacion siembra
            # desconfianza (y con razon).
            print(f"[{self.nombre}] Empate estadistico de {len(empate)}: lider "
                  f"por primaria = {lider0.get('etiqueta')} "
                  f"({mp}={lider0.get(mp)}); el desempate por secundarias "
                  f"({', '.join(m for m, _ in self.objetivos_secundarios[:2])}) "
                  f"elige a {ganador.get('etiqueta')}.")

        ids_empate = {id(r) for r in empate}
        resto = [r for r in por_primaria if id(r) not in ids_empate]
        ordenados = empate + resto

        serie_g = self._serie_primaria(ganador)
        ic = bootstrap_ic(serie_g, semilla=f"{sem}|primaria") if serie_g else None

        # GRUPO DE EMPATADOS COMPLETO (no solo el lider y un rival): cada finalista
        # con su valor primario y su IC. Es lo que la Fase B (carrera factorial)
        # necesita para cruzar SOLO los empatados de cada banco. Si el grupo tiene
        # un unico miembro, el banco salio resuelto y no hay carrera.
        finalistas = []
        for r in empate:
            s = self._serie_primaria(r)
            ic_r = bootstrap_ic(s, semilla=f"{sem}|fin|{r.get('etiqueta')}") if s else None
            finalistas.append({
                "etiqueta": r.get("etiqueta"),
                "config": r.get("_config"),
                "valor": r.get(mp),
                "ic_bajo": ic_r["ic_bajo"] if ic_r else None,
                "ic_alto": ic_r["ic_alto"] if ic_r else None,
                "amplitud": ic_r["amplitud"] if ic_r else None,
                "n": ic_r["n"] if ic_r else None,
                "es_lider": r is ganador,
                "pareto": bool(r.get("pareto")),
            })

        # COTA (de momento 3): con pocas preguntas el IC es ancho y "empatan"
        # casi todas; nos quedamos con el lider + los 2 mejores por valor primario
        # para que la Fase B no explote ni corone sobre ruido. Ordena lider primero.
        finalistas.sort(key=lambda f: (not f["es_lider"], -(f["valor"] or 0.0)))
        finalistas = finalistas[:3]
        rival = resto[0] if resto else (empate[1] if len(empate) > 1 else None)
        comp_rival = None
        if rival is not None and serie_g and self._serie_primaria(rival):
            comp_rival = comparacion_pareada(serie_g, self._serie_primaria(rival),
                                             semilla=f"{sem}|rival")
            comp_rival["rival"] = rival.get("etiqueta")
        return ordenados, ganador, ic, comp_rival, finalistas

    def _nodo_juzgar(self, estado: EstadoBanco) -> dict:
        """El supervisor LLM lee el ranking y emite su veredicto."""
        sesion = self._sesion_actual
        resultados = estado["resultados"]
        if not resultados:
            return {"veredicto": {"estado": "FALLIDO", "motivo": "Sin resultados validos."}}

        ganador = resultados[0]
        # Anclaje: solo en el run 1 pasamos el veredicto previo como contexto
        # (memoria ENTRE sesiones, util para acumular experiencia). En los runs de
        # estabilidad (2..n) lo suprimimos: si no, el supervisor tenderia a repetir
        # su propio veredicto del run 1 e inflaria el acuerdo de forma artificial.
        # Asi cada sonda de estabilidad es un juicio INDEPENDIENTE del ranking.
        if self._run_actual <= 1:
            pr = {}
            try:
                pr = self.memoria.priors(getattr(sesion, "perfil", None) or {},
                                         self.nombre, self.clave_decision,
                                         excluir_run_id=sesion.run_id)
            except Exception:  # noqa: BLE001
                pr = {}
            contexto_previo = (
                f"Prior por perfil (documentos parecidos; pista, no certeza): "
                f"tipico={pr.get('ganador')} (acuerdo {pr.get('acuerdo')}, "
                f"{pr.get('n_vecinos')} vecinos, valor esperado~{pr.get('valor_esperado')})."
                if pr.get("disponible")
                else "Sin documentos parecidos en memoria: juzga solo por el ranking."
            )
        else:
            contexto_previo = (
                "Evaluacion independiente (sonda de estabilidad sobre un remuestreo "
                "de las preguntas): juzga solo por el ranking de abajo, sin "
                "condicionarte por veredictos anteriores."
            )

        system = (
            f"Eres el supervisor del banco de {self.descripcion} "
            f"para documentos tipo '{sesion.doc_type}'.\n"
            f"{contexto_previo}\n\n"
            f"Total experimentos validos: {len(resultados)}.\n\n"
            "Analiza el ranking y responde UNICAMENTE con JSON puro "
            f"(sin markdown) con esta forma:\n{self.esquema_json}"
        )
        human = (f"Ranking:\n{estado['ranking_texto']}\n\nEmite tu veredicto.")

        respuesta = construir_llm_juez().invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        veredicto = extraer_json(respuesta.content) or self.construir_fallback(ganador)

        # VEREDICTO DETERMINISTA: la clave de decision ES la etiqueta evaluada del
        # lider (verdad estadistica), no texto libre del juez. El juez conserva su
        # papel en confianza/justificacion/alertas; si discrepo del lider, se
        # guarda como `eleccion_juez` (informativa). Esto elimina el desajuste
        # historico "Lider: X / Veredicto: Y" y hace que el acuerdo entre runs se
        # mida a nivel de configuracion real (etiqueta), igual que los finalistas.
        eleccion_juez = veredicto.get(self.clave_decision)
        etiqueta_lider = ganador.get("etiqueta")
        if etiqueta_lider:
            if eleccion_juez and str(eleccion_juez) != str(etiqueta_lider):
                veredicto["eleccion_juez"] = eleccion_juez
            veredicto[self.clave_decision] = etiqueta_lider

        # El supervisor ya leyó la confianza en el ranking; la adjuntamos también
        # al veredicto para que persista en memoria. `_primaria` (valor + IC de la
        # métrica de calidad del banco) es lo que el meta-supervisor usará para
        # comparar vecinos por su rendimiento, no solo por qué config ganó.
        conf = estado.get("confianza_estadistica")
        if conf and conf.get("disponible"):
            ic = conf["ic_primaria"]
            veredicto["_primaria"] = {
                "nombre": conf["metrica_primaria"],
                "valor": ic["media"],
                "ic_bajo": ic["ic_bajo"],
                "ic_alto": ic["ic_alto"],
                "n": ic["n"],
            }
            if conf.get("comparacion"):
                veredicto["_comparacion_lider"] = conf["comparacion"]
            # Grupo de empatados completo (>1 => el banco no se resolvio y sus
            # finalistas competiran en la Fase B factorial).
            if conf.get("finalistas"):
                finalistas_fb = list(conf["finalistas"])
                extra = _finalista_por_metrica_final(
                    estado.get("resultados", []), finalistas_fb,
                    self.metrica_primaria[0])
                if extra is not None:
                    finalistas_fb.append(extra)
                    print(f"[{self.nombre}] Finalista EXTRA para la final: "
                          f"'{extra['etiqueta']}' (mejor por la metrica de "
                          f"coronacion, mrr={extra['valor']:.4f}; la primaria "
                          f"de este banco es {self.metrica_primaria[0]}).")
                veredicto["_finalistas"] = finalistas_fb

        # RANKING COMPACTO {etiqueta: valor_primaria} de TODOS los experimentos
        # validos del run. Sin esto, la memoria solo conservaba al lider y a los
        # finalistas: cualquier analisis transversal (observatorio, DocuAgent)
        # era ciego al rendimiento del resto de la parrilla. Coste: ~20 numeros
        # por experimento; beneficio: EDA y duelos pareados entre CUALQUIER par
        # de configs a posteriori, sin re-ejecutar nada.
        metrica = self.metrica_primaria[0]
        veredicto["_ranking"] = {
            str(r.get("etiqueta")): r.get(metrica)
            for r in estado.get("resultados", [])
            if "error" not in r and r.get(metrica) is not None}
        # METRICAS POR CONFIG (secundarias + contexto): sin esto, la memoria
        # solo conservaba la primaria y cualquier analisis transversal de una
        # secundaria (p.ej. ¿correlaciona rango_similitud con mrr?) era
        # imposible a posteriori. Coste: un dict pequeño por experimento.
        nombres = [m for m, _ in self.objetivos_secundarios]
        nombres += [m for m in self.metricas_contexto if m not in nombres]
        veredicto["_metricas"] = {
            str(r.get("etiqueta")): {m: r.get(m) for m in nombres
                                     if r.get(m) is not None}
            for r in estado.get("resultados", []) if "error" not in r}

        self.memoria.registrar_experimento(
            sesion.run_id, self.nombre, veredicto, run=self._run_actual)
        print(f"[{self.nombre}] Veredicto: {veredicto.get(self.clave_decision)} "
              f"(confianza={veredicto.get('confianza')})")
        return {"veredicto": veredicto}

    # ------------------------------------------------------------------ #
    # Montaje y ejecucion del grafo
    # ------------------------------------------------------------------ #
    def _construir_grafo(self):
        """Compila el grafo. Sin checkpointer: el estado es efimero e interno."""
        c = StateGraph(EstadoBanco)
        c.add_node("preparar", self._nodo_preparar)
        c.add_node("experimentar", self._nodo_experimentar)
        c.add_node("rankear", self._nodo_rankear)
        c.add_node("juzgar", self._nodo_juzgar)
        c.add_edge(START, "preparar")
        c.add_edge("preparar", "experimentar")
        c.add_edge("experimentar", "rankear")
        c.add_edge("rankear", "juzgar")
        c.add_edge("juzgar", END)
        return c.compile()

    def ejecutar(self, sesion: Sesion, run: int = 1, n_runs: int = 1,
                 usar_set_completo: bool = False) -> dict:
        """
        Punto de entrada: asigna la sesion a la variable de instancia y corre el
        grafo. Devuelve el veredicto.

        Estabilidad: si hay mas de un run, los runs 2..n evaluan sobre un
        remuestreo bootstrap de las preguntas de test (ver muestreo.py). Asi el
        "acuerdo entre runs" mide robustez real y no el determinismo del juez. El
        run 1 usa el conjunto completo. Trabajamos sobre una COPIA de la sesion
        para no alterar las preguntas originales (de las que depende la herencia
        entre bancos y la memoria).

        `usar_set_completo=True` fuerza el set COMPLETO sin remuestrear, aunque sea
        un run > 1: lo usa la palanca de preguntas (Fase A.2) para reevaluar sobre
        el set AMPLIADO, de modo que ese run (con mas preguntas) se vuelva el
        canonico y estreche el IC que lee `certeza`.
        """
        if (not usar_set_completo and n_runs > 1 and run > 1
                and len(sesion.preguntas_test) >= 2):
            sesion = copy.copy(sesion)
            sesion.preguntas_test = resamplear_preguntas(
                sesion.preguntas_test, sesion.doc_type, run)
        self._run_actual = run
        self._sesion_actual = sesion
        salida = self.grafo.invoke({})
        return salida.get("veredicto", {})

    # ------------------------------------------------------------------ #
    # Presentacion del ranking
    # ------------------------------------------------------------------ #
    def _tendencias(self, resultados: list[dict],
                    ganador: dict | None = None) -> list[str]:
        """Lineas de tendencia para el log (sobreescribible por banco): como
        evolucionan las metricas con los parametros de la parrilla, y avisos
        de BORDE (si el optimo esta en el extremo, la parrilla esta corta).
        `ganador` es el CORONADO por el veredicto (tras desempates): los
        chequeos de borde deben seguirle a el, no al lider bruto por primaria
        (cuando el desempate voltea, es el coronado quien va a la final)."""
        return []

    def _formatear_ranking(self, ordenados: list[dict], confianza: dict | None = None) -> str:
        mp, sp = self.metrica_primaria
        columnas = [mp] + [m for m, _ in self.objetivos_secundarios]
        columnas += [m for m in self.metricas_contexto if m not in columnas]
        objetivos_txt = " > ".join([f"{mp}({sp})"]
                                   + [f"{m}({s})" for m, s in self.objetivos_secundarios])
        lineas = [
            f"=== DECISION: primaria '{mp}' + secundarios (sin pesos inventados) ===",
            f"Prioridad: {objetivos_txt}",
            "[P] = frente de Pareto (no dominada). El lider se decide por la primaria; "
            "los rivales sin diferencia pareada CONCLUYENTE cuentan como empate y se "
            "desempatan por los secundarios.",
            "",
        ]
        for i, r in enumerate(ordenados, start=1):
            marca = "P" if r.get("pareto") else " "
            metricas = " | ".join(f"{m}={r.get(m, 0):.3f}" for m in columnas)
            lineas.append(f"{i:2}. [{marca}] {r['etiqueta']:22} | {metricas}")

        if confianza and confianza.get("disponible"):
            ic = confianza["ic_primaria"]
            lineas.append("\n=== CONFIANZA ESTADISTICA (bootstrap, sin coste) ===")
            lineas.append(
                f"Lider '{confianza['lider']}': {mp} medio {ic['media']:.3f} "
                f"(IC95 {ic['ic_bajo']:.3f}-{ic['ic_alto']:.3f}, n={ic['n']}). "
                f"IC ancho => media poco fiable; conviene mas evidencia."
            )
            comp = confianza.get("comparacion")
            if comp:
                if comp.get("concluyente"):
                    lineas.append(
                        f"La ventaja del lider sobre '{comp['rival']}' es CONCLUYENTE "
                        f"(diferencia {comp['dif_media']:+.3f}, "
                        f"IC95 {comp['ic_bajo']:+.3f} a {comp['ic_alto']:+.3f}, no cruza 0)."
                    )
                else:
                    lineas.append(
                        f"La ventaja del lider sobre '{comp['rival']}' NO es concluyente "
                        f"(diferencia {comp['dif_media']:+.3f}, "
                        f"IC95 {comp['ic_bajo']:+.3f} a {comp['ic_alto']:+.3f} cruza 0; "
                        f"P(lider mejor)={comp['prob_a_mejor']}). Empate estadistico: "
                        f"si el rival es mas barato o simple, preferirlo es razonable."
                    )
        return "\n".join(lineas)
