"""
CACHE POR DOCUMENTO — compartido por los bancos.

Los chunks de una configuracion dada son IDENTICOS entre runs (el documento no
cambia; los runs solo remuestrean preguntas), asi que embeber y montar indices
se paga UNA vez por documento, no una por run. Ademas es estadisticamente mas
limpio: al compartir indice, la variacion entre runs procede solo del
remuestreo de preguntas (el eje disenado), no del ruido de re-embeber.

El cache se vacia automaticamente al cambiar de documento (run_id) para no
crecer durante el batch. Cada banco usa su propio `espacio` para no pisarse.
"""
import hashlib

_CACHES: dict = {}


def por_documento(espacio: str, run_id: str, clave, fabricar):
    """Devuelve el valor cacheado para (espacio, documento, clave); si no
    existe, lo fabrica una vez. Cambiar de documento vacia el espacio."""
    c = _CACHES.setdefault(espacio, {"run_id": None, "valores": {}})
    if c["run_id"] != run_id:
        c["run_id"] = run_id
        c["valores"] = {}
    if clave not in c["valores"]:
        c["valores"][clave] = fabricar()
    return c["valores"][clave]


def huella_troceado(chunks) -> str:
    """Huella del troceado: dos configs con la misma huella producen chunks
    IDENTICOS (mismo experimento / mismo indice)."""
    h = hashlib.md5()
    h.update(str(len(chunks)).encode())
    for c in chunks:
        h.update(hashlib.md5((getattr(c, "page_content", "") or "").encode()).digest())
    return h.hexdigest()
