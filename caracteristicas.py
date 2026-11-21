"""
Traduce los landmarks de un frame a las señales numéricas de un ejercicio
(ángulos de 3 puntos + señales especiales), y agrega las señales de todos
los frames de una repetición en:
  - un vector de características (min/max/media/rango de cada señal), y
  - la secuencia de cada señal remuestreada a config.PUNTOS_SECUENCIA puntos
    (para modelos que miran la forma temporal de la repetición).

Todas las señales con signo están expresadas respecto a hacia dónde mira la
persona, no respecto a la imagen: grabar mirando a la izquierda o a la
derecha tiene que dar los mismos números.
"""

import numpy as np

import config
import pose_utils


def _puntos_necesarios(cfg):
    puntos = set()
    for p1, p2, p3 in cfg["angulos_clave"].values():
        puntos.update([p1, p2, p3])
    p1, p2, p3 = cfg["angulo_rep"]
    puntos.update([p1, p2, p3])
    for especial in cfg.get("senales_especiales", []):
        if especial == "desviacion_cadera":
            puntos.update(["hombro", "cadera", "tobillo"])
        elif especial == "inclinacion_tronco":
            puntos.update(["hombro", "cadera"])
        elif especial == "elevacion_talon":
            puntos.update(["rodilla", "tobillo", "talon", "pie"])
        elif especial == "codo_x_norm":
            puntos.update(["nariz", "hombro", "codo", "cadera"])
    return puntos


def senales_del_ejercicio(ejercicio):
    """Nombres de las señales por frame que se agregan (sin la de segmentación, que repite un ángulo clave)."""
    cfg = config.EJERCICIOS[ejercicio]
    return list(cfg["angulos_clave"].keys()) + list(cfg.get("senales_especiales", []))


def calcular_senales_frame(landmarks, ejercicio, ancho, alto):
    """
    Regresa (señales, lado, visibilidad_minima) o (None, None, None) si el
    frame no tiene suficiente confianza en los puntos que necesita este
    ejercicio.
    """
    cfg = config.EJERCICIOS[ejercicio]
    puntos_necesarios = _puntos_necesarios(cfg)
    lado = pose_utils.lado_visible(landmarks, puntos_necesarios)
    vis_min = pose_utils.visibilidad_minima(landmarks, puntos_necesarios, lado)
    if vis_min < config.VISIBILIDAD_MINIMA:
        return None, None, None

    def p(nombre):
        return pose_utils.punto_px(landmarks, nombre, lado, ancho, alto)

    senales = {}
    for nombre_angulo, (p1, p2, p3) in cfg["angulos_clave"].items():
        senales[nombre_angulo] = pose_utils.angulo(p(p1), p(p2), p(p3))

    p1n, p2n, p3n = cfg["angulo_rep"]
    senales["_angulo_rep"] = pose_utils.angulo(p(p1n), p(p2n), p(p3n))

    for especial in cfg.get("senales_especiales", []):
        if especial == "desviacion_cadera":
            senales["desviacion_cadera"] = pose_utils.desviacion_perpendicular_normalizada(
                p("cadera"), p("hombro"), p("tobillo")
            )
        elif especial == "inclinacion_tronco":
            senales["inclinacion_tronco"] = pose_utils.angulo_respecto_vertical(p("cadera"), p("hombro"))
        elif especial == "elevacion_talon":
            # Talón por encima de la punta del pie = positivo. Se normaliza con la
            # espinilla (rodilla-tobillo), que no cambia de largo al bajar en la
            # sentadilla (cadera-tobillo sí, y eso hacía que la señal midiera profundidad).
            largo_espinilla = pose_utils.distancia(p("rodilla"), p("tobillo")) or 1.0
            senales["elevacion_talon"] = (p("pie")[1] - p("talon")[1]) / largo_espinilla
        elif especial == "codo_x_norm":
            # Positivo = codo por delante del hombro, sin importar hacia qué lado mira.
            mirando = 1.0 if p("nariz")[0] >= p("hombro")[0] else -1.0
            largo_torso = pose_utils.distancia(p("hombro"), p("cadera")) or 1.0
            senales["codo_x_norm"] = mirando * (p("codo")[0] - p("hombro")[0]) / largo_torso

    return senales, lado, vis_min


def agregar_repeticion(frames_senales, ejercicio):
    """
    frames_senales: lista de dicts (una por frame de la repetición). Regresa un
    dict plano con <señal>_min/_max/_media/_rango y seq_<señal>_<i>.
    """
    resultado = {}
    n = config.PUNTOS_SECUENCIA
    for senal in senales_del_ejercicio(ejercicio):
        valores = np.array([f[senal] for f in frames_senales], dtype=float)
        resultado[f"{senal}_min"] = valores.min()
        resultado[f"{senal}_max"] = valores.max()
        resultado[f"{senal}_media"] = valores.mean()
        resultado[f"{senal}_rango"] = valores.max() - valores.min()
        remuestreada = np.interp(np.linspace(0, len(valores) - 1, n), np.arange(len(valores)), valores)
        for i, v in enumerate(remuestreada):
            resultado[f"seq_{senal}_{i}"] = v
    return resultado


def columnas_caracteristicas(ejercicio):
    """Columnas agregadas (min/max/media/rango), en orden fijo."""
    columnas = []
    for s in senales_del_ejercicio(ejercicio):
        columnas += [f"{s}_min", f"{s}_max", f"{s}_media", f"{s}_rango"]
    return columnas


def columnas_secuencia(ejercicio):
    """Columnas de la secuencia remuestreada, agrupadas por señal (señal-major)."""
    return [f"seq_{s}_{i}" for s in senales_del_ejercicio(ejercicio) for i in range(config.PUNTOS_SECUENCIA)]
