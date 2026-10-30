"""
Traduce los landmarks de un frame a las señales numéricas de un ejercicio
(ángulos de 3 puntos + señales especiales), y agrega las señales de todos
los frames de una repetición en un solo vector de características para el
clasificador (min/max/media/rango de cada señal).
"""

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
        elif especial == "talon_y_norm":
            puntos.update(["cadera", "tobillo", "talon"])
        elif especial == "codo_x_norm":
            puntos.update(["hombro", "codo", "cadera"])
    return puntos


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
        elif especial == "talon_y_norm":
            cadera_px, tobillo_px, talon_px = p("cadera"), p("tobillo"), p("talon")
            largo_pierna = ((cadera_px[0] - tobillo_px[0]) ** 2 + (cadera_px[1] - tobillo_px[1]) ** 2) ** 0.5
            senales["talon_y_norm"] = talon_px[1] / largo_pierna if largo_pierna else 0.0
        elif especial == "codo_x_norm":
            senales["codo_x_norm"] = pose_utils.desplazamiento_horizontal_normalizado(
                p("codo"),
                p("hombro"),
                abs(p("cadera")[1] - p("hombro")[1]) or 1.0,
            )

    return senales, lado, vis_min


def agregar_repeticion(frames_senales):
    """
    frames_senales: lista de dicts (una por frame de la repetición), todas
    con las mismas claves. Regresa un solo dict plano:
    {<señal>_min, <señal>_max, <señal>_media, <señal>_rango, duracion_frames}.

    talon_y_norm es un caso especial: lo que importa es el ascenso del talón
    DURANTE la repetición, no su altura absoluta (depende de la cámara), así
    que aquí se resta la línea base (talón al inicio de la repetición).
    """
    claves = frames_senales[0].keys()
    caracteristicas = {"duracion_frames": len(frames_senales)}

    for clave in claves:
        valores = [f[clave] for f in frames_senales]
        if clave == "talon_y_norm":
            base = valores[0]
            valores = [base - v for v in valores]  # positivo = talón subió respecto al inicio
            clave = "despegue_talon"
        caracteristicas[f"{clave}_min"] = min(valores)
        caracteristicas[f"{clave}_max"] = max(valores)
        caracteristicas[f"{clave}_media"] = sum(valores) / len(valores)
        caracteristicas[f"{clave}_rango"] = max(valores) - min(valores)

    return caracteristicas


def columnas_caracteristicas(ejercicio):
    """Nombres de columnas numéricas que produce agregar_repeticion para este ejercicio,
    en un orden fijo (necesario para que entrenar.py y tiempo_real.py usen el mismo orden)."""
    cfg = config.EJERCICIOS[ejercicio]
    senales = list(cfg["angulos_clave"].keys()) + ["_angulo_rep"]
    for especial in cfg.get("senales_especiales", []):
        senales.append("despegue_talon" if especial == "talon_y_norm" else especial)
    columnas = ["duracion_frames"]
    for s in senales:
        columnas += [f"{s}_min", f"{s}_max", f"{s}_media", f"{s}_rango"]
    return columnas
