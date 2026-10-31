"""
Wrapper de MediaPipe Pose (Tasks API) y geometría sobre los landmarks:
ángulos entre tres puntos, elección del lado visible a la cámara y
desviaciones con signo (para distinguir, por ejemplo, cadera caída de
cadera elevada, que con un ángulo sin signo se ven igual).
"""

import math
import urllib.request

import config


def descargar_modelo_mediapipe():
    """Descarga pose_landmarker_full.task si no está ya en disco."""
    if config.MEDIAPIPE_MODELO_PATH.exists():
        return
    config.RUTA_MODELOS_MEDIAPIPE.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo de MediaPipe Pose a {config.MEDIAPIPE_MODELO_PATH} ...")
    urllib.request.urlretrieve(config.MEDIAPIPE_MODELO_URL, config.MEDIAPIPE_MODELO_PATH)
    print("Listo.")


class DetectorPose:
    """
    Detecta los 33 landmarks de MediaPipe Pose en modo VIDEO (frame a frame,
    con timestamps). El modo VIDEO de MediaPipe exige timestamps
    estrictamente crecientes durante toda la vida del detector -- por eso
    hay que llamar a reiniciar() antes de procesar un video nuevo (cada
    video vuelve a contar sus propios timestamps desde 0).
    """

    def __init__(self):
        # Import diferido: mediapipe es pesado de importar y solo lo necesitan
        # los scripts que de verdad procesan video/cámara.
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        descargar_modelo_mediapipe()

        self._mp = mp
        self._vision = vision
        self._opciones = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MEDIAPIPE_MODELO_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = self._vision.PoseLandmarker.create_from_options(self._opciones)

    def reiniciar(self):
        """Hay que llamarlo antes de cada video nuevo (ver nota de la clase)."""
        self._landmarker.close()
        self._landmarker = self._vision.PoseLandmarker.create_from_options(self._opciones)

    def detectar(self, frame_bgr, timestamp_ms: int):
        """Regresa la lista de 33 landmarks (x, y, z, visibility) o None si no detectó a nadie."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen_mp = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = self._landmarker.detect_for_video(imagen_mp, timestamp_ms)
        if not resultado.pose_landmarks:
            return None
        return resultado.pose_landmarks[0]

    def cerrar(self):
        self._landmarker.close()


def lado_visible(landmarks, nombres_puntos):
    """
    Decide si usar el lado izquierdo o derecho: el que tenga mayor visibilidad
    promedio en los puntos relevantes de este ejercicio (el lado que de verdad
    está dando el frente/perfil a la cámara).
    """
    vis_izq = sum(landmarks[config.PUNTOS[n][0]].visibility for n in nombres_puntos) / len(nombres_puntos)
    vis_der = sum(landmarks[config.PUNTOS[n][1]].visibility for n in nombres_puntos) / len(nombres_puntos)
    return "izq" if vis_izq >= vis_der else "der"


def visibilidad_minima(landmarks, nombres_puntos, lado):
    idx_lado = 0 if lado == "izq" else 1
    return min(landmarks[config.PUNTOS[n][idx_lado]].visibility for n in nombres_puntos)


def punto_px(landmarks, nombre_punto, lado, ancho, alto):
    """Coordenada en píxeles (x, y) de un punto. Hay que desnormalizar con
    ancho/alto reales: los landmarks de MediaPipe vienen en [0,1] relativos a
    cada eje, y la imagen casi nunca es cuadrada, así que usar directamente
    los valores normalizados distorsiona los ángulos."""
    idx_lado = 0 if lado == "izq" else 1
    lm = landmarks[config.PUNTOS[nombre_punto][idx_lado]]
    return (lm.x * ancho, lm.y * alto)


def angulo(a, b, c):
    """Ángulo en grados en el vértice b, formado por los segmentos b->a y b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    norma1 = math.hypot(*v1)
    norma2 = math.hypot(*v2)
    if norma1 == 0 or norma2 == 0:
        return 0.0
    coseno = (v1[0] * v2[0] + v1[1] * v2[1]) / (norma1 * norma2)
    coseno = max(-1.0, min(1.0, coseno))
    return math.degrees(math.acos(coseno))

def angulo_respecto_vertical(a, b):
    """Ángulo entre el segmento a->b y la vertical (0° = segmento vertical)."""
    v = (b[0] - a[0], b[1] - a[1])
    norma = math.hypot(*v)
    if norma == 0:
        return 0.0
    coseno = abs(v[1]) / norma  # componente vertical / norma
    coseno = max(-1.0, min(1.0, coseno))
    return math.degrees(math.acos(coseno))


def desviacion_perpendicular_normalizada(p, a, b):
    """
    Distancia con signo de p a la línea a-b, normalizada por la longitud a-b.
    Signo positivo = p queda del lado "arriba" de la línea (y menor, imagen),
    signo negativo = "abajo". Se usa para cadera caída/elevada en lagartijas:
    la sola distancia sin signo no distingue dirección.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    largo = math.hypot(bx - ax, by - ay)
    if largo == 0:
        return 0.0
    # componente z del producto cruz 2D (b-a) x (p-a), en el sistema de imagen
    # (y crece hacia abajo) un cruce positivo significa p por encima de la línea.
    cruz = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return -(cruz / largo) / largo


def desplazamiento_horizontal_normalizado(p_actual, p_referencia, largo_normalizacion):
    """Desplazamiento en x de p_actual respecto a p_referencia, normalizado."""
    if largo_normalizacion == 0:
        return 0.0
    return (p_actual[0] - p_referencia[0]) / largo_normalizacion
