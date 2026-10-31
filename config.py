"""
Configuración central del Evaluador de ejercicios.

Un solo lugar con rutas, la definición de cada ejercicio (puntos de
MediaPipe Pose, ángulos clave, umbrales para segmentar repeticiones,
etiquetas de error) e hiperparámetros, para que extraer_caracteristicas.py,
entrenar.py, evaluar.py y tiempo_real.py no se desincronicen.

Ejercicios y ángulos tomados del póster del equipo (lagartijas, sentadillas,
curl de bíceps; vista lateral; puntos 11 hombro, 13 codo, 15 muñeca,
23 cadera, 25 rodilla, 27 tobillo de MediaPipe Pose).
"""

import os
from pathlib import Path

import torch

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent

RUTA_DATOS = RAIZ / "data"
RUTA_VIDEOS_PROPIOS = RUTA_DATOS / "propios"  # <ejercicio>/<etiqueta>/*.mp4, grabados por el equipo
RUTA_CARACTERISTICAS = RUTA_DATOS / "caracteristicas"  # CSV por ejercicio (sí van a Git, son pequeños)

RUTA_SALIDAS = RAIZ / "salidas"
RUTA_MODELOS_MEDIAPIPE = RAIZ / "modelos_mediapipe"  # .task de MediaPipe (no va a Git, se descarga)

# Modelo de MediaPipe Pose Landmarker (Tasks API). "full" es el balance
# precisión/velocidad recomendado por Google para CPU; "lite" es más rápido
# si una laptop del equipo no da abasto en tiempo real.
MEDIAPIPE_MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MEDIAPIPE_MODELO_PATH = RUTA_MODELOS_MEDIAPIPE / "pose_landmarker_full.task"

# =========================================================
# PUNTOS DE MEDIAPIPE POSE (33 puntos) QUE USAMOS
# =========================================================
# Cada nombre -> (índice lado izquierdo, índice lado derecho). El lado que se
# usa en cada frame se elige dinámicamente por visibilidad (quien esté dando
# el frente a la cámara), no está fijo a "izquierdo".

PUNTOS = {
    "hombro": (11, 12),
    "codo": (13, 14),
    "muneca": (15, 16),
    "cadera": (23, 24),
    "rodilla": (25, 26),
    "tobillo": (27, 28),
    "talon": (29, 30),
    "pie": (31, 32),
}

# =========================================================
# DEFINICIÓN DE CADA EJERCICIO
# =========================================================
# angulo_rep: (p1, vertice, p3) usado para segmentar repeticiones (detecta el
#   "valle" de una repetición: flexión -> extensión).
# umbral_superior / umbral_inferior: histéresis en grados para marcar
#   inicio/fin de una repetición. Son valores de referencia biomecánica
#   (no calibrados con los videos del equipo todavía) -- AJUSTAR cuando haya
#   grabaciones reales, ver notas en GUIA_EQUIPO.md.
# angulos_clave: ángulos (aparte del de segmentación) que se calculan por
#   frame y se agregan (min/max/media/rango) como características.
# etiquetas: "correcto" siempre al índice 0; el resto son los errores del
#   póster. Deben coincidir exactamente con los nombres de carpeta en
#   data/propios/<ejercicio>/.

# Nota: la clasificación correcto/error la hace la red entrenada (modelo.py) a
# partir de las características agregadas de cada repetición -- no hay reglas de
# umbral fijo tipo "si el ángulo mínimo es mayor a X, es error Y". Los únicos
# umbrales de este archivo son umbral_superior/umbral_inferior, que solo sirven
# para SEGMENTAR dónde empieza y termina cada repetición.

EJERCICIOS = {
    "lagartijas": {
        "nombre_legible": "Lagartijas",
        "angulo_rep": ("hombro", "codo", "muneca"),
        # Calibrado con las grabaciones reales del equipo (ver diagnóstico en
        # el commit que ajustó estos valores): el ángulo del codo en el punto
        # más bajo varió entre 90° y 149° según la toma -- mucho más alto de
        # lo que asumía el valor de referencia inicial (110°), que dejaba
        # fuera repeticiones válidas y, por diseño, TODAS las de
        # no_bajar_suficiente (su .angulo mínimo nunca iba a bajar tanto).
        "umbral_superior": 160.0,
        "umbral_inferior": 150.0,
        "angulos_clave": {
            "codo": ("hombro", "codo", "muneca"),
            "alineacion_cadera": ("hombro", "cadera", "tobillo"),
        },
        "senales_especiales": ["desviacion_cadera"],
        "etiquetas": ["correcto", "cadera_caida", "cadera_elevada", "no_bajar_suficiente"],
    },
    "sentadillas": {
        "nombre_legible": "Sentadillas",
        "angulo_rep": ("cadera", "rodilla", "tobillo"),
        "umbral_superior": 165.0,
        "umbral_inferior": 120.0,
        "angulos_clave": {
            "rodilla": ("cadera", "rodilla", "tobillo"),
        },
        "senales_especiales": ["inclinacion_tronco", "talon_y_norm"],
        "etiquetas": ["correcto", "no_bajar_suficiente", "inclinarse_demasiado", "talones_levantados"],
        # "rodillas hacia adentro" del póster requiere toma FRONTAL (valgo de rodilla no se
        # ve en vista lateral) y queda fuera de esta primera versión -- ver README, sección
        # "Decisiones y limitaciones".
    },
    "curl_biceps": {
        "nombre_legible": "Curl de bíceps",
        "angulo_rep": ("hombro", "codo", "muneca"),
        "umbral_superior": 150.0,
        "umbral_inferior": 70.0,
        "angulos_clave": {
            "codo": ("hombro", "codo", "muneca"),
        },
        "senales_especiales": ["inclinacion_tronco", "codo_x_norm"],
        "etiquetas": ["correcto", "balancear_torso", "codo_se_adelanta", "no_completar_movimiento"],
    },
}

LISTA_EJERCICIOS = list(EJERCICIOS.keys())

VISIBILIDAD_MINIMA = 0.5  # por debajo de esto, MediaPipe no confía en el punto -> se descarta el frame

# =========================================================
# HIPERPARÁMETROS DEL CLASIFICADOR (un MLP por ejercicio)
# =========================================================

CAPAS_OCULTAS = [32, 16]
DROPOUT = 0.3
BATCH_SIZE = 16
NUM_EPOCAS = 200
TASA_APRENDIZAJE = 1e-3
PACIENCIA_EARLY_STOPPING = 20
RANDOM_STATE = 42

# Reparto de datos: agrupado por PERSONA, no por repetición individual, para
# que las repeticiones de una misma persona no queden a la vez en train y en
# prueba ("fuga" de datos: ejemplos muy parecidos -mismo cuerpo, misma
# cámara, mismo día- en ambos lados inflarían las métricas sin que el
# modelo esté generalizando de verdad).
FRACCION_VALIDACION = 0.15
FRACCION_PRUEBA = 0.15

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TIEMPO REAL (webcam)
# =========================================================

CAMARA_INDICE = 0
VENTANA_SUAVIZADO_ANGULO = 3  # media móvil de N frames sobre el ángulo de segmentación, reduce parpadeo
