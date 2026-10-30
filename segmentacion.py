"""
Segmentación de repeticiones a partir de un ángulo clave (codo o rodilla
según el ejercicio), por histéresis de dos umbrales.

Las tres repeticiones del póster comparten la misma forma: el ángulo
empieza "arriba" (extendido), baja a un "valle" (flexión) y vuelve a subir.
Una repetición completa es el tramo entre dos pasos por arriba con un valle
de por medio -- así se evita contar como repetición un simple temblor del
ángulo cerca del umbral (se exige cruzar el umbral inferior primero).
"""

import config


class SegmentadorRepeticiones:
    def __init__(self, umbral_superior, umbral_inferior):
        self.umbral_superior = umbral_superior
        self.umbral_inferior = umbral_inferior
        self._estado = "arriba"  # arriba -> bajando -> arriba (repetición completa)
        self._frames_actuales = []
        self._historial_suavizado = []

    def _suavizar(self, valor_angulo):
        self._historial_suavizado.append(valor_angulo)
        self._historial_suavizado = self._historial_suavizado[-config.VENTANA_SUAVIZADO_ANGULO:]
        return sum(self._historial_suavizado) / len(self._historial_suavizado)

    def agregar_frame(self, valor_angulo_segmentacion, datos_frame):
        """
        datos_frame: dict con todas las señales de ese frame (los ángulos
        clave, desviaciones, etc. que calculó caracteristicas.py).
        Regresa la lista de frames de una repetición cuando se completa una,
        o None si todavía está en curso.
        """
        valor_suave = self._suavizar(valor_angulo_segmentacion)
        self._frames_actuales.append(datos_frame)

        if self._estado == "arriba" and valor_suave < self.umbral_inferior:
            self._estado = "bajando"

        elif self._estado == "bajando" and valor_suave > self.umbral_superior:
            self._estado = "arriba"
            rep_completa = self._frames_actuales
            self._frames_actuales = [datos_frame]  # el frame actual también abre la siguiente rep
            return rep_completa

        return None
