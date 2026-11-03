"""
Segmentación de repeticiones a partir de un ángulo clave (codo o rodilla
según el ejercicio), por CAÍDA RELATIVA, no por umbral absoluto en grados.

Las tres repeticiones del póster comparten la misma forma: el ángulo
empieza "arriba" (extendido), baja a un "valle" (flexión) y vuelve a subir.
Un umbral absoluto (ej. "repetición = el codo bajó de 110°") no funciona
bien entre personas distintas: el ángulo mínimo real de una repetición
depende del brazo/cuerpo de cada quien y de qué tan marcado hagan el error
-- con dos personas de datos reales ya se vio que el mismo umbral absoluto
dejaba fuera una fracción grande de repeticiones válidas de una persona
mientras funcionaba bien para otra (sobre todo en las etiquetas de "no bajó
lo suficiente", que por definición son las más cercanas al punto de
partida). Por eso aquí se mide CAÍDA RELATIVA desde el punto más alto
reciente: una repetición es "bajó al menos `caida_minima_grados` desde el
último pico" seguido de "volvió a subir esos mismos grados desde el
valle" -- sin importar en qué ángulo absoluto haya pasado eso. La
profundidad real (qué tanto bajó) sigue siendo una característica para el
clasificador (vía caracteristicas.agregar_repeticion), no algo que decida
el segmentador.
"""

import config


class SegmentadorRepeticiones:
    def __init__(self, caida_minima_grados):
        self.caida_minima_grados = caida_minima_grados
        self._estado = "arriba"  # arriba -> bajando -> arriba (repetición completa)
        self._pico = None
        self._valle = None
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

        if self._estado == "arriba":
            self._pico = valor_suave if self._pico is None else max(self._pico, valor_suave)
            if self._pico - valor_suave >= self.caida_minima_grados:
                self._estado = "bajando"
                self._valle = valor_suave

        elif self._estado == "bajando":
            self._valle = min(self._valle, valor_suave)
            if valor_suave - self._valle >= self.caida_minima_grados:
                self._estado = "arriba"
                self._pico = valor_suave
                rep_completa = self._frames_actuales
                self._frames_actuales = [datos_frame]  # el frame actual también abre la siguiente rep
                return rep_completa

        return None
