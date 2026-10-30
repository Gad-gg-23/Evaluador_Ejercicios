"""
Evaluador de ejercicios en vivo: abre la webcam, detecta la pose con
MediaPipe, segmenta repeticiones por el ángulo clave del ejercicio elegido
y, al completar cada repetición, la clasifica con el modelo entrenado
(correcto / error específico) y lo muestra en pantalla.

Uso:
    python tiempo_real.py --ejercicio sentadillas
    python tiempo_real.py --ejercicio lagartijas --camara 1

Controles: 'q' para salir.
"""

import argparse
import time

import cv2
import pandas as pd
import torch

import caracteristicas
import config
import modelo
import pose_utils
import segmentacion

CONEXIONES_ESQUELETO = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

COLOR_OK = (0, 200, 0)
COLOR_ERROR = (0, 0, 230)
COLOR_ESQUELETO = (255, 200, 0)


def cargar_modelo(ejercicio):
    ruta_modelo = config.RUTA_SALIDAS / f"modelo_{ejercicio}.pt"
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe {ruta_modelo}. Corre primero extraer_caracteristicas.py y entrenar.py "
            f"para '{ejercicio}'."
        )
    paquete = torch.load(ruta_modelo, map_location=config.DISPOSITIVO, weights_only=False)
    red = modelo.ClasificadorRepeticion(len(paquete["columnas"]), len(paquete["etiquetas"])).to(config.DISPOSITIVO)
    red.load_state_dict(paquete["state_dict"])
    red.eval()
    return red, paquete


def clasificar_repeticion(red, paquete, frames_rep):
    columnas, etiquetas = paquete["columnas"], paquete["etiquetas"]
    agregadas = caracteristicas.agregar_repeticion(frames_rep)
    fila = pd.Series(agregadas).reindex(columnas).fillna(0.0)
    media = pd.Series(paquete["media"], index=columnas)
    desviacion = pd.Series(paquete["desviacion"], index=columnas)
    x = torch.tensor(((fila - media) / desviacion).values.astype("float32")).unsqueeze(0)
    with torch.no_grad():
        logits = red(x.to(config.DISPOSITIVO))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = int(probs.argmax())
    return etiquetas[idx], float(probs[idx])


def dibujar_esqueleto(frame, landmarks, ancho, alto):
    for a, b in CONEXIONES_ESQUELETO:
        pa = (int(landmarks[a].x * ancho), int(landmarks[a].y * alto))
        pb = (int(landmarks[b].x * ancho), int(landmarks[b].y * alto))
        cv2.line(frame, pa, pb, COLOR_ESQUELETO, 2)
    for lm in landmarks:
        if lm.visibility >= config.VISIBILIDAD_MINIMA:
            cv2.circle(frame, (int(lm.x * ancho), int(lm.y * alto)), 4, COLOR_ESQUELETO, -1)


def main():
    parser = argparse.ArgumentParser(description="Evaluador de ejercicios en vivo con la webcam.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, required=True)
    parser.add_argument("--camara", type=int, default=config.CAMARA_INDICE)
    args = parser.parse_args()

    ejercicio = args.ejercicio
    cfg = config.EJERCICIOS[ejercicio]
    red, paquete = cargar_modelo(ejercicio)

    detector = pose_utils.DetectorPose()
    segmentador = segmentacion.SegmentadorRepeticiones(cfg["umbral_superior"], cfg["umbral_inferior"])
    captura = cv2.VideoCapture(args.camara)

    ultima_etiqueta, ultima_confianza = "-", 0.0
    contador_reps = 0
    inicio = time.perf_counter()

    print(f"Evaluador de ejercicios - {cfg['nombre_legible']} (vista lateral). Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break
            alto, ancho = frame.shape[:2]
            timestamp_ms = int((time.perf_counter() - inicio) * 1000)

            landmarks = detector.detectar(frame, timestamp_ms)
            angulo_rep_actual = None
            if landmarks is not None:
                dibujar_esqueleto(frame, landmarks, ancho, alto)
                senales, _lado, _vis = caracteristicas.calcular_senales_frame(landmarks, ejercicio, ancho, alto)
                if senales is not None:
                    angulo_rep_actual = senales["_angulo_rep"]
                    rep_completa = segmentador.agregar_frame(angulo_rep_actual, senales)
                    if rep_completa is not None and len(rep_completa) >= 3:
                        contador_reps += 1
                        ultima_etiqueta, ultima_confianza = clasificar_repeticion(red, paquete, rep_completa)
                        print(f"Repetición {contador_reps}: {ultima_etiqueta} ({ultima_confianza:.0%})")

            color = COLOR_OK if ultima_etiqueta == "correcto" else COLOR_ERROR
            cv2.putText(frame, f"{cfg['nombre_legible']}  |  reps: {contador_reps}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Ultima: {ultima_etiqueta} ({ultima_confianza:.0%})", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if angulo_rep_actual is not None:
                cv2.putText(frame, f"angulo seg.: {angulo_rep_actual:.0f} deg", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Evaluador de ejercicios", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()
        detector.cerrar()


if __name__ == "__main__":
    main()
