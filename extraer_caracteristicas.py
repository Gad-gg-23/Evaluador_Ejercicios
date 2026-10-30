"""
Convierte los videos de data/propios/<ejercicio>/<etiqueta>/<persona>/*.mp4
en un CSV de características por repetición: data/caracteristicas/<ejercicio>.csv

Cada repetición detectada en cada video es una fila. La etiqueta y la
persona salen de la ruta del archivo, no hay que escribirlas a mano en
ningún lado -- así que lo único que necesita el equipo es grabar y ponerlos
en la carpeta correcta (ver GUIA_EQUIPO.md).

Uso:
    python extraer_caracteristicas.py                   # los 3 ejercicios
    python extraer_caracteristicas.py --ejercicio sentadillas
"""

import argparse
import csv
from pathlib import Path

import cv2

import caracteristicas
import config
import pose_utils
import segmentacion


EXTENSIONES_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}


def procesar_video(detector, ruta_video, ejercicio):
    """Regresa la lista de repeticiones (cada una: dict de características) detectadas en el video."""
    cfg = config.EJERCICIOS[ejercicio]
    captura = cv2.VideoCapture(str(ruta_video))
    fps = captura.get(cv2.CAP_PROP_FPS) or 30.0

    segmentador = segmentacion.SegmentadorRepeticiones(cfg["umbral_superior"], cfg["umbral_inferior"])
    repeticiones = []
    num_frame = 0

    while True:
        ok, frame = captura.read()
        if not ok:
            break
        alto, ancho = frame.shape[:2]
        timestamp_ms = int(num_frame * 1000 / fps)
        num_frame += 1

        landmarks = detector.detectar(frame, timestamp_ms)
        if landmarks is None:
            continue

        senales, _lado, _vis = caracteristicas.calcular_senales_frame(landmarks, ejercicio, ancho, alto)
        if senales is None:
            continue

        rep_completa = segmentador.agregar_frame(senales["_angulo_rep"], senales)
        if rep_completa is not None and len(rep_completa) >= 3:
            repeticiones.append(caracteristicas.agregar_repeticion(rep_completa))

    captura.release()
    return repeticiones


def extraer_ejercicio(ejercicio, detector):
    cfg = config.EJERCICIOS[ejercicio]
    columnas = ["ejercicio", "etiqueta", "persona", "video"] + caracteristicas.columnas_caracteristicas(ejercicio)
    filas = []

    for etiqueta in cfg["etiquetas"]:
        carpeta_etiqueta = config.RUTA_VIDEOS_PROPIOS / ejercicio / etiqueta
        if not carpeta_etiqueta.exists():
            print(f"  (sin carpeta {carpeta_etiqueta}, se omite)")
            continue
        for carpeta_persona in sorted(p for p in carpeta_etiqueta.iterdir() if p.is_dir()):
            persona = carpeta_persona.name
            videos = sorted(
                v for v in carpeta_persona.iterdir() if v.suffix.lower() in EXTENSIONES_VIDEO
            )
            for ruta_video in videos:
                reps = procesar_video(detector, ruta_video, ejercicio)
                print(f"  {ejercicio}/{etiqueta}/{persona}/{ruta_video.name}: {len(reps)} repetición(es)")
                for rep in reps:
                    fila = {"ejercicio": ejercicio, "etiqueta": etiqueta, "persona": persona, "video": ruta_video.name}
                    fila.update(rep)
                    filas.append(fila)

    config.RUTA_CARACTERISTICAS.mkdir(parents=True, exist_ok=True)
    ruta_csv = config.RUTA_CARACTERISTICAS / f"{ejercicio}.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"-> {len(filas)} repeticiones guardadas en {ruta_csv}")


def main():
    parser = argparse.ArgumentParser(description="Extrae características de repeticiones desde video.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, default=None)
    args = parser.parse_args()

    ejercicios = [args.ejercicio] if args.ejercicio else config.LISTA_EJERCICIOS
    detector = pose_utils.DetectorPose()
    try:
        for ejercicio in ejercicios:
            print(f"\n=== {ejercicio} ===")
            extraer_ejercicio(ejercicio, detector)
    finally:
        detector.cerrar()


if __name__ == "__main__":
    main()
