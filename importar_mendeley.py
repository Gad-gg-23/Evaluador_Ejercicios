"""
Importa videos del dataset de apoyo de Mendeley ("A Multi-View Raw Video
Dataset of Seven Fitness Exercises with Good/Bad Form Labels",
https://data.mendeley.com/datasets/kgbb3yn47p) a la estructura que espera
extraer_caracteristicas.py: data/propios/<ejercicio>/<etiqueta>/<persona>/.

Por qué un manifiesto y no adivinar el nombre de archivo automáticamente:
Mendeley no tiene una API de descarga como Kaggle (hay que bajar el .zip a
mano desde el navegador) y el formato exacto del nombre de cada archivo
puede variar entre versiones del dataset. Es más confiable que alguien del
equipo revise una vez la carpeta descomprimida y llene el manifiesto, que
adivinar un patrón y etiquetar mal sin que nadie se dé cuenta.

Este dataset solo trae "good"/"bad" (no los errores específicos del
póster), así que al importarlo el "bad" se guarda como
"incorrecto_generico": sirve para que el modelo vea más cuerpos y cámaras
distintas en la clase "correcto" y para pre-entrenar la noción de "algo
está mal" en la clase "bad", pero NO reemplaza los videos propios con los
errores específicos (cadera_caida, talones_levantados, etc.) -- esos los
tiene que grabar el equipo.

Uso:
    1. Descarga y descomprime el dataset de Mendeley en una carpeta, ej.
       data/mendeley_bruto/ (ignorada por Git, ver .gitignore).
    2. Revisa un par de archivos para confirmar exercise/calidad/sujeto y
       llena un manifiesto CSV con columnas:
       ruta_origen,ejercicio,calidad,persona
       donde ejercicio es uno de: lagartijas, sentadillas, curl_biceps
       y calidad es "good" o "bad".
    3. python importar_mendeley.py --manifiesto mi_manifiesto.csv
"""

import argparse
import csv
import shutil
from pathlib import Path

import config

MAPEO_CALIDAD = {"good": "correcto", "bad": "incorrecto_generico"}


def main():
    parser = argparse.ArgumentParser(description="Importa videos del dataset de Mendeley usando un manifiesto.")
    parser.add_argument("--manifiesto", required=True, help="CSV con columnas ruta_origen,ejercicio,calidad,persona")
    args = parser.parse_args()

    copiados = 0
    with open(args.manifiesto, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            ejercicio = fila["ejercicio"].strip()
            calidad = fila["calidad"].strip().lower()
            persona = fila["persona"].strip()
            origen = Path(fila["ruta_origen"].strip())

            if ejercicio not in config.LISTA_EJERCICIOS:
                print(f"  omitido (ejercicio desconocido '{ejercicio}'): {origen}")
                continue
            if calidad not in MAPEO_CALIDAD:
                print(f"  omitido (calidad desconocida '{calidad}', usa good/bad): {origen}")
                continue
            if not origen.exists():
                print(f"  omitido (no existe): {origen}")
                continue

            etiqueta = MAPEO_CALIDAD[calidad]
            destino_dir = config.RUTA_VIDEOS_PROPIOS / ejercicio / etiqueta / f"mendeley_{persona}"
            destino_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origen, destino_dir / origen.name)
            copiados += 1

    print(f"{copiados} video(s) copiados a data/propios/.")
    print("Nota: 'incorrecto_generico' no es ninguna de las etiquetas de error del póster; "
          "solo úsalo si decides agregar esa clase extra, o ignóralo en entrenar.py filtrando el CSV.")


if __name__ == "__main__":
    main()
