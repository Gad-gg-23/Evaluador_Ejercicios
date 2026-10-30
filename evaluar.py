"""
Evalúa el modelo guardado de cada ejercicio sobre su partición de prueba
(misma semilla y misma lógica de división que entrenar.py, así que es
prueba de verdad, no datos ya vistos). Guarda matriz de confusión y reporte
por clase en salidas/.

Uso:
    python evaluar.py
    python evaluar.py --ejercicio curl_biceps
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import config
import entrenar
import modelo


def evaluar_ejercicio(ejercicio):
    print(f"\n=== Evaluando {ejercicio} ===")
    ruta_modelo = config.RUTA_SALIDAS / f"modelo_{ejercicio}.pt"
    if not ruta_modelo.exists():
        print(f"  No existe {ruta_modelo}. Corre primero entrenar.py.")
        return

    paquete = torch.load(ruta_modelo, map_location=config.DISPOSITIVO, weights_only=False)
    columnas, etiquetas = paquete["columnas"], paquete["etiquetas"]

    df = entrenar.cargar_datos(ejercicio)
    _, _, df_prueba = entrenar.dividir(df, config.FRACCION_PRUEBA, config.FRACCION_VALIDACION, config.RANDOM_STATE)
    if df_prueba.empty:
        print("  Partición de prueba vacía, nada que evaluar todavía.")
        return

    media = pd.Series(paquete["media"], index=columnas)
    desviacion = pd.Series(paquete["desviacion"], index=columnas)
    x = torch.tensor(((df_prueba[columnas] - media) / desviacion).values.astype("float32"))
    y_real = df_prueba["etiqueta"].map(etiquetas.index).values

    red = modelo.ClasificadorRepeticion(len(columnas), len(etiquetas)).to(config.DISPOSITIVO)
    red.load_state_dict(paquete["state_dict"])
    red.eval()
    with torch.no_grad():
        y_pred = red(x.to(config.DISPOSITIVO)).argmax(dim=1).cpu().numpy()

    print(classification_report(y_real, y_pred, labels=range(len(etiquetas)), target_names=etiquetas, zero_division=0))

    matriz = confusion_matrix(y_real, y_pred, labels=range(len(etiquetas)))
    disp = ConfusionMatrixDisplay(matriz, display_labels=etiquetas)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Matriz de confusión - {ejercicio}")
    fig.tight_layout()

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta_png = config.RUTA_SALIDAS / f"matriz_confusion_{ejercicio}.png"
    fig.savefig(ruta_png)
    plt.close(fig)
    print(f"  Matriz de confusión guardada en {ruta_png}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el modelo de cada ejercicio sobre su partición de prueba.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, default=None)
    args = parser.parse_args()

    for ejercicio in [args.ejercicio] if args.ejercicio else config.LISTA_EJERCICIOS:
        evaluar_ejercicio(ejercicio)


if __name__ == "__main__":
    main()
