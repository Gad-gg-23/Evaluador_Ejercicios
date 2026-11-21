"""
Métricas honestas del modelo de cada ejercicio con VALIDACIÓN CRUZADA
AGRUPADA POR SERIE: en cada partición se entrena un modelo nuevo (misma
arquitectura e hiperparámetros que entrenar.py) y se prueba sobre series
(videos) que ese modelo nunca vio. Así cada repetición se predice
exactamente una vez sin haber estado en su propio entrenamiento.

  - por serie (default): simula el día de la entrega -> mismas personas,
    grabaciones nuevas.
  - --por-persona: leave-one-person-out -> qué tan bien generaliza a alguien
    que el modelo nunca vio (requiere >= 3 personas).

Guarda el reporte por clase y la matriz de confusión en salidas/.

Uso:
    python evaluar.py
    python evaluar.py --ejercicio curl_biceps
    python evaluar.py --por-persona
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

import config
import entrenar
import modelo

PARTICIONES_POR_SERIE = 4


def predicciones_cruzadas(df, ejercicio, etiquetas, por_persona):
    y = df["etiqueta"].map(etiquetas.index).to_numpy()
    if por_persona:
        grupos, divisor = df["persona"].to_numpy(), LeaveOneGroupOut()
    else:
        grupos = df["serie"].to_numpy()
        divisor = StratifiedGroupKFold(PARTICIONES_POR_SERIE, shuffle=True, random_state=config.RANDOM_STATE)
    y_pred = np.empty_like(y)
    for i, (idx_train, idx_prueba) in enumerate(divisor.split(df, y, grupos), 1):
        m = modelo.ModeloRepeticion(ejercicio, etiquetas).entrenar(df.iloc[idx_train])
        y_pred[idx_prueba] = m.predecir(df.iloc[idx_prueba])
        if por_persona:
            f1 = f1_score(y[idx_prueba], y_pred[idx_prueba], average="macro", zero_division=0)
            print(f"  sin ver a {grupos[idx_prueba][0]}: F1 macro = {f1:.3f}")
        else:
            print(f"  partición {i}: {len(idx_prueba)} repeticiones de prueba")
    return y, y_pred


def evaluar_ejercicio(ejercicio, por_persona):
    modo = "persona fuera" if por_persona else "por serie"
    print(f"\n=== Evaluando {ejercicio} ({modo}) ===")
    df = entrenar.cargar_datos(ejercicio)
    etiquetas = config.EJERCICIOS[ejercicio]["etiquetas"]
    if por_persona and df["persona"].nunique() < 3:
        print("  Se necesitan al menos 3 personas para evaluar 'persona fuera'.")
        return
    if not por_persona and df["serie"].nunique() < PARTICIONES_POR_SERIE:
        print(f"  Se necesitan al menos {PARTICIONES_POR_SERIE} series para evaluar.")
        return

    y, y_pred = predicciones_cruzadas(df, ejercicio, etiquetas, por_persona)
    reporte = classification_report(y, y_pred, labels=range(len(etiquetas)), target_names=etiquetas,
                                    zero_division=0, digits=3)
    print(reporte)
    print(f"  F1 macro ({modo}): {f1_score(y, y_pred, average='macro', zero_division=0):.3f}")

    sufijo = "_persona_fuera" if por_persona else ""
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    (config.RUTA_SALIDAS / f"reporte_{ejercicio}{sufijo}.txt").write_text(reporte, encoding="utf-8")

    matriz = confusion_matrix(y, y_pred, labels=range(len(etiquetas)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(matriz, display_labels=etiquetas).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Matriz de confusión - {ejercicio} ({modo})")
    fig.tight_layout()
    ruta_png = config.RUTA_SALIDAS / f"matriz_confusion_{ejercicio}{sufijo}.png"
    fig.savefig(ruta_png)
    plt.close(fig)
    print(f"  Matriz de confusión guardada en {ruta_png}")


def main():
    parser = argparse.ArgumentParser(description="Métricas con validación cruzada agrupada.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, default=None)
    parser.add_argument("--por-persona", action="store_true",
                        help="leave-one-person-out en vez de validación cruzada por serie")
    args = parser.parse_args()

    for ejercicio in [args.ejercicio] if args.ejercicio else config.LISTA_EJERCICIOS:
        evaluar_ejercicio(ejercicio, args.por_persona)


if __name__ == "__main__":
    main()
