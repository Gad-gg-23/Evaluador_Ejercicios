"""
Entrena el modelo de cada ejercicio (ensamble CNN-1D + MLP, ver modelo.py)
sobre data/caracteristicas/<ejercicio>.csv y lo guarda en
salidas/modelo_<ejercicio>.pt.

El modelo que se guarda se entrena con TODAS las repeticiones: es el que se
usa en tiempo real, y el día de la entrega se prueba con las mismas personas
del equipo, así que conviene que haya visto a todas. Las métricas honestas
(sobre series que el modelo NO vio) las calcula evaluar.py con validación
cruzada agrupada por serie.

Uso:
    python entrenar.py                       # los 3 ejercicios
    python entrenar.py --ejercicio lagartijas
"""

import argparse

import pandas as pd

import caracteristicas
import config
import modelo


def cargar_datos(ejercicio):
    ruta_csv = config.RUTA_CARACTERISTICAS / f"{ejercicio}.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No existe {ruta_csv}. Corre primero extraer_caracteristicas.py.")
    df = pd.read_csv(ruta_csv)
    version = df["version"] if "version" in df.columns else pd.Series(1, index=df.index)
    viejas = df[version.fillna(1).astype(int) != config.VERSION_CARACTERISTICAS]
    if not viejas.empty:
        print(
            f"  AVISO: se ignoran {len(viejas)} filas extraídas con una versión anterior de las características. "
            f"Tienen que volver a correr `extraer`: {sorted(viejas['persona'].unique())}"
        )
    df = df.drop(viejas.index)

    # Mismo video subido dos veces con otro nombre -> filas idénticas que solo inflan una clase.
    columnas = caracteristicas.columnas_caracteristicas(ejercicio)
    duplicadas = df.duplicated(subset=["persona", "etiqueta"] + columnas)
    if duplicadas.any():
        print(f"  AVISO: se quitan {int(duplicadas.sum())} repeticiones duplicadas (mismo video con otro nombre).")
    df = df[~duplicadas].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{ruta_csv} no tiene repeticiones de la versión actual: corre `extraer` primero.")

    desconocidas = set(df["etiqueta"]) - set(config.EJERCICIOS[ejercicio]["etiquetas"])
    if desconocidas:
        print(f"  AVISO: se ignoran etiquetas que no están en config: {sorted(desconocidas)}")
        df = df[~df["etiqueta"].isin(desconocidas)].reset_index(drop=True)

    # una "serie" = un video concreto de una persona con una etiqueta
    df["serie"] = df["persona"] + "/" + df["etiqueta"] + "/" + df["video"].astype(str)
    return df


def entrenar_ejercicio(ejercicio):
    print(f"\n=== Entrenando {ejercicio} ===")
    df = cargar_datos(ejercicio)
    etiquetas = config.EJERCICIOS[ejercicio]["etiquetas"]

    faltantes = set(etiquetas) - set(df["etiqueta"].unique())
    if faltantes:
        print(f"  AVISO: todavía no hay ejemplos de: {sorted(faltantes)}")

    m = modelo.ModeloRepeticion(ejercicio, etiquetas).entrenar(df)
    acierto_train = (m.predecir(df) == df["etiqueta"].map(etiquetas.index).to_numpy()).mean()

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta = config.RUTA_SALIDAS / f"modelo_{ejercicio}.pt"
    m.guardar(ruta, metadatos={
        "repeticiones": len(df), "series": df["serie"].nunique(),
        "personas": sorted(df["persona"].unique()),
    })
    print(f"  {len(df)} repeticiones, {df['serie'].nunique()} series, personas: {sorted(df['persona'].unique())}")
    print(f"  exactitud sobre entrenamiento: {acierto_train:.3f} (NO es la métrica a reportar: usa `evaluar`)")
    print(f"  modelo guardado en {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de repeticiones por ejercicio.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, default=None)
    args = parser.parse_args()

    for ejercicio in [args.ejercicio] if args.ejercicio else config.LISTA_EJERCICIOS:
        entrenar_ejercicio(ejercicio)


if __name__ == "__main__":
    main()
