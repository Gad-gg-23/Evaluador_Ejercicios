"""
Entrena un MLP por ejercicio sobre data/caracteristicas/<ejercicio>.csv
(generado por extraer_caracteristicas.py) y guarda el modelo + el
escalador (media/desviación) + el orden de columnas y etiquetas en
salidas/modelo_<ejercicio>.pt.

División train/validación/prueba AGRUPADA POR PERSONA: las repeticiones de
un mismo integrante del equipo quedan todas del mismo lado, para no repetir
la fuga de datos que ya se detectó en la práctica de Temperatura (ahí,
mediciones muy parecidas terminaban a la vez en entrenamiento y en prueba).
Si hay menos de 3 personas todavía, se usa una división aleatoria simple
nada más para poder probar el pipeline -- no hay que confiar en esas
métricas hasta tener más gente.

Uso:
    python entrenar.py                       # los 3 ejercicios
    python entrenar.py --ejercicio lagartijas
"""

import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import caracteristicas
import config
import modelo


def cargar_datos(ejercicio):
    ruta_csv = config.RUTA_CARACTERISTICAS / f"{ejercicio}.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No existe {ruta_csv}. Corre primero extraer_caracteristicas.py.")
    df = pd.read_csv(ruta_csv)
    if df.empty:
        raise ValueError(f"{ruta_csv} está vacío: todavía no hay repeticiones extraídas para {ejercicio}.")
    return df


def dividir(df, fraccion_prueba, fraccion_val, random_state):
    if df["persona"].nunique() >= 3:
        divisor_prueba = GroupShuffleSplit(n_splits=1, test_size=fraccion_prueba, random_state=random_state)
        idx_resto, idx_prueba = next(divisor_prueba.split(df, groups=df["persona"]))
        df_resto, df_prueba = df.iloc[idx_resto], df.iloc[idx_prueba]

        fraccion_val_ajustada = fraccion_val / (1 - fraccion_prueba)
        divisor_val = GroupShuffleSplit(n_splits=1, test_size=fraccion_val_ajustada, random_state=random_state)
        idx_train, idx_val = next(divisor_val.split(df_resto, groups=df_resto["persona"]))
        return df_resto.iloc[idx_train], df_resto.iloc[idx_val], df_prueba

    # Fallback sin agrupar (pocas personas todavía): split aleatorio simple.
    df_resto, df_prueba = train_test_split(df, test_size=fraccion_prueba, random_state=random_state)
    fraccion_val_ajustada = fraccion_val / (1 - fraccion_prueba)
    df_train, df_val = train_test_split(df_resto, test_size=fraccion_val_ajustada, random_state=random_state)
    return df_train, df_val, df_prueba


def entrenar_ejercicio(ejercicio):
    print(f"\n=== Entrenando {ejercicio} ===")
    df = cargar_datos(ejercicio)
    cfg = config.EJERCICIOS[ejercicio]
    etiquetas = cfg["etiquetas"]
    columnas = caracteristicas.columnas_caracteristicas(ejercicio)

    faltantes = set(etiquetas) - set(df["etiqueta"].unique())
    if faltantes:
        print(f"  AVISO: todavía no hay ejemplos de: {sorted(faltantes)}")

    num_personas = df["persona"].nunique()
    if num_personas < 3:
        print(
            f"  AVISO: solo {num_personas} persona(s) en los datos; se usa un split aleatorio "
            "(no agrupado). Agrega videos de más integrantes antes de confiar en las métricas."
        )

    df_train, df_val, df_prueba = dividir(df, config.FRACCION_PRUEBA, config.FRACCION_VALIDACION, config.RANDOM_STATE)

    media = df_train[columnas].mean()
    desviacion = df_train[columnas].std().replace(0, 1.0).fillna(1.0)

    def preparar(df_split):
        x = ((df_split[columnas] - media) / desviacion).values.astype("float32")
        y = df_split["etiqueta"].map(etiquetas.index).values.astype("int64")
        return torch.tensor(x), torch.tensor(y)

    x_train, y_train = preparar(df_train)
    x_val, y_val = preparar(df_val)
    x_prueba, y_prueba = preparar(df_prueba)

    red = modelo.ClasificadorRepeticion(len(columnas), len(etiquetas)).to(config.DISPOSITIVO)
    optimizador = torch.optim.Adam(red.parameters(), lr=config.TASA_APRENDIZAJE)
    perdida_fn = nn.CrossEntropyLoss()

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    cargador = torch.utils.data.DataLoader(
        dataset_train, batch_size=min(config.BATCH_SIZE, max(len(dataset_train), 1)), shuffle=True
    )

    mejor_f1 = -1.0
    mejor_estado = None
    sin_mejora = 0

    for epoca in range(config.NUM_EPOCAS):
        red.train()
        for xb, yb in cargador:
            xb, yb = xb.to(config.DISPOSITIVO), yb.to(config.DISPOSITIVO)
            optimizador.zero_grad()
            perdida = perdida_fn(red(xb), yb)
            perdida.backward()
            optimizador.step()

        red.eval()
        with torch.no_grad():
            pred_val = red(x_val.to(config.DISPOSITIVO)).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val.numpy(), pred_val, average="macro", zero_division=0) if len(y_val) else 0.0

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= config.PACIENCIA_EARLY_STOPPING:
                print(f"  early stopping en época {epoca} (mejor F1 val = {mejor_f1:.3f})")
                break

    if mejor_estado is not None:
        red.load_state_dict(mejor_estado)

    red.eval()
    with torch.no_grad():
        pred_prueba = red(x_prueba.to(config.DISPOSITIVO)).argmax(dim=1).cpu().numpy()
    f1_prueba = f1_score(y_prueba.numpy(), pred_prueba, average="macro", zero_division=0) if len(y_prueba) else 0.0
    print(
        f"  F1 macro validación: {mejor_f1:.3f} | F1 macro prueba: {f1_prueba:.3f} "
        f"({len(df_train)} train / {len(df_val)} val / {len(df_prueba)} prueba)"
    )

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": red.state_dict(),
            "columnas": columnas,
            "etiquetas": etiquetas,
            "media": media.values.astype("float32"),
            "desviacion": desviacion.values.astype("float32"),
        },
        config.RUTA_SALIDAS / f"modelo_{ejercicio}.pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de repeticiones por ejercicio.")
    parser.add_argument("--ejercicio", choices=config.LISTA_EJERCICIOS, default=None)
    args = parser.parse_args()

    for ejercicio in [args.ejercicio] if args.ejercicio else config.LISTA_EJERCICIOS:
        entrenar_ejercicio(ejercicio)


if __name__ == "__main__":
    main()
