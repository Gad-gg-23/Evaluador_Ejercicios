"""
Clasificador de una repetición -> clase (correcto / error_X).

Es un ENSAMBLE de dos redes que promedian sus probabilidades:

  1. CNN-1D híbrida: convoluciones 1D sobre la SECUENCIA de cada señal
     (ángulos, inclinación, talón, codo... remuestreadas a
     config.PUNTOS_SECUENCIA puntos) -> captura la FORMA del movimiento
     (p. ej. el torso que se balancea a mitad del curl, el talón que se
     despega solo abajo). Su salida se concatena con las características
     agregadas (min/max/media/rango) antes de la capa final.
  2. MLP sobre las características agregadas -> más estable con pocos datos.

En la comparación con validación cruzada por serie (ver README) el
ensamble fue el más estable en los 3 ejercicios: la CNN gana en
lagartijas/sentadillas, el MLP en curl, y promediar se queda con lo mejor
de ambos.

`ModeloRepeticion` se encarga de todo (normalizar, entrenar, predecir,
guardar/cargar), así entrenar.py, evaluar.py y tiempo_real.py no repiten
esa lógica.
"""

import numpy as np
import torch
import torch.nn as nn

import caracteristicas
import config


class CNN1DHibrida(nn.Module):
    def __init__(self, canales, n_agregadas, n_clases):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(canales, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.cabeza = nn.Sequential(
            nn.Dropout(config.DROPOUT), nn.Linear(32 + n_agregadas, 32), nn.ReLU(),
            nn.Dropout(config.DROPOUT), nn.Linear(32, n_clases),
        )

    def forward(self, secuencia, agregadas):
        return self.cabeza(torch.cat([self.conv(secuencia), agregadas], dim=1))


class MLPAgregadas(nn.Module):
    def __init__(self, n_agregadas, n_clases):
        super().__init__()
        capas, tam_anterior = [], n_agregadas
        for tam in config.CAPAS_OCULTAS:
            capas += [nn.Linear(tam_anterior, tam), nn.ReLU(), nn.Dropout(config.DROPOUT)]
            tam_anterior = tam
        capas.append(nn.Linear(tam_anterior, n_clases))
        self.red = nn.Sequential(*capas)

    def forward(self, secuencia, agregadas):
        return self.red(agregadas)


class ModeloRepeticion:
    """Ensamble CNN-1D + MLP para un ejercicio."""

    def __init__(self, ejercicio, etiquetas):
        self.ejercicio = ejercicio
        self.etiquetas = list(etiquetas)
        self.col_agg = caracteristicas.columnas_caracteristicas(ejercicio)
        self.col_seq = caracteristicas.columnas_secuencia(ejercicio)
        self.canales = len(caracteristicas.senales_del_ejercicio(ejercicio))
        self.redes = []

    # ---------- preparación ----------
    def _matrices(self, df):
        agg = df[self.col_agg].to_numpy(dtype="float64")
        seq = df[self.col_seq].to_numpy(dtype="float64").reshape(len(df), self.canales, -1)
        return agg, seq

    def _tensores(self, df):
        agg, seq = self._matrices(df)
        agg = np.nan_to_num((agg - self.mu_agg) / self.sd_agg)
        seq = np.nan_to_num((seq - self.mu_seq) / self.sd_seq)
        return (torch.tensor(seq, dtype=torch.float32, device=config.DISPOSITIVO),
                torch.tensor(agg, dtype=torch.float32, device=config.DISPOSITIVO))

    def _nuevas_redes(self):
        n = len(self.etiquetas)
        return [CNN1DHibrida(self.canales, len(self.col_agg), n).to(config.DISPOSITIVO),
                MLPAgregadas(len(self.col_agg), n).to(config.DISPOSITIVO)]

    # ---------- entrenamiento ----------
    def entrenar(self, df, semilla=config.RANDOM_STATE):
        torch.manual_seed(semilla)
        agg, seq = self._matrices(df)
        self.mu_agg, self.sd_agg = agg.mean(0), agg.std(0) + 1e-6
        self.mu_seq = seq.mean((0, 2), keepdims=True)
        self.sd_seq = seq.std((0, 2), keepdims=True) + 1e-6
        x_seq, x_agg = self._tensores(df)
        y = torch.tensor(df["etiqueta"].map(self.etiquetas.index).to_numpy(), device=config.DISPOSITIVO)

        # pesos por clase: si un error tiene menos repeticiones no se ignora
        conteo = np.bincount(y.cpu().numpy(), minlength=len(self.etiquetas)).clip(1)
        pesos = torch.tensor(len(y) / (len(self.etiquetas) * conteo), dtype=torch.float32, device=config.DISPOSITIVO)
        perdida_fn = nn.CrossEntropyLoss(weight=pesos, label_smoothing=config.SUAVIZADO_ETIQUETAS)

        self.redes = self._nuevas_redes()
        for red in self.redes:
            opt = torch.optim.AdamW(red.parameters(), lr=config.TASA_APRENDIZAJE, weight_decay=config.WEIGHT_DECAY)
            for _ in range(config.NUM_EPOCAS):
                red.train()
                perm = torch.randperm(len(y), device=config.DISPOSITIVO)
                for i in range(0, len(y), config.BATCH_SIZE):
                    lote = perm[i:i + config.BATCH_SIZE]
                    if len(lote) < 2:  # BatchNorm necesita al menos 2 ejemplos
                        continue
                    opt.zero_grad()
                    perdida_fn(red(x_seq[lote], x_agg[lote]), y[lote]).backward()
                    opt.step()
        return self

    # ---------- predicción ----------
    def probabilidades(self, df):
        x_seq, x_agg = self._tensores(df)
        with torch.no_grad():
            probs = [torch.softmax(red.eval()(x_seq, x_agg), dim=1) for red in self.redes]
        return (sum(probs) / len(probs)).cpu().numpy()

    def predecir(self, df):
        """Regresa índices de clase (posición en self.etiquetas)."""
        return self.probabilidades(df).argmax(1)

    # ---------- guardar / cargar ----------
    def guardar(self, ruta, metadatos=None):
        torch.save({
            "ejercicio": self.ejercicio, "etiquetas": self.etiquetas,
            "version_caracteristicas": config.VERSION_CARACTERISTICAS,
            "mu_agg": self.mu_agg, "sd_agg": self.sd_agg, "mu_seq": self.mu_seq, "sd_seq": self.sd_seq,
            "redes": [red.state_dict() for red in self.redes],
            "metadatos": metadatos or {},
        }, ruta)

    @classmethod
    def cargar(cls, ruta):
        p = torch.load(ruta, map_location=config.DISPOSITIVO, weights_only=False)
        if p.get("version_caracteristicas") != config.VERSION_CARACTERISTICAS:
            raise ValueError(f"{ruta} se entrenó con otra versión de características: vuelve a correr `entrenar`.")
        m = cls(p["ejercicio"], p["etiquetas"])
        m.mu_agg, m.sd_agg, m.mu_seq, m.sd_seq = p["mu_agg"], p["sd_agg"], p["mu_seq"], p["sd_seq"]
        m.redes = m._nuevas_redes()
        for red, estado in zip(m.redes, p["redes"]):
            red.load_state_dict(estado)
            red.eval()
        m.metadatos = p.get("metadatos", {})
        return m
