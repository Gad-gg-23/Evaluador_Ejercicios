"""MLP pequeño: características agregadas de una repetición -> clase (correcto / error_X)."""

import torch.nn as nn

import config


class ClasificadorRepeticion(nn.Module):
    def __init__(self, num_entradas, num_clases, capas_ocultas=None, dropout=None):
        super().__init__()
        capas_ocultas = capas_ocultas or config.CAPAS_OCULTAS
        dropout = config.DROPOUT if dropout is None else dropout

        capas = []
        tam_anterior = num_entradas
        for tam in capas_ocultas:
            capas += [nn.Linear(tam_anterior, tam), nn.ReLU(), nn.Dropout(dropout)]
            tam_anterior = tam
        capas.append(nn.Linear(tam_anterior, num_clases))

        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)
