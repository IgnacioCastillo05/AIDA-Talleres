from __future__ import annotations
import torch
from torch import nn
from shared.contract import OBSERVATION_SIZE, ACTION_SIZE

class QNetwork(nn.Module):
    """Starter network. You may redesign the internal architecture.

    The only fixed requirement is:
      input  : float32 tensor [batch, 16]
      output : float32 tensor [batch, 9]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Red "dueling" (Wang et al., 2016): Q(s,a) = V(s) + A(s,a) - mean_a A(s,a).

    Por qué ayuda aquí: en muchos estados del mercado casi todas las acciones
    valen parecido (el valor depende sobre todo del inventario y del régimen).
    Separar V(s) (cuánto vale el estado) de A(s,a) (cuánto mejor es cada acción)
    hace que el aprendizaje del valor del estado no dependa de probar cada acción.

    El contrato no cambia: entra [batch, 16] float32 y salen [batch, 9] float32.
    """
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, ACTION_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


def build_model(arch: str) -> nn.Module:
    if arch == "mlp":
        return QNetwork()
    if arch == "dueling":
        return DuelingQNetwork()
    raise ValueError(f"Unknown architecture {arch}; choose 'mlp' or 'dueling'")
