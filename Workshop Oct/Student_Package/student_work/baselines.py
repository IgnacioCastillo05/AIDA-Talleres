"""Políticas de referencia transparentes (sin aprendizaje).

Sirven para responder la pregunta clave de cualquier modelo de RL:
¿le gana a una regla simple que podemos explicar en una línea?
Todas usan el mismo contrato: reciben la observación de 16 valores y devuelven
una acción 0..8.
"""
from __future__ import annotations

import numpy as np

# Índices de la observación oficial (ver shared/contract.py -> OBSERVATION_NAMES)
RETURN_5 = 1
INVENTORY = 8


def hold_policy(obs: np.ndarray) -> int:
    """Nunca opera. NAV final = capital inicial. Es el "no hacer daño"."""
    return 0


def momentum_policy(obs: np.ndarray, threshold: float = 0.015) -> int:
    """La regla del demo 02 (momentum de 5 pasos con órdenes de mercado pequeñas)."""
    momentum_5 = float(obs[RETURN_5])
    inventory = float(obs[INVENTORY])
    if momentum_5 > threshold and inventory < 0.8:
        return 1  # BUY_MARKET_SMALL
    if momentum_5 < -threshold and inventory > -0.8:
        return 2  # SELL_MARKET_SMALL
    return 0


def market_maker_policy(obs: np.ndarray) -> int:
    """Creador de mercado con control de inventario.

    Las órdenes límite compran en el bid y venden en el ask, es decir, ganan medio
    spread frente al precio medio cuando se ejecutan. Para no acumular riesgo,
    siempre cotizamos del lado que REDUCE el inventario.
    """
    inventory = float(obs[INVENTORY])
    return 8 if inventory > 0 else 7  # SELL_LIMIT_LARGE / BUY_LIMIT_LARGE


BASELINES = {
    "hold": hold_policy,
    "momentum": momentum_policy,
    "market_maker": market_maker_policy,
}
