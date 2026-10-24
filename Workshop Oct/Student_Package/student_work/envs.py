"""Entornos de ENTRENAMIENTO del equipo (no forman parte del contrato).

`TrainingMarketEnv` hereda de `shared.practice_market.PracticeMarketEnv`, así que
la observación (16 valores) y las acciones (9) son EXACTAMENTE las oficiales.
Lo único que cambiamos es:

1. Más regímenes de mercado (tendencia bajista, baja liquidez, estrés y
   regímenes aleatorios) para que la política generalice.
2. La recompensa: además del cambio de NAV, penalizamos el riesgo de inventario
   (enfoque media-varianza) y el costo de liquidar la posición al final.

La recompensa SOLO existe durante el entrenamiento: el modelo exportado recibe
observaciones y devuelve 9 puntajes; el torneo no usa nuestra recompensa.
"""
from __future__ import annotations

import numpy as np

from shared.practice_market import PRACTICE_SCENARIOS, PracticeMarketEnv

# Regímenes extra (mismo formato que PRACTICE_SCENARIOS). Simulan los casos que
# el documento anuncia para la evaluación: crisis, baja liquidez y tendencias
# en ambas direcciones.
EXTRA_SCENARIOS = {
    "downtrend": dict(drift=-0.00035, volatility=0.0012, spread_bps=6.0, depth=750, shock_prob=0.0, shock_scale=0.0),
    "low_liquidity": dict(drift=0.0, volatility=0.0015, spread_bps=40.0, depth=150, shock_prob=0.0, shock_scale=0.0),
    "stress": dict(drift=-0.0005, volatility=0.0030, spread_bps=25.0, depth=300, shock_prob=0.05, shock_scale=0.04),
}
ALL_SCENARIOS = {**PRACTICE_SCENARIOS, **EXTRA_SCENARIOS}


def random_scenario(rng: np.random.Generator) -> dict:
    """Régimen aleatorio ("domain randomization"): cada episodio es un mercado distinto."""
    return dict(
        drift=float(rng.uniform(-0.0005, 0.0005)),
        volatility=float(np.exp(rng.uniform(np.log(0.0005), np.log(0.005)))),
        spread_bps=float(rng.uniform(3.0, 40.0)),
        depth=int(rng.integers(150, 1000)),
        shock_prob=float(rng.choice([0.0, 0.0, 0.02, 0.05])),
        shock_scale=float(rng.uniform(0.01, 0.04)),
    )


class TrainingMarketEnv(PracticeMarketEnv):
    """Mercado de práctica con régimen configurable y recompensa ajustada al riesgo.

    reward_mode:
      "basic" -> la recompensa original del paquete (para comparar).
      "risk"  -> dNAV/1000 - 0.5*kappa*(varianza del dNAV/1000 por el inventario)
                 y, al final del episodio, el costo de cerrar la posición.
    """

    def __init__(self, cfg: dict, name: str = "custom", steps: int = 250, seed: int = 0,
                 reward_mode: str = "risk", risk_aversion: float = 0.01):
        # El padre valida el nombre del escenario; lo creamos como "stable" y
        # luego reemplazamos la configuración del mercado.
        super().__init__(scenario="stable", steps=steps, seed=seed)
        self.cfg = dict(cfg)
        self.scenario = name
        self.reward_mode = reward_mode
        self.risk_aversion = float(risk_aversion)
        self.reset()

    @classmethod
    def named(cls, name: str, **kw) -> "TrainingMarketEnv":
        return cls(ALL_SCENARIOS[name], name=name, **kw)

    def step(self, action: int):
        obs, base_reward, done, info = super().step(action)
        if self.reward_mode == "basic":
            return obs, base_reward, done, info

        nav = info["nav_cents"]
        mid = info["mid_cents"]
        inv = info["inventory"]
        d_nav = nav - self._last_nav
        self._last_nav = nav

        # Desviación estándar del próximo cambio de NAV causada por el inventario
        # (en las mismas unidades de la recompensa: dólares*10).
        risk_std = abs(inv) * mid * self.cfg["volatility"] / 1000.0
        reward = d_nav / 1000.0 - 0.5 * self.risk_aversion * risk_std ** 2

        if done:
            # Cerrar la posición cuesta medio spread por acción: lo cobramos al
            # final para que la política aprenda a terminar con poco inventario.
            half_spread = mid * self.cfg["spread_bps"] / 20_000.0
            reward -= abs(inv) * half_spread / 1000.0
        return obs, float(reward), done, info

    def reset(self):
        obs = super().reset()
        self._last_nav = float(self.cash)
        return obs
