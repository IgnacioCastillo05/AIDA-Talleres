"""Milestone 3 - tarea del estudiante: cambiar UN solo umbral de la regla de momentum
y volver a correr los tres regímenes.

    python -m student_work.exp_threshold

Primero con la semilla del demo (19, 150 pasos) para reproducir el demo 02, y luego
con 20 semillas para ver si la conclusión se sostiene o era suerte.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from shared.practice_market import PracticeMarketEnv
from student_work.baselines import momentum_policy

THRESHOLDS = [0.005, 0.015, 0.030, 0.060]
SCENARIOS = ["stable", "trending", "volatile"]


def pnl(policy, scenario: str, seed: int, steps: int) -> tuple[float, int]:
    env = PracticeMarketEnv(scenario=scenario, steps=steps, seed=seed)
    obs, done = env.reset(), False
    while not done:
        obs, _, done, info = env.step(policy(obs))
    return (info["nav_cents"] - 10_000_000) / 100, info["trade_count"]


def main():
    print("A) Semilla del demo (19), 150 pasos")
    print(f"{'umbral':>8} " + " ".join(f"{s:>20}" for s in SCENARIOS))
    for th in THRESHOLDS:
        pol = partial(momentum_policy, threshold=th)
        cells = [pnl(pol, s, 19, 150) for s in SCENARIOS]
        print(f"{th:8.3f} " + " ".join(f"{p:>10,.0f} ({t:3d} tr)" for p, t in cells))

    print("\nB) 20 semillas (1000..1019), 300 pasos: PnL medio +/- desviación")
    print(f"{'umbral':>8} " + " ".join(f"{s:>20}" for s in SCENARIOS))
    for th in THRESHOLDS:
        pol = partial(momentum_policy, threshold=th)
        cells = []
        for s in SCENARIOS:
            vals = np.asarray([pnl(pol, s, 1000 + k, 300)[0] for k in range(20)])
            cells.append(f"{vals.mean():>9,.0f} +/-{vals.std():>6,.0f}")
        print(f"{th:8.3f} " + " ".join(f"{c:>20}" for c in cells))


if __name__ == "__main__":
    main()
