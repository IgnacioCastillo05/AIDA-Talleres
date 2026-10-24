"""Evaluación robusta: varios regímenes x varias semillas x varias métricas.

Ejemplos (desde la raíz de Student_Package):

    python -m student_work.evaluate --model outputs/model.pt
    python -m student_work.evaluate --model outputs/model.pt --baselines hold momentum market_maker
    python -m student_work.evaluate --model outputs/model.pt --seeds 30 --markdown outputs/eval.md

Separación de semillas (para no "hacer trampa" con el test):
    entrenamiento : seed + episodio  (números pequeños, < 5000)
    validación    : 5000..           (se usa para escoger el mejor checkpoint)
    test          : 9000..           (solo para el reporte final)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from shared.contract import STARTING_CASH_CENTS
from student_work.baselines import BASELINES
from student_work.envs import ALL_SCENARIOS, TrainingMarketEnv

PRACTICE = ["stable", "trending", "volatile"]
EXTRA = ["downtrend", "low_liquidity", "stress"]
VALIDATION_SEED = 5000
TEST_SEED = 9000

Policy = Callable[[np.ndarray], int]


def torch_policy(model: torch.nn.Module) -> Policy:
    model.eval()

    @torch.inference_mode()
    def act(obs: np.ndarray) -> int:
        return int(model(torch.as_tensor(obs).unsqueeze(0)).argmax(dim=1).item())
    return act


def load_policy(path: Path) -> Policy:
    # Se carga exactamente como lo hará el profesor: TorchScript en CPU.
    return torch_policy(torch.jit.load(str(path), map_location="cpu"))


def run_episode(policy: Policy, scenario: str, seed: int, steps: int = 300) -> dict:
    env = TrainingMarketEnv.named(scenario, steps=steps, seed=seed, reward_mode="basic")
    obs = env.reset()
    done = False
    peak = float(STARTING_CASH_CENTS)
    max_dd = 0.0
    max_abs_inv = 0
    while not done:
        obs, _, done, info = env.step(policy(obs))
        peak = max(peak, info["nav_cents"])
        max_dd = max(max_dd, peak - info["nav_cents"])
        max_abs_inv = max(max_abs_inv, abs(info["inventory"]))
    return {
        "pnl": (info["nav_cents"] - STARTING_CASH_CENTS) / 100.0,
        "max_drawdown": max_dd / 100.0,
        "trades": info["trade_count"],
        "final_inventory": info["inventory"],
        "max_abs_inventory": max_abs_inv,
    }


def evaluate(policy: Policy, scenarios: list[str], n_seeds: int, base_seed: int) -> dict:
    """Devuelve {escenario: {métrica: arreglo por semilla}}."""
    out = {}
    for i, sc in enumerate(scenarios):
        rows = [run_episode(policy, sc, base_seed + 100 * i + k) for k in range(n_seeds)]
        out[sc] = {key: np.asarray([r[key] for r in rows], dtype=np.float64) for key in rows[0]}
    return out


def robust_score(results: dict) -> float:
    """Un solo número para escoger modelos: PnL medio menos la mitad del drawdown
    medio, promediado entre regímenes. Premia ganar dinero SIN grandes caídas."""
    return float(np.mean([r["pnl"].mean() - 0.5 * r["max_drawdown"].mean() for r in results.values()]))


def summary_table(name: str, results: dict) -> list[str]:
    lines = []
    for sc, r in results.items():
        win = 100.0 * (r["pnl"] > 0).mean()
        lines.append(
            f"| {name} | {sc} | {r['pnl'].mean():,.0f} | {r['pnl'].std():,.0f} | {r['pnl'].min():,.0f} | "
            f"{win:.0f}% | {r['max_drawdown'].mean():,.0f} | {r['trades'].mean():.0f} | "
            f"{np.abs(r['final_inventory']).mean():.0f} |"
        )
    lines.append(f"| **{name}** | **score robusto** | **{robust_score(results):,.0f}** | | | | | | |")
    return lines


HEADER = [
    "| política | escenario | PnL medio $ | desv. $ | peor $ | % semillas ganadoras | drawdown medio $ | trades | inventario final abs |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]


def main():
    p = argparse.ArgumentParser(description="Evaluación multi-semilla y multi-régimen.")
    p.add_argument("--model", type=Path, action="append", default=[],
                   help="Ruta a un modelo TorchScript (se puede repetir).")
    p.add_argument("--baselines", nargs="*", default=[], choices=list(BASELINES))
    p.add_argument("--scenarios", nargs="*", default=PRACTICE + EXTRA, choices=list(ALL_SCENARIOS))
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=TEST_SEED)
    p.add_argument("--markdown", type=Path, default=None, help="Guardar la tabla en un archivo .md")
    a = p.parse_args()

    policies: dict[str, Policy] = {name: BASELINES[name] for name in a.baselines}
    for path in a.model:
        policies[path.stem] = load_policy(path)
    if not policies:
        p.error("Indica al menos un --model o un --baselines")

    lines = list(HEADER)
    for name, pol in policies.items():
        lines += summary_table(name, evaluate(pol, a.scenarios, a.seeds, a.base_seed))
    text = "\n".join(lines)
    print(text)
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(text + "\n", encoding="utf-8")
        print(f"\nTabla guardada en {a.markdown}")


if __name__ == "__main__":
    main()
