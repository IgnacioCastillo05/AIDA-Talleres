"""Gráfica de la curva de validación que guarda train.py (outputs/model.csv).

    python -m student_work.plot_training --csv outputs/model.csv --out outputs/training_curve.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("outputs/model.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/training_curve.png"))
    a = p.parse_args()

    with a.csv.open() as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    pnl_cols = [c for c in rows[0] if c.startswith("pnl_")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(steps, [float(r["val_score"]) for r in rows], color="black", lw=2)
    best = max(rows, key=lambda r: float(r["val_score"]))
    ax1.axvline(int(best["step"]), ls="--", color="gray")
    ax1.set_ylabel("score robusto (validación)")
    ax1.set_title("Curva de validación (semillas 5000+, nunca usadas para entrenar)")
    for c in pnl_cols:
        ax2.plot(steps, [float(r[c]) for r in rows], label=c.removeprefix("pnl_"))
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("PnL medio por régimen ($)")
    ax2.set_xlabel("pasos de entrenamiento")
    ax2.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
