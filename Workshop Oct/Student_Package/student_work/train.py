"""Entrenador mejorado del equipo: Double DQN + Dueling + n-step + regímenes aleatorios
+ recompensa ajustada al riesgo + selección del mejor checkpoint en validación.

Uso (desde la raíz de Student_Package):

    python -m student_work.train                      # configuración final (por defecto)
    python -m student_work.train --steps 5000 --seed 7  # lo que ejecuta START_HERE.bat opción 7

Cada mejora se puede apagar con una bandera para hacer ablaciones, por ejemplo:

    python -m student_work.train --algo dqn --arch mlp --reward basic --regimes practice \\
        --n-step 1 --out outputs/ablation_starter_like.pt

El trainer original del paquete se conserva en student_work/train_starter.py.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared.contract import ACTION_SIZE
from student_work.envs import ALL_SCENARIOS, PRACTICE_SCENARIOS, TrainingMarketEnv, random_scenario
from student_work.evaluate import EXTRA, PRACTICE, VALIDATION_SEED, evaluate, robust_score, torch_policy
from student_work.model import build_model


class Replay:
    def __init__(self, capacity=100_000):
        self.data = deque(maxlen=capacity)
    def add(self, *row): self.data.append(row)
    def sample(self, n):
        batch = random.sample(self.data, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )
    def __len__(self): return len(self.data)


class NStepBuffer:
    """Convierte transiciones de 1 paso en transiciones de n pasos:
    R = r_t + g*r_{t+1} + ... + g^(n-1)*r_{t+n-1}, y el "siguiente estado" es s_{t+n}.
    La recompensa (PnL) llega con mucho ruido; sumar varios pasos propaga la
    señal más rápido que el bootstrap de un solo paso."""
    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.buf: deque = deque()

    def push(self, s, a, r, ns, done) -> list[tuple]:
        self.buf.append((s, a, r, ns, done))
        out = []
        if len(self.buf) >= self.n:
            out.append(self._make())
            self.buf.popleft()
        if done:  # al final del episodio vaciamos lo que quede
            while self.buf:
                out.append(self._make())
                self.buf.popleft()
        return out

    def _make(self):
        s, a = self.buf[0][0], self.buf[0][1]
        ret, ns, done = 0.0, self.buf[-1][3], self.buf[-1][4]
        for k, (_, _, r, nsk, dk) in enumerate(self.buf):
            ret += (self.gamma ** k) * r
            if dk:
                ns, done = nsk, True
                break
        return s, a, ret, ns, done


def make_env(episode: int, seed: int, regimes: str, rng: np.random.Generator,
             reward: str, risk_aversion: float) -> TrainingMarketEnv:
    kw = dict(steps=250, seed=seed + episode, reward_mode=reward, risk_aversion=risk_aversion)
    if regimes == "practice":
        name = list(PRACTICE_SCENARIOS)[episode % len(PRACTICE_SCENARIOS)]
        return TrainingMarketEnv.named(name, **kw)
    # "randomized": mitad regímenes con nombre, mitad mercados aleatorios.
    if rng.random() < 0.5:
        name = str(rng.choice(list(ALL_SCENARIOS)))
        return TrainingMarketEnv.named(name, **kw)
    return TrainingMarketEnv(random_scenario(rng), name="random", **kw)


def train(a: argparse.Namespace):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    q = build_model(a.arch)
    target = build_model(a.arch)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=a.lr)
    replay = Replay()
    gamma_n = a.gamma ** a.n_step
    log_rows = []
    best_score, best_step = -float("inf"), 0
    global_step = 0
    episode = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)

    while global_step < a.steps:
        env = make_env(episode, a.seed, a.regimes, rng, a.reward, a.risk_aversion)
        nstep = NStepBuffer(a.n_step, a.gamma)
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done and global_step < a.steps:
            epsilon = max(a.eps_min, 1.0 - global_step / max(1, a.steps * a.eps_fraction))
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    action = int(q(torch.as_tensor(state).unsqueeze(0)).argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            for tr in nstep.push(state, action, reward * a.reward_scale, next_state, done):
                replay.add(*tr)
            state = next_state
            ep_reward += reward
            global_step += 1

            if len(replay) >= a.warmup and global_step % a.train_every == 0:
                s, act, r, ns, d = replay.sample(a.batch_size)
                pred = q(s).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if a.algo == "double":
                        # Double DQN: la red online ELIGE la acción, la red target la EVALÚA.
                        # Reduce la sobreestimación típica del max() con recompensas ruidosas.
                        best_a = q(ns).argmax(dim=1, keepdim=True)
                        nxt = target(ns).gather(1, best_a).squeeze(1)
                    else:
                        nxt = target(ns).max(dim=1).values
                    y = r + gamma_n * (1.0 - d) * nxt
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                # Actualización "suave" de la red target (Polyak): más estable que copiar cada N pasos.
                with torch.no_grad():
                    for pt, po in zip(target.parameters(), q.parameters()):
                        pt.mul_(1.0 - a.tau).add_(po, alpha=a.tau)

            if global_step % a.eval_every == 0 or global_step == a.steps:
                # Validación con semillas que NUNCA se usan para entrenar ni para el test final.
                q.eval()
                res = evaluate(torch_policy(q), PRACTICE + EXTRA, a.val_seeds, VALIDATION_SEED)
                q.train()
                score = robust_score(res)
                row = {"step": global_step, "epsilon": round(epsilon, 3), "val_score": round(score, 2)}
                row.update({f"pnl_{k}": round(float(v["pnl"].mean()), 2) for k, v in res.items()})
                log_rows.append(row)
                tag = ""
                if score > best_score:
                    best_score, best_step = score, global_step
                    torch.jit.script(q.eval()).save(str(a.out))
                    q.train()
                    tag = "  <- mejor hasta ahora (guardado)"
                print(f"[val] step={global_step:7d} eps={epsilon:.2f} score={score:9.1f} "
                      + " ".join(f"{k}={v['pnl'].mean():7.0f}" for k, v in res.items()) + tag)
        episode += 1
        if episode % a.log_episodes == 0:
            print(f"episode={episode:04d} scenario={env.scenario:13s} steps={global_step:7d}/{a.steps} "
                  f"reward={ep_reward:9.2f} nav=${info['nav_cents']/100:,.2f}")

    log_path = a.out.with_suffix(".csv")
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader(); w.writerows(log_rows)
    print(f"Best validation score {best_score:.1f} at step {best_step}")
    print(f"Saved TorchScript policy to: {a.out}")
    print(f"Validation curve saved to: {log_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the improved team DQN.")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("outputs/model.pt"))
    p.add_argument("--algo", choices=["dqn", "double"], default="double")
    p.add_argument("--arch", choices=["mlp", "dueling"], default="dueling")
    p.add_argument("--reward", choices=["basic", "risk"], default="risk")
    p.add_argument("--regimes", choices=["practice", "randomized"], default="randomized")
    p.add_argument("--risk-aversion", type=float, default=0.01)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--eps-fraction", type=float, default=0.5)
    p.add_argument("--eps-min", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--val-seeds", type=int, default=4)
    p.add_argument("--log-episodes", type=int, default=20)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
