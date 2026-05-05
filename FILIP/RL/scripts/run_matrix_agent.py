#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import json, numpy as np
from envs.pendulum_env import ReactionWheelEnv

F = Path("data/agent_matrices.npz")
if not F.exists():
    raise SystemExit(f"missing {F} — run export_agent_matrices.py first")

npz = np.load(str(F), allow_pickle=True)
meta = json.loads(npz["meta"].tolist())
layers = [n for n, _, _ in meta["layers"] if n.startswith("q_net.q_net.")]

def forward(x):
    a = x.astype(np.float32)
    for i, name in enumerate(layers):
        W = npz[f"{name}.W"]
        b = npz[f"{name}.b"]
        a = W @ a + b
        if i != len(layers)-1:
            a = np.maximum(0, a)
    return a

def main(steps=200):
    env = ReactionWheelEnv()
    obs, _ = env.reset()
    for t in range(steps):
        logits = forward(obs)
        act = int(np.argmax(logits))
        obs, r, term, trunc, info = env.step(act)
        print(t, act, float(r))
        if term or trunc:
            break

if __name__ == '__main__':
    main()
