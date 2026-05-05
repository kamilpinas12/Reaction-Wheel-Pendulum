#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import numpy as np
import torch
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pendulum_env import ReactionWheelEnv


def rec(m, p=""):
    for n, c in m.named_children():
        name = f"{p}.{n}" if p else n
        if isinstance(c, torch.nn.Linear):
            yield name, c
        else:
            yield from rec(c, name)


def main():
    ckpt = Path("data/final.zip")
    out = Path("data/agent_matrices.npz")
    env = ReactionWheelEnv()
    model = DQN.load(str(ckpt), env=env, device="cpu")
    arr = {}
    meta = []
    for name, lin in rec(model.policy):
        W = lin.weight.detach().cpu().numpy()
        arr[f"{name}.W"] = W
        if lin.bias is not None:
            arr[f"{name}.b"] = lin.bias.detach().cpu().numpy()
        meta.append((name, [int(x) for x in W.shape], [] if lin.bias is None else [int(x) for x in lin.bias.shape]))

    meta_dict = {
        "layers": meta,
        "obs": [int(x) for x in env.observation_space.shape],
        "n_actions": int(getattr(env.action_space, "n", None)) if getattr(env.action_space, "n", None) is not None else None,
    }
    arr["meta"] = np.array(json.dumps(meta_dict))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arr)
    print("saved", out)


if __name__ == "__main__":
    main()
