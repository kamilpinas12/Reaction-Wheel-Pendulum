#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import json, numpy as np
from agents.dqn_agent import DQNAgent
from envs.pendulum_env import ReactionWheelEnv
from utils.config_manager import get as cfg_get

def main(steps=200, seed=123):
    env = ReactionWheelEnv()
    agent = DQNAgent(env)
    ckpt = Path(cfg_get('training').get('output_dir', './data')) / 'final.zip'
    if ckpt.exists(): agent.load(str(ckpt))

    npz = np.load('data/agent_matrices.npz', allow_pickle=True)
    layers = [n for n, _, _ in json.loads(npz['meta'].tolist())['layers'] if n.startswith('q_net.q_net.')]

    def mx(obs):
        a = obs.astype(np.float32)
        for i, n in enumerate(layers):
            a = npz[f'{n}.W'] @ a + npz[f'{n}.b']
            if i + 1 < len(layers): a = np.maximum(a, 0)
        return a

    obs, _ = env.reset(seed=seed)
    mismatches = 0
    for t in range(steps):
        a1, _ = agent.predict(obs)
        a2 = int(np.argmax(mx(obs)))
        mismatches += a1 != a2
        obs, r, term, trunc, _ = env.step(a1)
        print(t, int(a1), a2, float(r), 'OK' if a1 == a2 else 'DIFF')
        if term or trunc: break
    print('mismatches:', mismatches)

if __name__ == '__main__': main()
