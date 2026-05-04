#!/usr/bin/env python3
import numpy as np
from pathlib import Path
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

F_mat = Path('matlab/agent_weights.mat')
F_npz = Path('data/agent_matrices.npz')

def load_agent():
    if loadmat and F_mat.exists():
        d = loadmat(str(F_mat), squeeze_me=True, struct_as_record=False)
        a = d['agent']
        return a, d.get('obs_seed'), d.get('torques')
    if F_npz.exists():
        npz = np.load(str(F_npz), allow_pickle=True)
        # build a simple dict with W1..W3,b1..b3
        meta = __import__('json').loads(npz['meta'].tolist())
        layers = [n for n,_,_ in meta['layers'] if n.startswith('q_net.q_net.')]
        out = {}
        for i, name in enumerate(layers, start=1):
            out[f'W{i}'] = npz[f'{name}.W']
            out[f'b{i}'] = npz[f'{name}.b']
        env = __import__('envs.pendulum_env', fromlist=['ReactionWheelEnv']).ReactionWheelEnv()
        obs_seed, _ = env.reset(seed=123)
        return out, obs_seed, env.torques
    raise SystemExit('no agent weights found')

def forward1(x, a):
    x = x.reshape(-1)
    def g(name):
        return a[name] if isinstance(a, dict) else getattr(a, name)
    W1 = g('W1'); W2 = g('W2'); W3 = g('W3')
    b1 = np.asarray(g('b1')).reshape(-1)
    b2 = np.asarray(g('b2')).reshape(-1)
    b3 = np.asarray(g('b3')).reshape(-1)
    z1 = W1.dot(x) + b1
    a1 = np.maximum(z1, 0)
    z2 = W2.dot(a1) + np.asarray(b2).reshape(-1)
    a2 = np.maximum(z2, 0)
    z3 = W3.dot(a2) + np.asarray(b3).reshape(-1)
    return z3

def main():
    a, obs_seed, torques = load_agent()
    obs_seed = np.asarray(obs_seed).reshape(-1)
    cases = [obs_seed, obs_seed + np.array([0.05,0,0,0], dtype=float), obs_seed + np.array([0,-0.05,0.02,0], dtype=float)]
    for i, o in enumerate(cases, 1):
        logits = forward1(o, a)
        act = int(np.argmax(logits))
        torque = float(torques[act])
        print(f'case {i}: action={act} torque={torque}')

if __name__ == '__main__':
    main()
