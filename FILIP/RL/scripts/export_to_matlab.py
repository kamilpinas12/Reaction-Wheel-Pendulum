
from pathlib import Path
import sys
import json
import numpy as np

try:
    from scipy.io import savemat
except Exception:
    savemat = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pendulum_env import ReactionWheelEnv


def main():
    npz_path = Path('data/agent_matrices.npz')
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path}; run export_agent_matrices.py first")

    npz = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(npz['meta'].tolist())
    layers = [n for n,_,_ in meta['layers'] if n.startswith('q_net.q_net.')]

    agent = {}
    for i, name in enumerate(layers, start=1):
        agent[f'W{i}'] = npz[f'{name}.W']
        agent[f'b{i}'] = npz[f'{name}.b']

    env = ReactionWheelEnv()
    obs_seed, _ = env.reset(seed=123)
    torques = env.torques

    out_dir = Path('matlab')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mat = out_dir / 'agent_weights.mat'

    if savemat is not None:
        savemat(str(out_mat), {'agent': agent, 'obs_seed': obs_seed, 'torques': torques})
        print('saved', out_mat)
    else:
        out_npz = out_mat.with_suffix('.npz')
        np.savez_compressed(str(out_npz), **agent, obs_seed=obs_seed, torques=torques)
        print('scipy not available — saved', out_npz)


if __name__ == '__main__':
    main()
