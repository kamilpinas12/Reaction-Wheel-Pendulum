import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.pendulum_env import ReactionWheelEnv
from utils.rollout import RolloutData, plot_rollout


def main():
    env = ReactionWheelEnv()
    obs, _ = env.reset()

    data = RolloutData()
    done = False
    timestep = 0

    while not done:
        action = 3
        obs, reward, terminated, truncated, info = env.step(action)
        data.add(timestep=timestep, action=action, reward=reward, info=info)
        done = terminated or truncated
        timestep += 1

    out_path = plot_rollout(data, Path('./data/env_rollout.png'))
    print(f'saved env rollout plot to {out_path}')


if __name__ == '__main__':
    main()