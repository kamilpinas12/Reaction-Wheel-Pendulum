import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dqn_agent import DQNAgent
from envs.pendulum_env import ReactionWheelEnv
from utils.config_manager import get as cfg_get
from utils.rollout import RolloutData, plot_rollout


def main():
    env = ReactionWheelEnv()
    agent = DQNAgent(env)

    training = cfg_get('training') or {}
    checkpoint = Path(training.get('output_dir', './data')) / 'final.zip'
    if checkpoint.exists():
        agent.load(str(checkpoint))

    obs, _ = env.reset()
    data = RolloutData()
    done = False
    timestep = 0

    while not done:
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        data.add(timestep=timestep, action=float(action), reward=reward, info=info)
        done = terminated or truncated
        timestep += 1

    out_path = plot_rollout(data, Path('./data/agent_rollout.png'))
    print(f'saved agent rollout plot to {out_path}')


if __name__ == '__main__':
    main()