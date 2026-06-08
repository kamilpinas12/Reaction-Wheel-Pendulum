import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import TD3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import (
    ActionRepeatWrapper,
    ParamRandomizationWrapper,
    TrigAndNormalizationObservationWrapper,
)
from reac_wheel_sim.reward_wrappers import create_reward_wrapper
from utils.common import setup_file_logger
from utils.config_manager import cfg_get
from utils.rollout import RolloutData, plot_rollout


def build_env(config_name: str) -> gym.Env:
    repeat_num = cfg_get("td3_agent.repeat_num", config_name, default=2)
    reward_type = cfg_get("reward.type", config_name, default="simple")
    reward_params = cfg_get("reward.params", config_name, default={}) or {}

    env = ReactionWheelEnv(config_name, render_mode="rgb_array")
    env = ParamRandomizationWrapper(env)
    env = TrigAndNormalizationObservationWrapper(env)
    env = ActionRepeatWrapper(env, repeat=repeat_num)
    env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)
    return gym.wrappers.RecordEpisodeStatistics(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a TD3 agent and plot states")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the TD3 checkpoint (defaults to the configured output_dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the plot and logs",
    )
    parser.add_argument("--config", default="config_td3.yaml", help="Config file name")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    args = parser.parse_args()

    config_name = args.config
    output_dir = Path(args.output_dir or cfg_get("base_agent.output_dir", config_name, default="./RL/logs/td3_train/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint or (output_dir / "td3_sb3_reaction_wheel.zip"))
    logger = setup_file_logger("TD3_Eval", output_dir / "eval.log")
    logger.info("Starting TD3 rollout")
    logger.info(f"Checkpoint: {checkpoint}")

    env = build_env(config_name)
    model = TD3.load(str(checkpoint), env=env, device=cfg_get("base_agent.device", config_name, default="cpu"))

    obs, _ = env.reset(seed=args.seed)
    data = RolloutData()
    done = False
    timestep = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        data.add(timestep=timestep, action=float(action[0]), reward=reward, info=info)
        done = terminated or truncated
        timestep += 1

    out_path = plot_rollout(data, output_dir / "td3_rollout.png")
    logger.info(f"Saved rollout plot to {out_path}")
    env.close()


if __name__ == "__main__":
    main()