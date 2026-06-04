import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from nets.a2c import ModelA2C
from agents.ppo_agent import PPOAgent
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import (
    ParamRandomizationWrapper,
    TrigAndNormalizationObservationWrapper,
    ActionRepeatWrapper
)
from utils.common import setup_file_logger
from utils.config_manager import cfg_get
from utils.custom_paths import LOGS_DIR


def build_env(
    config_name: str,
    render_mode: Optional[str] = None,
    repeat_num=1,
    **reward_params
) -> gym.Env:
    env = ReactionWheelEnv(config_name, render_mode="rgb_array")
    # env = ParamRandomizationWrapper(env)
    env = TrigAndNormalizationObservationWrapper(env)
    env = ActionRepeatWrapper(env, repeat=repeat_num)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def build_vec_env(config_name: str) -> gym.Env:
    return SyncVectorEnv([
        lambda: build_env(config_name=config_name, render_mode=None)
    ])
 

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on reaction wheel pendulum")
    parser.add_argument("--checkpoint", required=True, help="Path to PPO checkpoint (.pth)")
    parser.add_argument(
        "--output-dir",
        default=str(LOGS_DIR / "model_evals" / "ppo_indx"),
        help="Output directory for evaluation results",
    )
    parser.add_argument("--config", default="config_eval.yaml", help="Config file name")
    parser.add_argument("--device", default=None, help="Override device from config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config_name = args.config
    device = args.device or cfg_get("base_agent.device", config_name, default="cpu")
    seed = args.seed if args.seed is not None else cfg_get("env.seed", config_name, default=None)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    output_root = Path(args.output_dir)
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger("PPO_Eval", run_dir / "eval.log")
    logger.info("Starting PPO evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    hid_size = cfg_get("ppo_agent.hid_size", config_name, default=64)
    repeat_num = cfg_get("ppo_agent.repeat_num", config_name, default=1)
    model_activation = cfg_get('ppo_agent.activation', config_name, default="tanh")

    record_video = cfg_get("base_agent.eval.record", config_name, default=False)
    eval_render_mode = "rgb_array" if record_video else None
    eval_env = build_env(
        config_name=config_name,
        render_mode=eval_render_mode,
        repeat_num=repeat_num
    )
    if record_video:
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder=str(run_dir / "videos"),
            episode_trigger=lambda episode: True,
            name_prefix="ppo_eval",
        )

    obs_size = eval_env.observation_space.shape[0]
    act_size = eval_env.action_space.shape[0]

    model = ModelA2C(obs_size=obs_size, act_size=act_size, hid_size=hid_size, activation=model_activation).to(device)
    agent_env = build_vec_env(config_name=config_name)
    agent = PPOAgent(env=agent_env, model=model, logger=logger, config_name="config_eval.yaml")
    agent.load(args.checkpoint)
    agent.output_dir = run_dir

    mean_pos_err, std_pos_err = agent.evaluate(
        eval_env=eval_env,
        render=False,
    )

    summary = {
        "config": config_name,
        "checkpoint": args.checkpoint,
        "mean_pos_err": float(mean_pos_err),
        "std_pos_err": float(std_pos_err),
        "param_randomization": True,
    }

    summary_path = run_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved metrics to {summary_path}")
    logger.info("Evaluation finished")

    eval_env.close()
    agent_env.close()

if __name__ == "__main__":
    main()
