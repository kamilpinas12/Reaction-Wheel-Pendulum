import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo

from nets.a2c import ModelA2C
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import (
    ActionRepeatWrapper,
    ParamRandomizationWrapper,
    TrigAndNormalizationObservationWrapper,
)
from reac_wheel_sim.reward_wrappers import create_reward_wrapper
from utils.common import angle_normalize, setup_file_logger
from utils.config_manager import cfg_get
from utils.custom_paths import LOGS_DIR
from utils.rollout import RolloutData, plot_rollout


def parse_float_list(raw: Optional[str], default: List[float]) -> List[float]:
    if raw is None:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return default
    return [float(p) for p in parts]


def build_env(
    config_name: str,
    param_noise_pct: float,
    max_mass_pos_pct: float,
    repeat_num: int,
    reward_type: str,
    reward_params: Dict[str, float],
    enable_reward: bool,
    render_mode: Optional[str] = None,
) -> gym.Env:
    env = ReactionWheelEnv(config_name, render_mode=render_mode)
    env = ParamRandomizationWrapper(
        env,
        param_noise_pct=param_noise_pct,
        mass_pos_pct=max_mass_pos_pct,
    )
    env = TrigAndNormalizationObservationWrapper(env)
    # if repeat_num > 1:
    #     env = ActionRepeatWrapper(env, repeat=repeat_num)
    if enable_reward:
        env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def load_model(
    checkpoint_path: Path,
    obs_size: int,
    act_size: int,
    hid_size: int,
    device: str,
) -> ModelA2C:
    model = ModelA2C(obs_size=obs_size, act_size=act_size, hid_size=hid_size).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_action(model: ModelA2C, obs: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.FloatTensor(obs).to(device)
    if obs_tensor.ndim == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    with torch.no_grad():
        mu, var, _ = model(obs_tensor)
    action = mu
    return action.cpu().numpy()


def run_episode(
    env: gym.Env,
    model: ModelA2C,
    device: str,
    initial_state: np.ndarray,
    mass_shift_pct: float,
    param_noise_pct: float,
) -> Tuple[Dict[str, float], RolloutData]:
    options = {
        "initial_state": initial_state,
        "mass_shift_pct": float(mass_shift_pct),
        "param_noise_pct": float(param_noise_pct),
    }
    obs, info = env.reset(options=options)

    data = RolloutData()
    done = False
    step = 0
    pos_err_sum = 0.0
    energy_sum = 0.0

    while not done:
        action = predict_action(model, obs, device)
        obs, reward, terminated, truncated, info = env.step(action[0])
        done = terminated or truncated
        step += 1
        theta = float(info.get("theta", 0.0))
        theta_dot = float(info.get("theta_dot", 0.0))
        u_cmd = float(info.get("u_cmd", 0.0))
        theta_err = float(angle_normalize(theta - np.pi))

        pos_err_sum += theta_err ** 2
        energy_sum += abs(u_cmd)

        data.add(timestep=step, action=float(action[0][0]), reward=reward, info=info)

    metrics = {
        "pos_err_sum": float(pos_err_sum),
        "energy_sum": float(energy_sum),
    }
    metrics.update({
        "mass_shift_pct": float(mass_shift_pct),
        "param_noise_pct": float(param_noise_pct),
    })
    if "physical_params" in info:
        metrics["physical_params"] = [float(x) for x in info["physical_params"]]
    if "ground_truth_params" in info:
        metrics["ground_truth_params"] = [float(x) for x in info["ground_truth_params"]]
    return metrics, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO agent on reaction wheel pendulum")
    parser.add_argument("--checkpoint", required=True, help="Path to PPO checkpoint (.pth)")
    parser.add_argument(
        "--output-dir",
        default=str(LOGS_DIR / "model_evals" / "ppo_indx"),
        help="Output directory for evaluation results",
    )
    parser.add_argument("--config", default="config_ppo.yaml", help="Config file name")
    parser.add_argument("--device", default=None, help="Override device from config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--mass-positions",
        default=None,
        help="Comma-separated mass shift values in [0,1]",
    )
    parser.add_argument("--param-noise-pct", type=float, default=0.2, help="Param noise pct")
    parser.add_argument("--max-mass-pos-pct", type=float, default=1.0, help="Max mass shift pct")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--no-plots", action="store_true", help="Disable rollout plots")
    parser.add_argument("--no-reward", action="store_true", help="Disable reward wrapper")
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
    plots_dir = run_dir / "plots"
    videos_dir = run_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger("PPO_Eval", run_dir / "eval.log")
    logger.info("Starting PPO evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    mass_positions = parse_float_list(args.mass_positions, [0.0, 0.33, 0.66, 1.0])
    mass_positions = [float(np.clip(x, 0.0, 1.0)) for x in mass_positions]

    reward_type = cfg_get("reward.type", config_name, default="balanced")
    reward_params = cfg_get("reward.params", config_name, default={}) or {}
    repeat_num = cfg_get("ppo_agent.repeat_num", config_name, default=1)
    hid_size = cfg_get("ppo_agent.hid_size", config_name, default=64)

    base_env = build_env(
        config_name=config_name,
        param_noise_pct=args.param_noise_pct,
        max_mass_pos_pct=args.max_mass_pos_pct,
        repeat_num=repeat_num,
        reward_type=reward_type,
        reward_params=reward_params,
        enable_reward=not args.no_reward,
        render_mode=None,
    )
    obs_size = base_env.observation_space.shape[0]
    act_size = base_env.action_space.shape[0]
    base_env.close()

    model = load_model(
        checkpoint_path=Path(args.checkpoint),
        obs_size=obs_size,
        act_size=act_size,
        hid_size=hid_size,
        device=device,
    )

    initial_states = {
        "down": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "swing_right": np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float32),
        "swing_left": np.array([-np.pi / 2.0, 0.0, 0.0], dtype=np.float32),
    }

    all_metrics: List[Dict[str, float]] = []
    for label, init_state in initial_states.items():
        for mass_shift in mass_positions:
            scenario_name = f"{label}_mass{mass_shift:.2f}"

            render_mode = "rgb_array" if not args.no_video else None
            env = build_env(
                config_name=config_name,
                param_noise_pct=args.param_noise_pct,
                max_mass_pos_pct=args.max_mass_pos_pct,
                repeat_num=repeat_num,
                reward_type=reward_type,
                reward_params=reward_params,
                enable_reward=not args.no_reward,
                render_mode=render_mode,
            )

            if not args.no_video:
                env = RecordVideo(
                    env,
                    video_folder=str(videos_dir),
                    episode_trigger=lambda episode: True,
                    name_prefix=f"eval_{scenario_name}",
                )

            logger.info(f"Running scenario {scenario_name}")
            metrics, rollout = run_episode(
                env=env,
                model=model,
                device=device,
                initial_state=init_state,
                mass_shift_pct=mass_shift,
                param_noise_pct=args.param_noise_pct,
            )
            metrics["initial_state"] = label
            metrics["output_dir"] = str(run_dir)
            all_metrics.append(metrics)

            if not args.no_plots:
                plot_path = plot_rollout(rollout, plots_dir / f"rollout_{scenario_name}.png")
                logger.info(f"Saved rollout plot to {plot_path}")

            env.close()

    summary = {
        "config": config_name,
        "checkpoint": args.checkpoint,
        "mass_positions": mass_positions,
        "initial_states": list(initial_states.keys()),
        "param_noise_pct": args.param_noise_pct,
        "results": all_metrics,
    }

    summary_path = run_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved metrics to {summary_path}")
    logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
