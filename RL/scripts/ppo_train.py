import gymnasium as gym
import numpy as np
from pathlib import Path
import shutil
import os

from gymnasium.wrappers import RecordVideo

from agents.ppo_agent import PPOAgent
from agents.base_agent import BaseAgent
from nets.a2c import ModelA2C

from utils.callbacks import *
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import *
from reac_wheel_sim.reward_wrappers import create_reward_wrapper
from utils.config_manager import cfg_get
from utils.common import setup_file_logger
from utils.custom_paths import MODELS_DIR, CONFIGS_DIR, LOGS_DIR

def make_env(repeat_num, reward_type, **reward_params):
    def _init():
        env = ReactionWheelEnv("config_ppo.yaml", render_mode="rgb_array")
        env = ParamRandomizationWrapper(env)
        env = TrigAndNormalizationObservationWrapper(env)
        env = ActionRepeatWrapper(env, repeat=repeat_num)
        env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env
    return _init


def record_video(agent: BaseAgent, video_dir):
    env = ReactionWheelEnv("config_ppo.yaml", render_mode="rgb_array")
    env = TrigAndNormalizationObservationWrapper(env)
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode: True,
        name_prefix="ppo_eval",
    )
    try:
        obs, _ = env.reset(seed=0)
        done = False
        while not done:
            action, _  = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated
    finally:
        env.close()


def main():
    config_name = "config_ppo.yaml"
    num_envs = cfg_get('env.num_envs', config_name, default=4)
    reward_type = cfg_get('reward.type', config_name, default="simple")
    reward_params = cfg_get('reward.params', config_name, default={}) or {}
    model_hid_size = cfg_get('ppo_agent.hid_size', config_name, default=32)
    model_activation = cfg_get('ppo_agent.activation', config_name, default="tanh")
    repeat_num = cfg_get('ppo_agent.repeat_num', config_name, default=5)
    output_dir = Path(cfg_get('base_agent.output_dir', config_name))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_config_path = output_dir / "train_config.yaml"
    shutil.copy(CONFIGS_DIR / config_name, target_config_path)
    logger = setup_file_logger("Train_PPO", output_dir / "log.log")

    logger.info(f"Creating {num_envs} vectorized environments...")
    logger.info(f"Reward: {reward_type}, params: {reward_params}")
    env_fns = [make_env(repeat_num, reward_type, **reward_params) for _ in range(num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    eval_env = make_env(repeat_num, reward_type, **reward_params)()

    obs_size = envs.single_observation_space.shape[0]
    act_size = envs.single_action_space.shape[0]

    model = ModelA2C(obs_size=obs_size, act_size=act_size, hid_size=model_hid_size, activation=model_activation)
    agent = PPOAgent(env=envs, model=model, logger=logger)
    agent.load("/home/igorsiata/studia/Reaction-Wheel-Pendulum/RL/results/ppo/ppo_best_small/model.pth")

    agent.train(eval_env=eval_env)
    logger.info('Training finished')

    agent.evaluate(eval_env, False)

    video_dir = output_dir / "videos"
    record_video(agent, video_dir)

    save_path = output_dir / "model.pth"
    agent.save(save_path)
    envs.close()

if __name__ == "__main__":
    main()
