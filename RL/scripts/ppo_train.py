import gymnasium as gym
import numpy as np
from pathlib import Path
import shutil
import os

from agents.ppo_agent import PPOAgent
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
        env = ReactionWheelEnv("config_ppo.yaml")
        env = TrigObservationWrapper(env)
        env = ActionRepeatWrapper(env, repeat=repeat_num)
        env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        return env
    return _init


def main():
    config_name = "config_ppo.yaml"
    num_envs = cfg_get('env.num_envs', config_name, default=4)
    reward_type = cfg_get('reward.type', config_name, default="simple")
    reward_params = cfg_get('reward.params', config_name, default={}) or {}
    model_hid_size = cfg_get('ppo_agent.hid_size', config_name, default=32)
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

    model = ModelA2C(obs_size=obs_size, act_size=act_size, hid_size=model_hid_size)
    agent = PPOAgent(env=envs, model=model, logger=logger)

    agent.train(eval_env)
    logger.info('Training finished')

    agent.evaluate(eval_env, False)

    save_path = LOGS_DIR / "rl_ppo" / "ppo_final_model.pth"
    agent.save(save_path)
    envs.close()

if __name__ == "__main__":
    main()
