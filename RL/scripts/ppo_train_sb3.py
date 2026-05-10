import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch

from pathlib import Path
from utils.common import setup_file_logger
from agents.base_agent import BaseAgent
from utils.rollout_buffer import RolloutBuffer
from utils.config_manager import cfg_get
from utils.callbacks import *
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import *
from reac_wheel_sim.reward_wrappers import create_reward_wrapper


def make_env(reward_type, **reward_params):
    def _init():
        # env = ReactionWheelEnv()
        # env = RandomInitialStateWrapper(env, [np.pi, 5.0, 50.0])
        # env = TrigObservationWrapper(env)
        # env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)

        env = gym.make("Pendulum-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        return env
    return _init

def main_sb3():
    config_name = "config_ppo.yaml"
    output_dir = Path(cfg_get('base_agent.output_dir', config_name))
    logger = setup_file_logger("Train_PPO", output_dir / "log.log")

    num_envs = cfg_get('env.num_envs', config_name, default=4)
    reward_type = cfg_get('reward.type', config_name, default="simple")
    reward_params = cfg_get('reward.params', config_name, default={}) or {}
    eval_interval = cfg_get('base_agent.eval.interval', config_name, default=5000)
    
    # SB3 ma własne świetne narzędzia do wektoryzacji.
    # Przekazujemy mu Twoją funkcję _init z make_env.
    env = make_vec_env(
        make_env(reward_type, **reward_params), 
        n_envs=num_envs
    )
    
    # Środowisko ewaluacyjne (1 instancja)
    eval_env = make_env(reward_type, **reward_params)()
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path= output_dir / "best_models",
        log_path= output_dir, 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )

    # plot_cb = RewardPlottingCallback(plot_freq=eval_interval, plot_dir=output_dir / "plots")
    rollout_cb = RolloutCallback(eval_env, eval_interval, output_dir / "plots")
    callbacks = [eval_callback, rollout_cb]

    # Ustawienie hiperparametrów i architektury (zrównaj ze swoimi!)
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh, # lub ReLU
        net_arch=[64, 64]
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1, 
        use_sde=True,
        sde_sample_freq=4,
        device="cpu",
        tensorboard_log="./tensorboard_logs/SB3_baseline/"
    )

    print("Rozpoczynam trening Stable Baselines3...")
    model.learn(total_timesteps=1_000_000, callback=callbacks)
    model.save("ppo_sb3_reaction_wheel")

if __name__ == "__main__":
    main_sb3()