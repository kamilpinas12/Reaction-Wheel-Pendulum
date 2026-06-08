from pathlib import Path
from importlib.util import find_spec
import os
import shutil

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

from utils.common import setup_file_logger
from utils.config_manager import cfg_get
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import *
from reac_wheel_sim.reward_wrappers import create_reward_wrapper


def make_env(repeat_num, reward_type, **reward_params):
    def _init():
        env = ReactionWheelEnv("config_td3.yaml", render_mode="rgb_array")
        env = ParamRandomizationWrapper(env)
        env = TrigAndNormalizationObservationWrapper(env)
        # FIX: Reward wrapper must come BEFORE ActionRepeatWrapper
        env = create_reward_wrapper(env, reward_type=reward_type, **reward_params)
        env = ActionRepeatWrapper(env, repeat=repeat_num)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return _init


def main():
    config_name = "config_td3.yaml"
    output_dir = Path(cfg_get("base_agent.output_dir", config_name, default="./RL/logs/td3_train/"))
    num_envs = cfg_get("env.num_envs", config_name, default=1)
    reward_type = cfg_get("reward.type", config_name, default="simple")
    reward_params = cfg_get("reward.params", config_name, default={}) or {}
    writer_enable = cfg_get("base_agent.writer_enable", config_name, default=False)
    repeat_num = cfg_get("td3_agent.repeat_num", config_name, default=1)
    total_timesteps = cfg_get("base_agent.total_timesteps", config_name, default=1_000_000)
    eval_interval = cfg_get("base_agent.eval.interval", config_name, default=50_000)
    policy_name = cfg_get("td3_agent.policy", config_name, default="MlpPolicy")
    net_arch = cfg_get("td3_agent.net_arch", config_name, default=[256, 256])
    activation_name = cfg_get("td3_agent.activation", config_name, default="relu").lower()
    activation_fn = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}.get(activation_name, torch.nn.ReLU)
    action_noise_std = cfg_get("td3_agent.action_noise_std", config_name, default=0.1)
    buffer_size = cfg_get("td3_agent.buffer_size", config_name, default=1_000_000)
    learning_starts = cfg_get("td3_agent.learning_starts", config_name, default=10_000)
    batch_size = cfg_get("td3_agent.batch_size", config_name, default=512)
    train_freq = cfg_get("td3_agent.train_freq", config_name, default=1)
    gradient_steps = cfg_get("td3_agent.gradient_steps", config_name, default=1)
    tau = cfg_get("td3_agent.tau", config_name, default=0.005)
    policy_delay = cfg_get("td3_agent.policy_delay", config_name, default=2)
    target_policy_noise = cfg_get("td3_agent.target_policy_noise", config_name, default=0.2)
    target_noise_clip = cfg_get("td3_agent.target_noise_clip", config_name, default=0.5)
    learning_rate = cfg_get("base_agent.learning_rate", config_name, default=3e-4)
    gamma = cfg_get("base_agent.gamma", config_name, default=0.999)
    seed = cfg_get("base_agent.seed", config_name, default=42)
    device = cfg_get("base_agent.device", config_name, default="cpu")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_file_logger("Train_TD3", output_dir / "log.log")
    train_env = make_vec_env(make_env(repeat_num, reward_type, **reward_params), n_envs=num_envs)
    eval_env = make_env(repeat_num, reward_type, **reward_params)()
    
    record_video = cfg_get("base_agent.eval.record", config_name, default=False)
    if record_video:
        eval_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder=str(output_dir / "videos"),
            episode_trigger=lambda episode: True,
            name_prefix="td3_train",
        )

    act_size = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(act_size), sigma=action_noise_std * np.ones(act_size))
    
    # FIX 1: Adjust evaluation frequency to account for vectorized parallel steps
    adjusted_eval_freq = max(1, eval_interval // num_envs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir / "best_models",
        log_path=output_dir,
        eval_freq=adjusted_eval_freq,
        deterministic=True,
        render=False,
    )

    noise_initial_std = cfg_get("td3_agent.action_noise_initial_std", config_name, default=action_noise_std)
    noise_final_std = cfg_get("td3_agent.action_noise_final_std", config_name, default=0.0)
    noise_decay_timesteps = cfg_get("td3_agent.action_noise_decay_timesteps", config_name, default=total_timesteps)

    class ActionNoiseDecayCallback(BaseCallback):
        def __init__(self, noise, init_std, final_std, decay_timesteps, verbose=0):
            super().__init__(verbose)
            self.noise = noise
            self.init = float(init_std)
            self.final = float(final_std)
            self.decay = int(decay_timesteps) if decay_timesteps > 0 else 1

        def _on_step(self) -> bool:
            t = getattr(self.model, 'num_timesteps', 0)
            frac = min(1.0, t / float(self.decay))
            sigma = self.init * (1.0 - frac) + self.final * frac
            try:
                self.noise.sigma = np.ones_like(self.noise.sigma) * sigma
            except Exception:
                try:
                    self.noise.sigma[:] = sigma
                except Exception:
                    pass
            return True

    # FIX 2: Dynamic separate optimizer adjustment hook inside training initialization
    actor_lr = cfg_get("td3_agent.actor_lr", config_name, default=None)
    critic_lr = cfg_get("td3_agent.critic_lr", config_name, default=None)

    class CustomOptimizerSetupCallback(BaseCallback):
        def __init__(self, a_lr, c_lr):
            super().__init__()
            self.a_lr = a_lr
            self.c_lr = c_lr
            self.applied = False

        def _on_step(self) -> bool:
            if not self.applied:
                # This executes on step 1 after SB3 has successfully built the internal optimizers
                if self.a_lr is not None and hasattr(self.model, 'actor') and self.model.actor.optimizer is not None:
                    for g in self.model.actor.optimizer.param_groups:
                        g['lr'] = float(self.a_lr)
                if self.c_lr is not None and hasattr(self.model, 'critic') and self.model.critic.optimizer is not None:
                    for g in self.model.critic.optimizer.param_groups:
                        g['lr'] = float(self.c_lr)
                self.applied = True
            return True

    policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn)
    tensorboard_log = None
    if writer_enable and find_spec("tensorboard") is not None:
        tensorboard_log = str(output_dir / "tensorboard")
    elif writer_enable:
        logger.warning("TensorBoard is not installed; continuing without tensorboard logging.")

    model = TD3(
        policy_name,
        train_env,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tau=tau,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log,
    )

    callbacks = [eval_callback]
    if noise_decay_timesteps > 0:
        noise_cb = ActionNoiseDecayCallback(action_noise, noise_initial_std, noise_final_std, noise_decay_timesteps)
        callbacks.append(noise_cb)
        
    if actor_lr is not None or critic_lr is not None:
        callbacks.append(CustomOptimizerSetupCallback(actor_lr, critic_lr))

    logger.info("Starting TD3 training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(output_dir / "td3_sb3_reaction_wheel")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()