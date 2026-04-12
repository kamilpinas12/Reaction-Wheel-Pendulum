import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

class ReactionWheelEnv(gym.Env):
    def __init__(self):
        super(ReactionWheelEnv, self).__init__()

        self.dt = 0.02
        self.u_max = 1.0
        self.du_max = 0.15  # Increased from 0.08 to allow more aggressive control
        self.max_episode_steps = 500
        self.step_count = 0
        self.prev_u = 0.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Constants from the nonlinear model
        self.A = 0.1167
        self.B = -3.915
        self.C = 0.08
        self.D = 0.00229
        self.K = 484.73
        
        self.state = None

    @staticmethod
    def _angle_normalize(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_u = 0.0

        theta0 = self._angle_normalize(self.np_random.uniform(-0.25, 0.25))
        theta_dot0 = self.np_random.uniform(-0.1, 0.1)
        phi0 = self.np_random.uniform(-0.1, 0.1)

        # State: [theta, theta_dot, phi]
        self.state = np.array([theta0, theta_dot0, phi0], dtype=np.float32)
        return self._get_observation(), {}

    def _get_observation(self):
        theta, theta_dot, phi = self.state
        return np.array([theta, theta_dot, phi, self.prev_u], dtype=np.float32)

    def _dynamics(self, state, u):
        theta, theta_dot, phi = state

        phi_dot = self.K * (u - self.D * phi)
        theta_ddot = -self.A * theta_dot + self.B * np.sin(theta) + self.C * phi_dot

        return np.array([theta_dot, theta_ddot, phi_dot], dtype=np.float32)

    def _rk4_step(self, state, u):
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self._dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self._dynamics(state + self.dt * k3, u)

        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action):
        self.step_count += 1

        # Policy command (normalized to [-1, 1]) and physical command scaling.
        u_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.u_max

        # Slew-rate limiter to prevent direct min-max switching.
        u_low = self.prev_u - self.du_max
        u_high = self.prev_u + self.du_max
        u = float(np.clip(u_cmd, u_low, u_high))

        self.state = self._rk4_step(self.state, u).astype(np.float32)
        self.state[0] = self._angle_normalize(self.state[0])

        new_theta, new_theta_dot, new_phi = self.state
        delta_u = u - self.prev_u
        self.prev_u = u

        # Reward: maximize uprightness (theta=pi is up) and stabilize around theta=pi.
        upright_reward = np.cos(new_theta + np.pi)  # cos(theta+pi) = -cos(theta), max at theta=pi
        # Reduced penalties to allow learning; can increase later.
        shaping_penalty = 0.01 * new_theta_dot**2 + 0.001 * new_phi**2 + 0.001 * u**2 + 0.05 * delta_u**2
        near_upright_bonus = 2.0 if abs(self._angle_normalize(new_theta - np.pi)) < 0.3 and abs(new_theta_dot) < 1.5 else 0.0
        reward = upright_reward - shaping_penalty + near_upright_bonus
        
        # Safety termination only for clearly divergent trajectories.
        terminated = bool(abs(new_phi) > 150 or abs(new_theta_dot) > 30)
        truncated = self.step_count >= self.max_episode_steps

        info = {"u_cmd": u_cmd, "u_applied": u, "delta_u": delta_u}
        return self._get_observation(), reward, terminated, truncated, info


class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        if isinstance(infos, dict):
            infos = [infos]

        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info["r"]))
        return True


def plot_episode_rewards(episode_rewards):
    if len(episode_rewards) == 0:
        print("Brak zarejestrowanych nagrod epizodow.")
        return

    window = min(50, len(episode_rewards))
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    figure, axes = plt.subplots(1, 1, figsize=(10, 4))
    axes.plot(episode_rewards, alpha=0.4, label="reward/episode")
    axes.plot(np.arange(window - 1, len(episode_rewards)), moving_avg, linewidth=2.0, label=f"moving avg ({window})")
    axes.set_title("Learning curve")
    axes.set_xlabel("episode")
    axes.set_ylabel("return")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = "learning_curve.png"
        figure.savefig(output_path, dpi=150)
        print(f"Wykres uczenia zapisany do pliku: {output_path}")
        plt.close(figure)
    else:
        plt.show()


def export_actor_to_onnx(model, env, output_path="ddpg_actor.onnx"):
    try:
        actor = model.policy.actor.eval()
        # Move to CPU for stable ONNX export (avoid GPU/CPU device mismatch).
        actor = actor.to("cpu")
        obs_dim = int(env.observation_space.shape[0])
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device="cpu")

        torch.onnx.export(
            actor,
            dummy_obs,
            output_path,
            input_names=["obs"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,
            export_params=True,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"Actor exported to ONNX: {output_path}")
    except Exception as exc:
        print(f"ONNX export failed: {exc}")

# 3. Training Script
os.makedirs("training_logs", exist_ok=True)
env = Monitor(ReactionWheelEnv(), filename="training_logs/monitor.csv")
reward_callback = EpisodeRewardCallback()

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
if use_cuda:
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, falling back to CPU.")

# Create Ornstein-Uhlenbeck action noise for smooth exploration
n_actions = env.action_space.shape[0]
action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.2 * np.ones(n_actions)
)

# Initialize DDPG with action noise
model = DDPG(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    device=device,
    action_noise=action_noise
)

# Swing-up usually needs substantially more interaction steps than balancing.
try:
    model.learn(total_timesteps=200000, callback=reward_callback)
except KeyboardInterrupt:
    print("Training interrupted by user. Saving current model...")
    model.save("ddpg_reaction_wheel_interrupted")
except Exception as exc:
    print(f"Training failed: {exc}. Saving current model...")
    model.save("ddpg_reaction_wheel_failed")
    raise
finally:
    # Always keep a latest snapshot, even for interrupted/failed runs.
    model.save("ddpg_reaction_wheel_latest")

# Plot training progress (episode return).
plot_episode_rewards(reward_callback.episode_rewards)

# Save the nominal successful model filename as well.
model.save("ddpg_reaction_wheel")

# Export actor policy for MATLAB (ONNX import).
export_actor_to_onnx(model, env)


def test_policy(model, env, steps=1000):
    observation, _ = env.reset()
    dt = env.unwrapped.dt

    times = []
    thetas = []
    theta_dots = []
    phis = []
    commanded_actions = []
    applied_actions = []

    for step_index in range(steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        times.append(step_index * dt)
        thetas.append(observation[0])
        theta_dots.append(observation[1])
        phis.append(observation[2])
        commanded_actions.append(float(action[0]))
        applied_actions.append(float(info.get("u_applied", action[0])))

        if terminated or truncated:
            break

    figure, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(times, thetas)
    axes[0].set_ylabel("theta [rad]")
    axes[0].grid(True)

    axes[1].plot(times, theta_dots)
    axes[1].set_ylabel("theta_dot [rad/s]")
    axes[1].grid(True)

    axes[2].plot(times, phis)
    axes[2].set_ylabel("phi [rad]")
    axes[2].grid(True)

    axes[3].plot(times, commanded_actions)
    axes[3].set_ylabel("action cmd")
    axes[3].grid(True)

    axes[4].plot(times, applied_actions)
    axes[4].set_ylabel("action applied")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(True)

    figure.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = "state_trajectories.png"
        figure.savefig(output_path, dpi=150)
        print(f"Wykres zapisany do pliku: {output_path}")
        plt.close(figure)
    else:
        plt.show()


test_policy(model, env)