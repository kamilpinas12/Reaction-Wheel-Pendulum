import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import torch


class ReactionWheelEnv(gym.Env):
    def __init__(self):
        super(ReactionWheelEnv, self).__init__()

        self.dt = 0.01
        self.u_max = 1.0
        self.du_max = 0.5  # Increased from 0.08 to allow more aggressive control
        self.max_episode_steps = 5000
        self.step_count = 0
        self.prev_u = 0.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.K_sin = -4.097306
        self.K_reac_wheel = -0.008456
        self.K_pend_vel = -0.152144

        self.Ku = 457.44
        self.K_wheel_vel = -1.05

        # Store nominal parameters so wrappers can randomize without cumulative drift.
        self.nominal_params = {
            "K_sin": self.K_sin,
            "K_reac_wheel": self.K_reac_wheel,
            "K_pend_vel": self.K_pend_vel,
            "Ku": self.Ku,
            "K_wheel_vel": self.K_wheel_vel,
        }

        self.state = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_u = 0.0

        theta0 = self.np_random.uniform(-0.1, 0.1)
        theta_dot0 = self.np_random.uniform(-0.1, 0.1)
        phi0 = self.np_random.uniform(-0.1, 0.1)

        self.state = np.array([theta0, theta_dot0, phi0], dtype=np.float32)
        return self._get_observation(), {}

    def _get_observation(self):
        theta, theta_dot, phi = self.state
        return np.array(
            [np.sin(theta), np.cos(theta), theta_dot, phi, self.prev_u],
            dtype=np.float32,
        )

    def _dynamics(self, state, u):
        pend_pos, pend_vel, wheel_vel = state

        wheel_acc = self.Ku * u + self.K_wheel_vel * wheel_vel
        pend_acc = (
            self.K_sin * np.sin(pend_pos)
            + self.K_reac_wheel * wheel_acc
            + self.K_pend_vel * pend_vel
        )
        return np.array([pend_vel, pend_acc, wheel_acc], dtype=np.float32)

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
        self.state[0] = (self.state[0])

        new_theta, new_theta_dot, new_phi = self.state
        delta_u = u - self.prev_u
        self.prev_u = u

        upright_reward = np.cos(
            new_theta + np.pi
        )  # cos(theta+pi) = -cos(theta), max at theta=pi
        # Reduced penalties to allow learning; can increase later.
        shaping_penalty = (
            0.01 * new_theta_dot**2
            + 0.001 * new_phi**2
            + 0.001 * u**2
            + 0.05 * delta_u**2
        )
        near_upright_bonus = (
            2.0
            if abs((new_theta - np.pi)) < 0.3
            and abs(new_theta_dot) < 1.5
            else 0.0
        )
        reward = upright_reward - shaping_penalty + near_upright_bonus

        # Safety termination only for clearly divergent trajectories.
        # terminated = bool(abs(new_phi) > 150 or abs(new_theta_dot) > 30)
        truncated = self.step_count >= self.max_episode_steps

        info = {"u_cmd": u_cmd, "u_applied": u, "delta_u": delta_u}
        return self._get_observation(), reward, False, truncated, info


class ParamRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, range_pct=0.2):
        super().__init__(env)
        self.range_pct = range_pct

    def reset(self, **kwargs):
        e = self.env.unwrapped
        e.K_sin = e.nominal_params["K_sin"] * np.random.uniform(
            1 - self.range_pct, 1 + self.range_pct
        )
        e.K_reac_wheel = e.nominal_params["K_reac_wheel"] * np.random.uniform(
            1 - self.range_pct, 1 + self.range_pct
        )
        e.K_pend_vel = e.nominal_params["K_pend_vel"] * np.random.uniform(
            1 - self.range_pct, 1 + self.range_pct
        )

        obs, info = self.env.reset(**kwargs)
        info["ground_truth_params"] = np.array(
            [e.K_sin, e.K_reac_wheel, e.K_pend_vel], dtype=np.float32
        )

        return obs, info


class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_levels=None):
        super().__init__(env)
        if noise_levels is None:
            # [sin(theta), cos(theta), theta_dot, phi, prev_u]
            noise_levels = [0.01, 0.01, 0.01, 0.01, 0.0]
        self.noise_levels = np.array(noise_levels, dtype=np.float32)
        if self.noise_levels.shape != (self.observation_space.shape[0],):
            raise ValueError(
                "noise_levels must match observation dimension "
                f"{self.observation_space.shape[0]}"
            )

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_levels)
        return (obs + noise).astype(np.float32)
