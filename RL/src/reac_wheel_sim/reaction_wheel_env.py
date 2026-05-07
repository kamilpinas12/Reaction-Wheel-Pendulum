import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ReactionWheelEnv(gym.Env):
    def __init__(self):
        super(ReactionWheelEnv, self).__init__()

        self.dt = 0.01
        self.u_max = 1.0
        self.max_episode_steps = 1000  # 10s
        self.step_count = 0
        self.prev_u = 0.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.K_pend_vel = 0.1165  # A
        self.K_sin = -3.915  # B
        self.K_reac_wheel = -0.008  # C

        self.K_motor = 484.73  # K
        self.K_wheel_vel = 0.00229  # D

        # Store nominal parameters so wrappers can randomize without cumulative drift.
        self.nominal_params = {
            "K_sin": self.K_sin,
            "K_reac_wheel": self.K_reac_wheel,
            "K_pend_vel": self.K_pend_vel,
            "K_motor": self.K_motor,
            "K_wheel_vel": self.K_wheel_vel,
        }

        self.state = None

    @staticmethod
    def _angle_normalize(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.prev_u = 0.0

        pend_pos = 0.0
        pend_vel = 0.0
        wheel_vel = 0.0

        self.state = np.array([pend_pos, pend_vel, wheel_vel], dtype=np.float32)
        return self._get_observation(), self.nominal_params

    def _get_observation(self):
        pend_pos, pend_vel, wheel_vel = self.state
        return np.array(
            [pend_pos, pend_vel, wheel_vel, self.prev_u],
            dtype=np.float32,
        )

    def _dynamics(self, state, u):
        pend_pos, pend_vel, wheel_vel = state

        wheel_acc = self.K_motor * (u - self.K_wheel_vel * wheel_vel)
        pend_acc = (
            -self.K_pend_vel * pend_vel
            + self.K_sin * np.sin(pend_pos)
            + self.K_reac_wheel * wheel_acc
        )
        return np.array([pend_vel, pend_acc, wheel_acc], dtype=np.float32)

    def _rk4_step(self, u):
        k1 = self._dynamics(self.state, u)
        k2 = self._dynamics(self.state + 0.5 * self.dt * k1, u)
        k3 = self._dynamics(self.state + 0.5 * self.dt * k2, u)
        k4 = self._dynamics(self.state + self.dt * k3, u)

        return self.state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action):
        self.step_count += 1

        # Preporcess action
        u = float(np.clip(action[0], -1.0, 1.0)) * self.u_max

        # Env step
        self.state = self._rk4_step(u).astype(np.float32)
        self.state[0] = self._angle_normalize(self.state[0])

        reward = self._get_reward(u)

        # Safety termination only for clearly divergent trajectories.
        # terminated = bool(abs(new_phi) > 150 or abs(new_theta_dot) > 30)
        truncated = self.step_count >= self.max_episode_steps

        self.prev_u = u
        info = {"u_cmd": u,}
        return self._get_observation(), reward, False, truncated, info

    def _get_reward(self, u):
        theta, theta_dot, phi = self.state
        err = abs(self._angle_normalize(theta - np.pi))

        # 1. Primary Goal: Stay upright
        # Increased weight to make it more attractive than spinning
        upright_reward = 5.0 * np.exp(-3.0 * err)

        # 2. Phi Penalty: ONLY penalty near the physical limit
        phi_limit = 130.0
        phi_penalty = 0.0
        if abs(phi) > phi_limit:
            phi_penalty = 0.1 * (abs(phi) - phi_limit) ** 2

        # 3. Efficiency: Penalize spinning (The "Anti-Loop" penalty)
        # This is the secret to stopping the multiple circles.
        spinning_penalty = 0.1 * (theta_dot**2)

        # 4. Stay-Still Bonus: High reward for being at the top AND stopped
        stability_bonus = 0.0
        if err < 0.1:
            stability_bonus = 5.0 / (abs(theta_dot) + 0.1)

        return (
            upright_reward
            + stability_bonus
            - phi_penalty
            - spinning_penalty
            - (0.001 * u**2)
        )
