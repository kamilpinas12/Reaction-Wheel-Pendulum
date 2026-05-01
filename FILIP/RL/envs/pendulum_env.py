import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ReactionWheelEnv(gym.Env):
    def __init__(self):
        super(ReactionWheelEnv, self).__init__()

        self.dt = 0.02
        self.u_max = 1.0
        self.max_episode_steps = 500
        self.step_count = 0
        self.prev_u = 0.0

        self.action_num = 7
        self.action_space = spaces.Discrete(self.action_num)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Constants from the nonlinear model
        self.A = 0.1167
        self.B = -3.915
        self.C = 0.08
        self.D = 0.00229
        self.K = 484.73
        
        # State: [theta, theta_dot, phi]
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

        self.state = np.array([theta0, theta_dot0, phi0], dtype=np.float32)
        return self._get_observation(), {}
    
    def step(self, action):
        self.step_count += 1

        torques = np.linspace(-0.9, 0.9, self.action_num)
        u_cmd = float(torques[action])

        self.prev_u = u_cmd
        self.state = self._rk4_step(self.state, u_cmd).astype(np.float32)
        self.state[0] = self._angle_normalize(self.state[0])

        reward = self._get_reward(self.state, u_cmd)
        
        terminated = bool(abs(self.state[2]) > 150 or abs(self.state[1]) > 30)
        truncated = self.step_count >= self.max_episode_steps

        info = {"u_cmd": u_cmd, "theta": self.state[0], "theta_dot": self.state[1], "phi": self.state[2], "reward": reward}

        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        theta, theta_dot, phi = self.state
        return np.array([theta, theta_dot, phi, self.prev_u], dtype=np.float32)
    
    def _get_reward(self, state, u):
        theta, theta_dot, phi = state

        upright_reward = np.cos(theta + np.pi)
        shaping_penalty = 0.01 * theta_dot**2 + 0.001 * phi**2 + 0.001 * u**2
        near_upright_bonus = 2.0 if abs(self._angle_normalize(theta - np.pi)) < 0.3 and abs(theta_dot) < 1.5 else 0.0
        
        return upright_reward - shaping_penalty + near_upright_bonus

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
    