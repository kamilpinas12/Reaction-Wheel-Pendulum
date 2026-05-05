import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ReactionWheelEnv(gym.Env):
    def __init__(self):
        super(ReactionWheelEnv, self).__init__()

        self.dt = 0.02
        self.u_max = 1.0
        self.max_episode_steps = 1000
        self.step_count = 0
        self.prev_u = 0.0

        self.action_num = 7
        self.action_space = spaces.Discrete(self.action_num)
        self.torques = np.linspace(-0.9, 0.9, self.action_num)
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

        theta0 = self.np_random.uniform(-0.25, 0.25)
        theta_dot0 = self.np_random.uniform(-0.1, 0.1)
        phi0 = self.np_random.uniform(-0.1, 0.1)

        self.state = np.array([theta0, theta_dot0, phi0], dtype=np.float32)
        return self._get_observation(), {}
    
    def step(self, action):
        self.step_count += 1

        u_cmd = float(self.torques[action])

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
        err = abs(self._angle_normalize(theta - np.pi))

        # 1. Primary Goal: Stay upright
        # Increased weight to make it more attractive than spinning
        upright_reward = 5.0 * np.exp(-3.0 * err)

        # 2. Phi Penalty: ONLY penalty near the physical limit
        phi_limit = 130.0
        phi_penalty = 0.0
        if abs(phi) > phi_limit:
            phi_penalty = 0.1 * (abs(phi) - phi_limit)**2

        # 3. Efficiency: Penalize spinning (The "Anti-Loop" penalty)
        # This is the secret to stopping the multiple circles.
        spinning_penalty = 0.1 * (theta_dot**2)

        # 4. Stay-Still Bonus: High reward for being at the top AND stopped
        stability_bonus = 0.0
        if err < 0.1:
            stability_bonus = 5.0 / (abs(theta_dot) + 0.1)

        return upright_reward + stability_bonus - phi_penalty - spinning_penalty - (0.001 * u**2)

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
    