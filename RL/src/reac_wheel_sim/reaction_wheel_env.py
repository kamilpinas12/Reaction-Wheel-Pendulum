import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.config_manager import cfg_get


class ReactionWheelEnv(gym.Env):
    def __init__(self, config_name):
        super(ReactionWheelEnv, self).__init__()

        self.dt = cfg_get("env.dt", config_name, default=0.01)
        self.max_episode_steps = cfg_get(
            "env.max_episode_steps", config_name, default=1000
        )
        self.seed = cfg_get("env.seed", config_name, default=None)
        self.step_count = 0
        self.prev_u = 0.0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            shape=(1,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 4, dtype=np.float32), 
            high=np.array([np.inf] * 4, dtype=np.float32), 
            shape=(4,), 
            dtype=np.float32
        )
        self.initial_state_range = cfg_get("env.initial_state_range", config_name, None)

        # A
        self.K_pend_vel = cfg_get("env.K_pend_vel", config_name, default=0.085634)
        # B
        self.K_sin = cfg_get("env.K_sin", config_name, default=-9.101332)
        # C
        self.K_reac_wheel = cfg_get("env.K_reac_wheel", config_name, default=-0.009168)

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
        if seed is None and self.seed is not None:
            seed = self.seed
        super().reset(seed=seed, options=options)

        info = {}
        self.step_count = 0
        self.prev_u = 0.0

        # Enables overrites of rest functinality by passing options
        if options is not None and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        elif options is not None and "initial_range" in options:
            self.state = self._random_initial_state(options["initial_range"])
        elif self.initial_state_range is not None:
            self.state = self._random_initial_state(self.initial_state_range)
        else:
            self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        info["nominal_params"] = self.nominal_params.copy()
        return self._get_observation(), info
    
    def step(self, action):
        self.step_count += 1
        u = float(np.clip(action[0], -1.0, 1.0))

        self.state = self._rk4_step(u).astype(np.float32)
        self.state[0] = self._angle_normalize(self.state[0])

        reward = 0.0

        # Safety termination only for clearly divergent trajectories.
        # terminated = bool(abs(new_phi) > 150 or abs(new_theta_dot) > 30)
        truncated = self.step_count >= self.max_episode_steps

        self.prev_u = u
        info = {
            "u_cmd": u,
            "theta": self.state[0],
            "theta_dot": self.state[1],
            "phi": self.state[2],
        }
        return self._get_observation(), reward, False, truncated, info

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
        return pend_vel, pend_acc, wheel_acc

    def _rk4_step(self, u):
        s = self.state
        # K1
        k1_0, k1_1, k1_2 = self._dynamics(s, u)
        
        # K2
        s2 = (s[0] + 0.5*self.dt*k1_0, s[1] + 0.5*self.dt*k1_1, s[2] + 0.5*self.dt*k1_2)
        k2_0, k2_1, k2_2 = self._dynamics(s2, u)
        
        # K3
        s3 = (s[0] + 0.5*self.dt*k2_0, s[1] + 0.5*self.dt*k2_1, s[2] + 0.5*self.dt*k2_2)
        k3_0, k3_1, k3_2 = self._dynamics(s3, u)
        
        # K4
        s4 = (s[0] + self.dt*k3_0, s[1] + self.dt*k3_1, s[2] + self.dt*k3_2)
        k4_0, k4_1, k4_2 = self._dynamics(s4, u)

        new_state = np.array([
            s[0] + (self.dt / 6.0) * (k1_0 + 2.0*k2_0 + 2.0*k3_0 + k4_0),
            s[1] + (self.dt / 6.0) * (k1_1 + 2.0*k2_1 + 2.0*k3_1 + k4_1),
            s[2] + (self.dt / 6.0) * (k1_2 + 2.0*k2_2 + 2.0*k3_2 + k4_2)
        ], dtype=np.float32)
        
        return new_state
    
    def _random_initial_state(self, initial_range):
        generator = self.unwrapped.np_random
        pend_pos = generator.uniform(-initial_range[0], initial_range[0])
        pend_vel = generator.uniform(-initial_range[1], initial_range[1])
        wheel_vel = generator.uniform(-initial_range[2], initial_range[2])
        new_state = np.array([pend_pos, pend_vel, wheel_vel], dtype=np.float32)
        return new_state
