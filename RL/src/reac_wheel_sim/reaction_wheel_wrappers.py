import gymnasium as gym
import numpy as np
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv

class ParamRandomizationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: ReactionWheelEnv,
        K_sin_range=[-5, -1],
        K_reac_wheel_range=[0.1, 0.2],
        K_pend_vel_range=[0.1, 0.2],
    ):
        super().__init__(env)
        self.K_sin_range = K_sin_range
        self.K_reac_wheel_range = K_reac_wheel_range
        self.K_pend_vel_range = K_pend_vel_range

    def reset(self, **kwargs):
        self.env.K_sin = np.random.uniform(self.K_sin_range[0], self.K_sin_range[1])
        self.env.K_reac_wheel = np.random.uniform(
            self.K_reac_wheel_range[0], self.K_reac_wheel_range[1]
        )
        self.env.K_pend_vel = np.random.uniform(
            self.K_pend_vel_range[0], self.K_pend_vel_range[1]
        )

        obs, info = self.env.reset(**kwargs)
        info["ground_truth_params"] = np.array(
            [self.env.K_sin, self.env.K_reac_wheel, self.env.K_pend_vel], dtype=np.float32
        )

        return obs, info


# TODO normalize to observation value
class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_levels=None):
        super().__init__(env)
        if noise_levels is None:
            # [sin(theta), cos(theta), theta_dot, phi, prev_u]
            noise_levels = [0.0, 0.0, 0.0, 0.0]
        self.noise_levels = np.array(noise_levels, dtype=np.float32)
        if self.noise_levels.shape != (self.observation_space.shape[0],):
            raise ValueError(
                "noise_levels must match observation dimension "
                f"{self.observation_space.shape[0]}"
            )

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_levels)
        return (obs + noise).astype(np.float32)


class RandomInitialStateWrapper(gym.Wrapper):
    def __init__(self, env, initial_states=None):
        super().__init__(env)
        if initial_states is None:
            # [pend_pos, pend_vel, wheel_vel]
            initial_states = [0.0, 0.0, 0.0]
        self.initial_states = initial_states

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        pend_pos = self.np_random.uniform(-self.initial_states[0], self.initial_states[0])
        pend_vel = self.np_random.uniform(-self.initial_states[1], self.initial_states[1])
        wheel_vel = self.np_random.uniform(-self.initial_states[2], self.initial_states[2])

        new_state = np.array([pend_pos, pend_vel, wheel_vel], dtype=np.float32)
        self.env.unwrapped.state = new_state
        observation = self.env.unwrapped._get_observation()

        return observation, info