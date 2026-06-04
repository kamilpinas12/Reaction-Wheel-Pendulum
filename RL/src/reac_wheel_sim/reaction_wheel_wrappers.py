import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import product
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
import dataclasses
"""'
Zamiast ustawiania zakresów zrobiłem maksymalne zakresy zhardkodowane, a do konfiguracji jest 
procent ( od 0 do 1 ) tego jak daleko mozemy oddalać się od nominalnych wartości. Po przejściu do 
parametrów fizycznych tarcie jest w miare stałe a te dwa pozostałe parametry są dość dobrze skorelowane
więc jest parametr do tego w jakim procencie pozwalamy na oddalanie się ciężarka od nominalnej pozycji. 
Klasa zadba o to żeby te parametry zawsze były skorelowane. Wyobrażam sobie że będzie to wyglądać tak że 
w trakcie treningu będziemy sobie zwiększać tylko te liczby procentowe zamiast przejmować się 
sensownym zmienianiem zakresów poza tym wrapperem. 
"""




class ParamRandomizationWrapper(gym.Wrapper):
    # ml range
    # IP diff
    # f range 
    # [Ip, f, ml]
    IW = 0.00023
    KM = 484.73
    D = 0.00229

    def __init__(
        self,
        env: ReactionWheelEnv,
        nominal_params = [0.024, 0.022, 0.03],
        params_range = [0.005, 0.003, 0.02],
        Ip_corelation = 0.001
    ):
        super().__init__(env)
        self.nominal_params = nominal_params
        self.params_range = params_range
        self.Ip_corelation = Ip_corelation

    def get_bounds(self, idx):
        if idx > min(len(self.nominal_params), len(self.params_range)):
            raise ValueError("idx out of range")
        return [self.nominal_params[idx] - self.params_range[idx],
                self.nominal_params[idx] + self.params_range[idx]]

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Skip randomization if model param provided in options
        if options is not None and "model_params" in options:
            return obs, info
        
        rng = self.env.unwrapped.np_random

        Ip_range = self.get_bounds(0)
        f_range = self.get_bounds(1)
        ml_range = self.get_bounds(2)
        f = rng.uniform(f_range[0], f_range[1])
        ml = rng.uniform(ml_range[0], ml_range[1])

        if self.Ip_corelation is None:
            Ip = rng.uniform(Ip_range[0], Ip_range[1])
        else:
            Ip_mid = -0.221 * ml + 0.031 
            Ip = rng.uniform(Ip_mid - self.Ip_corelation, Ip_mid + self.Ip_corelation)
            Ip = np.clip(Ip, Ip_range[0], Ip_range[1])

        self.env.unwrapped.K_pend_vel = f / Ip
        self.env.unwrapped.K_sin = -(ml * 9.81) / Ip
        self.env.unwrapped.K_reac_wheel = -self.IW / Ip
        self.env.unwrapped.K_motor = self.KM
        self.env.unwrapped.K_wheel_vel = self.D

        info["phys_params"] = {
            "Ip": Ip,
            "f": f,
            "ml": ml
        }
        info["model_params"] = self.env.unwrapped.get_model_params()
        obs = self.env.unwrapped._get_observation()
        return obs, info


# TODO normalize to observation value
class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_levels=None):
        super().__init__(env)
        if noise_levels is None:
            noise_levels = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.noise_levels = np.array(noise_levels, dtype=np.float32)
        if self.noise_levels.shape != (self.observation_space.shape[0],):
            raise ValueError(
                "noise_levels must match observation dimension "
                f"{self.observation_space.shape[0]}"
            )

    def observation(self, obs):
        noise = self.env.unwrapped.np_random.normal(0.0, self.noise_levels)
        return (obs + noise).astype(np.float32)


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins=7):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n_bins)

        # self.torques = np.linspace(-0.9, 0.9, n_bins)
        self.discrete_to_continuous = np.linspace(
            env.action_space.low, env.action_space.high, n_bins
        ).flatten()

    def action(self, action):
        return np.array([self.discrete_to_continuous[action]], dtype=np.float32)


class TrigAndNormalizationObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_space = self.env.observation_space
        new_shape = (orig_space.shape[0],)
        new_low = np.concatenate(([-1.0, -1.0], orig_space.low[1:3])).astype(np.float32)
        new_high = np.concatenate(([1.0, 1.0], orig_space.high[1:3])).astype(np.float32)

        self.observation_space = spaces.Box(
            low=new_low, high=new_high, shape=new_shape, dtype=np.float32
        )

    def observation(self, obs):
        pend_pos = obs[0]
        pend_vel = obs[1]
        wheel_vel = obs[2]
        prev_u = obs[3]

        return np.array(
            [
                np.sin(pend_pos),
                np.cos(pend_pos),
                pend_vel / 5.0,
                wheel_vel / 400.0,
                # prev_u,
            ],
            dtype=np.float32,
        )


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
