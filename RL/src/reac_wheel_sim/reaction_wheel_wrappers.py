import gymnasium as gym
import numpy as np
from itertools import product
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv


''''
Zamiast ustawiania zakresów zrobiłem maksymalne zakresy zhardkodowane, a do konfiguracji jest 
procent ( od 0 do 1 ) tego jak daleko mozemy oddalać się od nominalnych wartości. Po przejściu do 
parametrów fizycznych tarcie jest w miare stałe a te dwa pozostałe parametry są dość dobrze skorelowane
więc jest parametr do tego w jakim procencie pozwalamy na oddalanie się ciężarka od nominalnej pozycji. 
Klasa zadba o to żeby te parametry zawsze były skorelowane. Wyobrażam sobie że będzie to wyglądać tak że 
w trakcie treningu będziemy sobie zwiększać tylko te liczby procentowe zamiast przejmować się 
sensownym zmienianiem zakresów poza tym wrapperem. 
'''
class ParamRandomizationWrapper(gym.Wrapper):
    NOMINAL = {"Ip": 0.028093, "f": 0.002439, "ml": 0.019931}
    BOUNDARY = {"Ip": 0.0199, "f": 0.00214, "ml": 0.0553}
    IW = 0.00023
    KM = 484.73
    D = 0.00229

    def __init__(
        self,
        env: ReactionWheelEnv,
        param_noise_pct=0.1,
        mass_pos_pct=0.1,
    ):
        super().__init__(env)
        self.param_noise_pct = float(param_noise_pct)
        self.mass_pos_pct = float(mass_pos_pct)

    def get_ground_truth_bounds(self):
        """Return fixed global min/max bounds for [K_sin, K_reac_wheel, K_pend_vel]."""
        values = []
        for shift, n_ip, n_f, n_ml in product((0.0, 1.0), repeat=4):
            Ip_base = (1.0 - shift) * self.NOMINAL["Ip"] + shift * self.BOUNDARY["Ip"]
            f_base = (1.0 - shift) * self.NOMINAL["f"] + shift * self.BOUNDARY["f"]
            ml_base = -4.52 * Ip_base + 0.14

            Ip = Ip_base * (1.0 + n_ip)
            f = f_base * (1.0 + n_f)
            ml = 0.5 * (ml_base * (1.0 + n_ml)) + 0.5 * (-4.52 * Ip + 0.14)

            values.append([
                -(ml * 9.81) / Ip,
                -self.IW / Ip,
                f / Ip,
            ])

        arr = np.asarray(values, dtype=np.float64)
        min_vals = arr.min(axis=0).astype(np.float32)
        max_vals = arr.max(axis=0).astype(np.float32)
        return min_vals, max_vals

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        rng = self.env.unwrapped.np_random
        max_shift = float(np.clip(self.mass_pos_pct, 0.0, 1.0))
        mass_shift_pct = rng.uniform(0.0, max_shift)

        Ip_base = (1.0 - mass_shift_pct) * self.NOMINAL["Ip"] + mass_shift_pct * self.BOUNDARY["Ip"]
        f_base = (1.0 - mass_shift_pct) * self.NOMINAL["f"] + mass_shift_pct * self.BOUNDARY["f"]
        ml_base = -4.52 * Ip_base + 0.14

        Ip = Ip_base * (1.0 + rng.uniform(-self.param_noise_pct, self.param_noise_pct))
        f = f_base * (1.0 + rng.uniform(-self.param_noise_pct, self.param_noise_pct))
        ml = ml_base * (1.0 + rng.uniform(-self.param_noise_pct, self.param_noise_pct))
        ml = 0.5 * ml + 0.5 * (-4.52 * Ip + 0.14)

        self.env.unwrapped.K_pend_vel = f / Ip
        self.env.unwrapped.K_sin = -(ml * 9.81) / Ip
        self.env.unwrapped.K_reac_wheel = -self.IW / Ip
        self.env.unwrapped.K_motor = self.KM
        self.env.unwrapped.K_wheel_vel = self.D

        self.env.nominal_params = {
            "K_sin": self.env.K_sin,
            "K_reac_wheel": self.env.K_reac_wheel,
            "K_pend_vel": self.env.K_pend_vel,
            "K_motor": self.env.K_motor,
            "K_wheel_vel": self.env.K_wheel_vel,
        }

        obs, info = self.env.reset(seed=seed, options=options)
        info["ground_truth_params"] = np.array(
            [self.env.K_sin, self.env.K_reac_wheel, self.env.K_pend_vel], dtype=np.float32
        )
        info["physical_params"] = np.array([Ip, f, ml], dtype=np.float32)
        info["mass_shift_pct"] = float(mass_shift_pct)

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