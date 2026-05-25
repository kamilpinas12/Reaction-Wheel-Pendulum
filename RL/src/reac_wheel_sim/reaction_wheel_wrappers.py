import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import product
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv

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

            values.append(
                [
                    -(ml * 9.81) / Ip,
                    -self.IW / Ip,
                    f / Ip,
                ]
            )

        arr = np.asarray(values, dtype=np.float64)
        min_vals = arr.min(axis=0).astype(np.float32)
        max_vals = arr.max(axis=0).astype(np.float32)
        return min_vals, max_vals

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        rng = self.env.unwrapped.np_random
        opts = options or {}
        max_shift = float(np.clip(self.mass_pos_pct, 0.0, 1.0))
        if "mass_shift_pct" in opts:
            mass_shift_pct = float(np.clip(opts["mass_shift_pct"], 0.0, 1.0))
        else:
            mass_shift_pct = rng.uniform(0.0, max_shift)

        param_noise_pct = float(opts.get("param_noise_pct", self.param_noise_pct))

        Ip_base = (1.0 - mass_shift_pct) * self.NOMINAL[
            "Ip"
        ] + mass_shift_pct * self.BOUNDARY["Ip"]
        f_base = (1.0 - mass_shift_pct) * self.NOMINAL[
            "f"
        ] + mass_shift_pct * self.BOUNDARY["f"]
        ml_base = -4.52 * Ip_base + 0.14

        Ip = Ip_base * (1.0 + rng.uniform(-param_noise_pct, param_noise_pct))
        f = f_base * (1.0 + rng.uniform(-param_noise_pct, param_noise_pct))
        ml = ml_base * (1.0 + rng.uniform(-param_noise_pct, param_noise_pct))
        ml = 0.5 * ml + 0.5 * (-4.52 * Ip + 0.14)

        self.env.unwrapped.K_pend_vel = f / Ip
        self.env.unwrapped.K_sin = -(ml * 9.81) / Ip
        self.env.unwrapped.K_reac_wheel = -self.IW / Ip
        self.env.unwrapped.K_motor = self.KM
        self.env.unwrapped.K_wheel_vel = self.D
        info["ground_truth_params"] = np.array(
            [self.env.K_sin, self.env.K_reac_wheel, self.env.K_pend_vel],
            dtype=np.float32,
        )
        info["physical_params"] = np.array([Ip, f, ml], dtype=np.float32)
        info["mass_shift_pct"] = float(mass_shift_pct)

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
                pend_vel / 4.0,
                wheel_vel / 300.0,
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
