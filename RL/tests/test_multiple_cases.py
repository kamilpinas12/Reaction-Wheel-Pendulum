import numpy as np
import gymnasium as gym
import pytest

from utils.custom_paths import LOGS_DIR
from reac_wheel_sim.pend_sim_dataset import PendSimDataset
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import (
    ActionRepeatWrapper,
    ObservationNoiseWrapper,
    ParamRandomizationWrapper,
    TrigObservationWrapper,
)
from reac_wheel_sim.signal_generator import SquareSignal, TrapezoidSignal


class CountingEnv(gym.Env):
    def __init__(self, terminate_after=3):
        super().__init__()
        self.terminate_after = terminate_after
        self.step_calls = 0
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_calls = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_calls += 1
        terminated = self.step_calls >= self.terminate_after
        obs = np.array([float(self.step_calls)], dtype=np.float32)
        return obs, 1.0, terminated, False, {"step_calls": self.step_calls}


def test_reaction_wheel_env_reset_and_step():
    env = ReactionWheelEnv(config_name="config_ppo.yaml")

    obs, info = env.reset(
        seed=123,
        options={"initial_state": [0.1, -0.2, 0.3]},
    )

    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert set(info["nominal_params"]) >= {
        "K_sin",
        "K_reac_wheel",
        "K_pend_vel",
        "K_motor",
        "K_wheel_vel",
    }

    env.max_episode_steps = 1
    next_obs, reward, terminated, truncated, step_info = env.step(np.array([2.5], dtype=np.float32))

    assert next_obs.shape == (4,)
    assert next_obs.dtype == np.float32
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert step_info["u_cmd"] == pytest.approx(1.0)
    assert np.isfinite(next_obs).all()


def test_observation_wrappers_transform_and_validate():
    base_env = ReactionWheelEnv(config_name="config_ppo.yaml")

    noisy_env = ObservationNoiseWrapper(base_env, noise_levels=[0.0, 0.0, 0.0, 0.0])
    obs, _ = noisy_env.reset(
        seed=7,
        options={"initial_state": [0.25, -0.5, 0.75]},
    )
    assert obs.shape == (4,)
    assert np.allclose(obs, np.array([0.25, -0.5, 0.75, 0.0], dtype=np.float32))

    with pytest.raises(ValueError):
        ObservationNoiseWrapper(base_env, noise_levels=[0.0, 0.0])

    trig_env = TrigObservationWrapper(ReactionWheelEnv(config_name="config_ppo.yaml"))
    trig_obs, _ = trig_env.reset(
        seed=11,
        options={"initial_state": [np.pi / 2, 0.0, 0.0]},
    )
    assert trig_obs.shape == (5,)
    assert trig_obs[0] == pytest.approx(1.0, abs=1e-6)
    assert trig_obs[1] == pytest.approx(0.0, abs=1e-6)


def test_action_repeat_wrapper_accumulates_reward_and_stops_on_truncation():
    env = CountingEnv(terminate_after=3)
    wrapped = ActionRepeatWrapper(env, repeat=5)

    wrapped.reset()
    obs, reward, terminated, truncated, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert env.step_calls == 3
    assert obs.shape == (1,)
    assert reward == pytest.approx(3.0)
    assert terminated is True
    assert truncated is False
    assert info["step_calls"] == 3


def test_param_randomization_wrapper_and_dataset_generation(tmp_path):
    base_env = ReactionWheelEnv(config_name="config_ppo.yaml")
    env = ParamRandomizationWrapper(base_env, param_noise_pct=0.1, mass_pos_pct=0.1)
    env = ObservationNoiseWrapper(env, noise_levels=[0.0, 0.0, 0.0, 0.0])

    obs, info = env.reset(seed=42)

    assert obs.shape == (4,)
    assert np.isfinite(obs).all()
    assert info["ground_truth_params"].shape == (3,)
    assert info["physical_params"].shape == (3,)
    assert 0.0 <= info["mass_shift_pct"] <= 0.1

    signals = [
        SquareSignal(amplitude_range=(0.9, 1.0)),
        TrapezoidSignal(amplitude_range=(0.9, 1.0)),
    ]

    cache_path = tmp_path / "dataset_cache.pt"
    dataset = PendSimDataset(
        env=env,
        n_episodes=3,
        min_seq_len=8,
        max_seq_len=12,
        signals=signals,
        seed=123,
        cache_path=cache_path,
    )

    assert len(dataset) == 3
    assert dataset.features.shape == (3, 12, 4)
    assert dataset.targets.shape == (3, 3)
    assert dataset.valid_lengths.shape == (3,)
    assert int(dataset.valid_lengths.min()) >= 1
    assert int(dataset.valid_lengths.max()) <= 12

    seq, target, valid_len = dataset[0]
    assert seq.shape == (12, 4)
    assert target.shape == (3,)
    assert 1 <= int(valid_len) <= 12
    assert np.isfinite(seq.numpy()).all()
    assert np.isfinite(target.numpy()).all()

    reloaded = PendSimDataset(
        env=env,
        n_episodes=1,
        min_seq_len=8,
        max_seq_len=12,
        signals=signals,
        seed=123,
        cache_path=cache_path,
        load_datast=True,
    )

    assert len(reloaded) == 3
    assert np.array_equal(reloaded.features.numpy(), dataset.features.numpy())
    assert np.array_equal(reloaded.targets.numpy(), dataset.targets.numpy())
