import logging
import pytest

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import ParamRandomizationWrapper, ObservationNoiseWrapper
from reac_wheel_sim.pend_sim_dataset import PendSimDataset
from reac_wheel_sim.signal_generator import SquareSignal, TrapezoidSignal
from utils.custom_paths import LOGS_DIR

# Importujemy nową funkcję z miejsca, w którym ją zapisałeś
from utils.visualizations import visualize_dataset_samples


def test_dataset_generation_and_visualization():
    signals = [
        SquareSignal(amplitude_range=(0.9, 1.0)),
        TrapezoidSignal(amplitude_range=(0.9, 1.0)),
    ]

    base_env = ReactionWheelEnv()
    env = ParamRandomizationWrapper(
        base_env,
        param_noise_pct=0.1,
        mass_pos_pct=0.1,
    )
    env = ObservationNoiseWrapper(env, noise_levels=[0.0, 0.0, 0.0, 0.0])

    try:
        dataset = PendSimDataset(
            env=env,
            n_episodes=12,
            min_seq_len=64,
            max_seq_len=256,
            signals=signals,
            seed=42,
        )
    except Exception as e:
        pytest.fail(f"Inicjalizacja datasetu przerwana błędem: {e}")

    # Użycie wyodrębnionej funkcji
    try:
        save_path = LOGS_DIR / "pytest" / "test_dataset_generated_samples.png"
        visualize_dataset_samples(
            dataset=dataset,
            n_samples=4,
            save_path=save_path,
            show=False
        )
    except Exception as e:
        logging.exception("Wystąpił błąd podczas wizualizacji próbek!")
        pytest.fail(f"Wizualizacja przerwana błędem: {e}")