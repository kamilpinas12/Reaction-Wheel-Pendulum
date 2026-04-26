
import logging
import pytest

from reac_wheel_sim.pend_sim_dataset import PendSimDataset
from reac_wheel_sim.signal_generator import SquareSignal, TrapezoidSignal
from config import LOGS_DIR


def test_dataset_generation():
	signals = [
		SquareSignal(amplitude_range=(0.9, 1.0)),
		TrapezoidSignal(amplitude_range=(0.9, 1.0)),
	]

	dataset = PendSimDataset(
		n_episodes=12,
		max_seq_len=160,
		range_pct=0.2,
		noise_levels=[0.0, 0.0, 0.0, 0.0, 0.0],
		signals=signals,
		seed=42,
	)

	try:
		dataset.visualize_generated_samples(
			n_samples=4,
			max_points=160,
			save_path=LOGS_DIR / "pytest" / "test_dataset_generated_samples.png",
			show=False,
		)
	except Exception as e:
		logging.exception("Wystąpił błąd podczas wizualizacji próbek!")
		pytest.fail(f"Wizualizacja przerwana błędem: {e}")


