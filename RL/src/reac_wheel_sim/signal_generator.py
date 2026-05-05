from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BaseSignal(ABC):
    amplitude_range: tuple[float, float] = (0.2, 1.0)

    def _sample_amplitude(self, rng: np.random.Generator) -> float:
        lo, hi = self.amplitude_range
        amplitude = float(rng.uniform(lo, hi))
        sign = float(rng.choice([-1.0, 1.0]))
        return sign * amplitude

    @abstractmethod
    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class SquareSignal(BaseSignal):
    hold_range: tuple[int, int] = (12, 40)

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        signal = np.zeros(length, dtype=np.float32)
        index = 0

        while index < length:
            level = self._sample_amplitude(rng)
            hold = int(rng.integers(self.hold_range[0], self.hold_range[1] + 1))
            next_index = min(length, index + hold)
            signal[index:next_index] = level
            index = next_index

        return signal


@dataclass(slots=True)
class TrapezoidSignal(BaseSignal):
    hold_range: tuple[int, int] = (10, 40)
    ramp_range: tuple[int, int] = (12, 50)

    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        signal = np.zeros(length, dtype=np.float32)
        index = 0
        current_level = 0.0

        while index < length:
            target_level = self._sample_amplitude(rng)
            ramp_steps = int(rng.integers(self.ramp_range[0], self.ramp_range[1] + 1))
            hold_steps = int(rng.integers(self.hold_range[0], self.hold_range[1] + 1))

            ramp = np.linspace(
                current_level,
                target_level,
                ramp_steps + 1,
                dtype=np.float32,
            )[1:]
            ramp_end = min(length, index + ramp_steps)
            signal[index:ramp_end] = ramp[: ramp_end - index]
            index = ramp_end

            if index >= length:
                break

            hold_end = min(length, index + hold_steps)
            signal[index:hold_end] = target_level
            index = hold_end
            current_level = target_level

        return signal


@dataclass(slots=True)
class RandomSignal(BaseSignal):
    def generate(self, length: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=length).astype(np.float32)
