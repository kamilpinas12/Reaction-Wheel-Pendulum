import numpy as np
import torch
from typing import Generator, NamedTuple


class RolloutBufferSamples(NamedTuple):
    """Struktura przechowująca gotową paczkę danych dla sieci."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """
    Bufor trajektorii dla algorytmów On-Policy (A2C, PPO).
    Przechowuje dane ze zrównoleglonych środowisk (VectorEnv).
    """

    def __init__(
        self,
        buffer_size: int,  # Ilość kroków (np. 2048)
        num_envs: int,  # Ilość środowisk (np. 8)
        obs_shape: tuple,  # Kształt obserwacji (np. (3,) dla Pendulum)
        act_shape: tuple,  # Kształt akcji (np. (1,))
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.pos = 0  # Aktualny wskaźnik zapisu
        self.full = False  # Flaga przepełnienia bufora

        # Pre-alokacja pamięci dla maksymalnej wydajności (Tensory od razu na GPU/CPU)
        self.observations = torch.zeros(
            (self.buffer_size, self.num_envs, *self.obs_shape), dtype=torch.float32
        ).to(self.device)
        self.actions = torch.zeros(
            (self.buffer_size, self.num_envs, *self.act_shape), dtype=torch.float32
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)
        self.values = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)
        self.log_probs = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)
        self.dones = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)

        # Obliczane po zebraniu danych
        self.advantages = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)
        self.returns = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        ).to(self.device)

    def reset(self) -> None:
        """Resetuje wskaźniki na początku nowej fazy zbierania danych."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """Dodaje pojedynczy krok ze wszystkich środowisk do bufora."""
        if self.full:
            raise RuntimeError(
                "RolloutBuffer is full! Oblicz zwroty i zresetuj bufor przed dodaniem nowych danych."
            )

        # Zapisujemy dane pod aktualnym wskaźnikiem
        self.observations[self.pos] = torch.as_tensor(obs).to(self.device)
        self.actions[self.pos] = torch.as_tensor(action).to(self.device)
        self.rewards[self.pos] = torch.as_tensor(reward).to(self.device)
        self.dones[self.pos] = torch.as_tensor(done).to(self.device)

        # Wartości i prawdopodobieństwa są już tensorami z sieci
        self.values[self.pos] = value.clone().detach()
        self.log_probs[self.pos] = log_prob.clone().detach()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        """
        Oblicza przewagi przy pomocy Generalized Advantage Estimation (GAE).
        To najważniejszy trik matematyczny z PPO/A2C.
        """
        last_values = last_values.clone().detach().flatten()
        last_dones = torch.as_tensor(dones).to(self.device).flatten()

        last_gae_lam = 0

        # Obliczamy od końca (Bootstrapping)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = (~last_dones).float()
                next_values = last_values
            else:
                next_non_terminal = (~self.dones[step + 1].bool()).float()
                next_values = self.values[step + 1]

            # Delta = r + gamma * V(s') - V(s)
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # GAE
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        # Returns = Advantages + Values
        self.returns = self.advantages + self.values

    def get(
        self, batch_size: int = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Zwraca dane z bufora do nauki. Jeśli podano batch_size,
        spłaszcza dane i zwraca potasowane paczki (wymagane w PPO).
        """
        assert self.full, "Bufor musi być pełny, zanim zaczniesz trenować!"

        # Spłaszczanie danych z kształtu (buffer_size, num_envs, ...)
        # do jednowymiarowej listy paczek (buffer_size * num_envs, ...)
        indices = np.random.permutation(self.buffer_size * self.num_envs)

        flat_obs = self.observations.view(-1, *self.obs_shape)
        flat_actions = self.actions.view(-1, *self.act_shape)
        flat_values = self.values.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)

        # Standardowa normalizacja Advantage (Triki RL, o których rozmawialiśmy)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # Zwracanie potasowanych Minibatchy
        if batch_size is None:
            batch_size = self.buffer_size * self.num_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.num_envs:
            yield RolloutBufferSamples(
                observations=flat_obs[indices[start_idx : start_idx + batch_size]],
                actions=flat_actions[indices[start_idx : start_idx + batch_size]],
                old_values=flat_values[indices[start_idx : start_idx + batch_size]],
                old_log_probs=flat_log_probs[
                    indices[start_idx : start_idx + batch_size]
                ],
                advantages=flat_advantages[indices[start_idx : start_idx + batch_size]],
                returns=flat_returns[indices[start_idx : start_idx + batch_size]],
            )
            start_idx += batch_size
