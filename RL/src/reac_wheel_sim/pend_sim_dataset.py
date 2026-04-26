import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from reac_wheel_sim.reaction_wheel_env import *
from reac_wheel_sim.signal_generator import *


class PendSimDataset(Dataset):
    def __init__(
        self,
        n_episodes=2000,
        max_seq_len=256,
        range_pct=0.2,
        noise_levels=None,
        signals: List[BaseSignal] = None,
        seed=None,
    ):
        self.n_episodes = n_episodes
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)
        
        if signals is None:
            raise ValueError("signals must contain at least one signal")
        self.signals = signals

        env = ReactionWheelEnv()
        env = ParamRandomizationWrapper(env, range_pct=range_pct)
        env = ObservationNoiseWrapper(env, noise_levels=noise_levels)
        self.env = env

        self.features = []
        self.targets = []
        self.valid_lengths = []
        self.create_dataset()

    def _generate_action_sequence(self, length):
        signal = self.signals[int(self.rng.integers(len(self.signals)))]
        return signal.generate(length, self.rng)

    def create_dataset(self):
        self.features.clear()
        self.targets.clear()
        self.valid_lengths.clear()

        for episode_idx in range(self.n_episodes):
            obs, info = self.env.reset(seed=int(self.rng.integers(0, 1_000_000_000)))
            target_params = np.asarray(info["ground_truth_params"], dtype=np.float32)
            action_sequence = self._generate_action_sequence(self.max_seq_len)

            seq_rows = []
            terminated = False
            truncated = False

            while not (terminated or truncated) and len(seq_rows) < self.max_seq_len:
                action_idx = len(seq_rows)
                action = np.array([action_sequence[action_idx]], dtype=np.float32)
                # step_features = np.concatenate([obs, action], axis=0)
                seq_rows.append(obs)

                obs, _, terminated, truncated, _ = self.env.step(action)

            valid_len = len(seq_rows)
            seq = np.zeros((self.max_seq_len, 5), dtype=np.float32)
            if valid_len > 0:
                seq[:valid_len] = np.asarray(seq_rows, dtype=np.float32)

            if not np.isfinite(seq).all() or not np.isfinite(target_params).all():
                raise ValueError(f"Found non-finite values in generated sample {episode_idx}")

            self.features.append(torch.from_numpy(seq))
            self.targets.append(torch.from_numpy(target_params))
            self.valid_lengths.append(valid_len)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def visualize_episode(self, index=0, max_points=None, save_path=None, show=True):
        if index < 0 or index >= len(self.features):
            raise IndexError(f"Episode index {index} out of range [0, {len(self.features) - 1}]")

        seq = self.features[index].cpu().numpy()
        target = self.targets[index].cpu().numpy()
        valid_len = self.valid_lengths[index]

        if valid_len == 0:
            raise ValueError(f"Episode {index} has zero valid steps")

        plot_len = valid_len if max_points is None else min(valid_len, max_points)
        seq = seq[:plot_len]

        obs = seq[:, :5]
        t = np.arange(plot_len)

        fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)

        axes[0].plot(t, obs[:, 0], label="sin(theta)", linewidth=1.2)
        axes[0].plot(t, obs[:, 1], label="cos(theta)", linewidth=1.2)
        axes[0].set_ylabel("Trig")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(t, obs[:, 2], label="theta_dot", linewidth=1.2)
        axes[1].set_ylabel("State")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        axes[2].plot(t, obs[:, 2], label="theta_dot", linewidth=1.2)
        axes[2].set_ylabel("theta_dot")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        axes[3].plot(t, obs[:, 3], label="phi", linewidth=1.2)
        axes[3].set_ylabel("phi")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right")

        axes[4].plot(t, obs[:, 4], label="prev_u", linewidth=1.2)
        axes[4].set_ylabel("Control")
        axes[4].set_xlabel("Step")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend(loc="upper right")

        fig.suptitle(
            "Generated Episode "
            f"{index} | target [K_sin, K_reac_wheel, K_pend_vel] = "
            f"[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
        )
        fig.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def visualize_target_distribution(self, bins=40, save_path=None, show=True):
        if len(self.targets) == 0:
            raise ValueError("Dataset is empty")

        targets = torch.stack(self.targets).cpu().numpy()
        names = ["K_sin", "K_reac_wheel", "K_pend_vel"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for idx, ax in enumerate(axes):
            ax.hist(targets[:, idx], bins=bins, alpha=0.8, edgecolor="black")
            ax.set_title(names[idx])
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)

        fig.suptitle("Generated Target Parameter Distribution")
        fig.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def visualize_generated_samples(
        self,
        n_samples=4,
        indices=None,
        max_points=None,
        save_path=None,
        show=True,
    ):
        if len(self.features) == 0:
            raise ValueError("Dataset is empty")

        if indices is None:
            n_samples = min(int(n_samples), len(self.features))
            indices = self.rng.choice(len(self.features), size=n_samples, replace=False)
        else:
            indices = list(indices)
            if len(indices) == 0:
                raise ValueError("indices must contain at least one episode index")

        n_samples = len(indices)
        fig, axes = plt.subplots(
            n_samples,
            5,
            figsize=(18, 3.2 * n_samples),
            sharex="col",
            squeeze=False,
        )

        for row, episode_idx in enumerate(indices):
            if episode_idx < 0 or episode_idx >= len(self.features):
                raise IndexError(
                    f"Episode index {episode_idx} out of range [0, {len(self.features) - 1}]"
                )

            seq = self.features[episode_idx].cpu().numpy()
            target = self.targets[episode_idx].cpu().numpy()
            valid_len = self.valid_lengths[episode_idx]

            if valid_len == 0:
                raise ValueError(f"Episode {episode_idx} has zero valid steps")

            plot_len = valid_len if max_points is None else min(valid_len, max_points)
            seq = seq[:plot_len]
            obs = seq[:, :5]
            t = np.arange(plot_len)

            ax = axes[row]
            ax[0].plot(t, obs[:, 4], color="tab:orange", linewidth=1.3)
            ax[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            ax[0].set_ylabel("prev_u")
            ax[0].grid(True, alpha=0.25)

            ax[1].plot(t, obs[:, 0], label="sin(theta)", linewidth=1.1)
            ax[1].plot(t, obs[:, 1], label="cos(theta)", linewidth=1.1)
            ax[1].set_ylabel("Trig")
            ax[1].grid(True, alpha=0.25)
            ax[1].legend(loc="upper right", fontsize=8)

            ax[2].plot(t, obs[:, 2], label="theta_dot", linewidth=1.1)
            ax[2].set_ylabel("theta_dot")
            ax[2].grid(True, alpha=0.25)
            ax[2].legend(loc="upper right", fontsize=8)

            ax[3].plot(t, obs[:, 3], label="phi", linewidth=1.1)
            ax[3].set_ylabel("phi")
            ax[3].grid(True, alpha=0.25)
            ax[3].legend(loc="upper right", fontsize=8)

            ax[4].plot(t, obs[:, 4], label="prev_u", linewidth=1.1)
            ax[4].set_ylabel("Control")
            ax[4].grid(True, alpha=0.25)
            ax[4].legend(loc="upper right", fontsize=8)

            ax[0].set_title(
                f"Episode {episode_idx} | target = "
                f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]",
                fontsize=10,
            )

        for col in range(5):
            axes[-1, col].set_xlabel("Step")

        fig.suptitle("Generated Dataset Samples", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)