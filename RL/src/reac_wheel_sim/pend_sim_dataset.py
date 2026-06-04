import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from reac_wheel_sim.reaction_wheel_env import *
from reac_wheel_sim.signal_generator import *
from reac_wheel_sim.reaction_wheel_wrappers import *


class PendSimDataset(Dataset):
    def __init__(
        self,
        env,
        n_episodes: int = 2000,
        min_seq_len: int = 64,
        max_seq_len: int = 256,
        signals: List[BaseSignal] = None,
        seed=None,
        cache_path=None,
        load_datast=False,
    ):
        if signals is None:
            raise ValueError("signals must contain at least one signal")
        
        self.n_episodes = n_episodes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)
        self.signals = signals
        self.env = env
        self.cache_path = cache_path

        self.features: torch.Tensor
        self.targets: torch.Tensor
        self.valid_lengths: torch.Tensor

        if load_datast and self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading dataset from cache: {self.cache_path}")
            self._load_from_cache()
        else:
            print("Generating new dataset...")
            self.create_dataset()
            if self.cache_path:
                self._save_to_cache()

    def _generate_action_sequence(self, length):
        signal: BaseSignal = self.rng.choice(self.signals)
        return signal.generate(length, self.rng)

    def create_dataset(self):
        features_list = []
        targets_list = []
        lengths_list = []

        for episode_idx in range(self.n_episodes):
            env_seed = int(self.rng.integers(0, 1_000_000_000))
            obs, info = self.env.reset(seed=env_seed)

            target_params = np.asarray(list(info["model_params"].values())[:3], dtype=np.float32)
            # Random episode length
            episode_len = int(self.rng.integers(self.min_seq_len, self.max_seq_len + 1))
            action_sequence = self._generate_action_sequence(episode_len)

            seq_rows = []
            for action_val in action_sequence:
                seq_rows.append(obs)
                action = np.array([action_val], dtype=np.float32)
                obs, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    break

            valid_len = len(seq_rows)
            feature_dim = seq_rows[0].shape[0] if valid_len > 0 else 0
            
            # Zero padding
            seq = np.zeros((self.max_seq_len, feature_dim), dtype=np.float32)
            if valid_len > 0:
                seq[:valid_len] = np.asarray(seq_rows, dtype=np.float32)

            if not np.isfinite(seq).all() or not np.isfinite(target_params).all():
                raise ValueError(f"Found non-finite values in generated sample {episode_idx}")

            features_list.append(torch.from_numpy(seq))
            targets_list.append(torch.from_numpy(target_params))
            lengths_list.append(valid_len)

        self.features = torch.stack(features_list)
        self.targets = torch.stack(targets_list)
        self.valid_lengths = torch.tensor(lengths_list, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.valid_lengths[index]
    
    def _save_to_cache(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        data_dict = {
            "features": self.features,
            "targets": self.targets,
            "valid_lengths": self.valid_lengths,
            "n_episodes": self.n_episodes,
            "max_seq_len": self.max_seq_len,
        }
        torch.save(data_dict, self.cache_path)
        print(f"Dataset saved to: {self.cache_path}")

    def _load_from_cache(self):
        data_dict = torch.load(self.cache_path, weights_only=True)
        self.features = data_dict["features"]
        self.targets = data_dict["targets"]
        self.valid_lengths = data_dict["valid_lengths"]
        
        self.n_episodes = data_dict.get("n_episodes", len(self.features))
        self.max_seq_len = data_dict.get("max_seq_len", self.features.shape[1])
