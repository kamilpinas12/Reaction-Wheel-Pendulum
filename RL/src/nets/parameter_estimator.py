import numpy as np
import torch.nn as nn



class PhysicalParameterEstimator(nn.Module):
    def __init__(self, n_features=5, n_params=3, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_params),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])

    