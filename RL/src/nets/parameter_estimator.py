import numpy as np
import torch.nn as nn



class ParamEstimatorGRU(nn.Module):
    def __init__(self, n_features=5, n_params=3, hidden_dim=32, fc_dim=32):
        super(ParamEstimatorGRU, self).__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, n_params),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])
    
class ParamEstimatorLSTM(nn.Module):
    def __init__(self, n_features=5, n_params=3, hidden_dim=32, fc_dim=32):
        super(ParamEstimatorLSTM, self).__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, n_params),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

class ParamEstimatorConv(nn.Module):
    def __init__(self, n_features=5, n_params=3, hidden_dim=32, fc_dim=32):
        super(ParamEstimatorConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, n_params),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)