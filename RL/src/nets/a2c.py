import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA2C(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hid_size=128, init_log_std=-0.5):
        super(ModelA2C, self).__init__()

        self.actor_net = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, hid_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hid_size, hid_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hid_size, act_size), std=0.01),
            nn.Tanh()
        )
        
        self.log_std = nn.Parameter(torch.full((act_size,), init_log_std))

        self.critic_net = nn.Sequential(
            self._layer_init(nn.Linear(obs_size, hid_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hid_size, hid_size)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hid_size, 1), std=1.0)
        )

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x: torch.Tensor):
        mu = self.actor_net(x)
        
        std = self.log_std.exp().expand_as(mu)
        var = std.pow(2)
        
        value = self.critic_net(x)
        
        return mu, var, value