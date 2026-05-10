import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA2C(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hid_size=128):
        super(ModelA2C, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size)
        )
        
        self.log_std = nn.Parameter(torch.zeros(act_size))

        self.critic_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, 1)
        )

    def forward(self, x: torch.Tensor):
        mu = self.actor_net(x)
        
        std = self.log_std.exp().expand_as(mu)
        var = std.pow(2)
        
        value = self.critic_net(x)
        
        return mu, var, value