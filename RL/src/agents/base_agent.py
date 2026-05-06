from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from utils.config_manager import get as cfg_get


class BaseAgent(ABC):

    def __init__(self, env=None):
        self.env = env
        self.learning_rate = float(cfg_get('base_agent.learning_rate'))
        gm = cfg_get('base_agent.gamma')
        self.gamma = float(gm) if gm is not None else 0.99
        self.seed = cfg_get('base_agent.seed')
        self.device = cfg_get('base_agent.device')
        
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
    
   
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True, render: bool = False) -> Tuple[float, float]:
        episode_rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward
