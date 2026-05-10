from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN

from agents.base_agent import BaseAgent
from utils.config_manager import cfg_get


class DQNAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env,config_name="config_dqn.yaml")
        g = lambda k: cfg_get(f'dqn_agent.{k}', self.cfg_name)
        self.policy, self.buffer_size, self.learning_starts = g('policy'), g('buffer_size'), g('learning_starts')
        self.batch_size, self.target_update_interval = g('batch_size'), g('target_update_interval')
        self.exploration_fraction, self.exploration_initial_eps = g('exploration_fraction'), g('exploration_initial_eps')
        self.exploration_final_eps, self.tau, self.train_freq = g('exploration_final_eps'), g('tau'), g('train_freq')

        self.model = DQN(policy=self.policy, env=self.env, learning_rate=self.learning_rate,
                         buffer_size=self.buffer_size, learning_starts=self.learning_starts,
                         batch_size=self.batch_size, tau=self.tau, gamma=self.gamma,
                         train_freq=self.train_freq, target_update_interval=self.target_update_interval,
                         exploration_fraction=self.exploration_fraction,
                         exploration_initial_eps=self.exploration_initial_eps,
                         exploration_final_eps=self.exploration_final_eps,
                         verbose=0, seed=self.seed, device=self.device)

    def train(self, total_timesteps: int, log_interval: int = 1000, callback=None) -> dict:
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callback)
        return {"total_timesteps": total_timesteps, "status": "completed"}

    def predict(self, observation: np.ndarray, deterministic: bool = True,
                state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.model.predict(observation, state=state, deterministic=deterministic)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p))

    def load(self, path: str) -> None:
        self.model = DQN.load(path, env=self.env)

    def get_exploration_epsilon(self) -> float:
        return getattr(self.model, 'exploration_rate', 0.0)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        setattr(self.model, 'learning_rate', learning_rate)
