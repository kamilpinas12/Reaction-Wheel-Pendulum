from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
import os
from pathlib import Path
from PIL import Image

from utils.config_manager import cfg_get
from utils.rollout import RolloutData, plot_rollout


class BaseAgent(ABC):
    def __init__(self, env=None, logger=None, config_name="config.yaml"):
        self.env = env
        self.cfg_name = config_name
        self.logger = logger

        self.learning_rate = float(cfg_get('base_agent.learning_rate', self.cfg_name))
        self.gamma = cfg_get('base_agent.gamma', self.cfg_name, default=0.99)
        self.seed = cfg_get('base_agent.seed', self.cfg_name)
        self.output_dir = Path(cfg_get('base_agent.output_dir', self.cfg_name))
        self.device = cfg_get('base_agent.device', self.cfg_name)
        self.total_timesteps = cfg_get('base_agent.total_timesteps', self.cfg_name)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if cfg_get('base_agent.writer_enable', self.cfg_name):
            alg_name = self.__class__.__name__
            self.writer = SummaryWriter(comment=alg_name)
        else:
            self.writer = None

        self.num_timesteps = 0
        self.model = None

    @staticmethod
    def _angle_normalize(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
   
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        if self.model is None:
            self.logger.error("No model to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to: {path}")
    
    def load(self, path: str) -> None:
        if self.model is None:
            self.logger.error("Model not found")
            return
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.logger.info(f"Model loaded from: {path}")
    
    def evaluate(self, eval_env, render: bool = False) -> Tuple[float, float]:
        num_episodes = cfg_get('base_agent.eval.num_episodes', self.cfg_name, 10)
        deterministic_eval = cfg_get('base_agent.eval.deterministic', self.cfg_name, True)
        self.logger.info(f"Evaluation begin on {num_episodes} episodes...")

        initial_states = [[0.0, 0.0, 0.0],
                          [np.pi, 0.0, 0.0],
                          [np.pi/2, 0.0, 0.0],
                          [-np.pi/2, 0.0, 0.0],
                          [np.pi, 1.0, 100.0],
                          [np.pi, -1.0, -100.0]]

        should_record = cfg_get('base_agent.eval.record', self.cfg_name)
        if should_record:
            run_video_path = self.output_dir / "videos" / f"step_{self.num_timesteps}"
            eval_env = RecordVideo(
                eval_env, 
                video_folder=run_video_path,
                episode_trigger=lambda x: x == 0,
                name_prefix="eval_video"
            )

        should_plot = cfg_get('base_agent.eval.plot', self.cfg_name)
        if should_plot:
            data = RolloutData()

        episode_lengths = []
        episode_pos_err_list = []
        episode_energy_used_list = []

        if self.model is not None:
            self.model.eval()
        
        for episode_idx in range(num_episodes):
            options = {"initial_state": initial_states[episode_idx]}
            obs, _ = eval_env.reset(options=options)
            episode_length = 0
            done = False
            episode_pos_err = 0.0
            episode_energy_used = 0.0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic_eval)
                obs, reward, terminated, truncated, info = eval_env.step(action[0])
                done = terminated or truncated
                episode_length += 1

                err = ((info["theta"] % (2 * np.pi)) - np.pi)**2
                episode_pos_err += err
                episode_energy_used += np.abs(info["u_cmd"])
                
                if render:
                    eval_env.render()
                if should_plot and not done:
                    data.add(timestep=episode_length, action=action.item(), reward=reward, info=info)

            episode_lengths.append(episode_length)
            episode_pos_err_list.append(episode_pos_err)
            episode_energy_used_list.append(episode_energy_used)
            
            if should_plot and episode_idx < num_episodes - 1:
                data.add_separator(episode_length)
            
        mean_pos_err = np.mean(episode_pos_err_list)
        std_pos_err = np.std(episode_pos_err_list)
        sum_energy = np.sum(episode_energy_used_list)
        mean_length = np.mean(episode_lengths)
        
        self.logger.info(f"Evaluation results: Position error: {mean_pos_err:.2f} ± {std_pos_err:.2f} | Length: {mean_length:.1f}")
        
        if self.writer is not None:
            self.writer.add_scalar('eval/mean_pos_err', mean_pos_err, self.num_timesteps)
            self.writer.add_scalar('eval/sum_energy', sum_energy, self.num_timesteps)
            self.writer.add_scalar('eval/mean_len   gth', mean_length, self.num_timesteps)

        if should_plot:
            path = plot_rollout(data, self.output_dir / f'agent_rollout_{self.num_timesteps}.png')
            self.logger.info(f'saved agent rollout plot to {path}')

            if self.writer is not None:
                img = Image.open(path)
                img_array = np.array(img)
                self.writer.add_image('eval/rollout_plot', img_array, self.num_timesteps, dataformats='HWC')
            
        return mean_pos_err, std_pos_err
