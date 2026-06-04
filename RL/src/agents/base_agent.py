from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
from PIL import Image

from utils.config_manager import cfg_get
from utils.rollout import RolloutData, plot_rollout


class BaseAgent(ABC):
    def __init__(self, env=None, logger=None, config_name="config.yaml"):
        self.env = env
        self.cfg_name = config_name
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.learning_rate = float(cfg_get('base_agent.learning_rate', self.cfg_name, 3e-4))
        self.lr_scheduling_final = cfg_get('base_agent.learning_rate_scheduling_end', self.cfg_name, None)
        if self.lr_scheduling_final is not None:
            self.lr_scheduling_final = float(self.lr_scheduling_final)
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

    def _log_scalar(self, tag: str, value: float) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, self.num_timesteps)

    def _extract_env_action(self, action: Union[np.ndarray, float, int]):
        action_arr = np.asarray(action)
        if action_arr.shape == ():
            return action_arr.item()
        if action_arr.shape[0] == 1:
            return action_arr[0]
        return action_arr

    def _get_initial_states(self) -> List[List[float]]:
        return [
            [0.0, 0.0, 0.0],
            [np.pi, 0.0, 0.0],
            [np.pi / 2, 0.0, 0.0],
            [-np.pi / 2, 0.0, 0.0],
            [np.pi, 4.0, 100.0],
            [np.pi, -4.0, -100.0],
        ]
    
    def _get_model_params(self):
        return [
            {"K_pend_vel": 0.1165, "K_sin": -3.915, "K_reac_wheel": -0.00786},
            {"K_pend_vel": 0.113, "K_sin": -27.233, "K_reac_wheel": -0.001154}, # bez ciezarka
            {"K_pend_vel": 0.085449, "K_sin": -11.4982, "K_reac_wheel": -0.009128}, # pos1
            {"K_pend_vel": 0.08683, "K_sin": -6.959842, "K_reac_wheel":  -0.008187} # pos3
        ]
    
    def _override_model_params(self, new_params):
        self.env.unwrapped.K_pend_vel = new_params["K_pend_vel"]
        self.env.unwrapped.K_sin = new_params["K_sin"]
        self.env.unwrapped.K_reac_wheel = new_params["K_reac_wheel"]
    
    def _fmt_params(self, p):
        return "{" + ", ".join(
                f"{k}: {v:.4f}" for k, v in p.items()
            ) + "}"

    @abstractmethod
    def train(self, total_timesteps: Optional[int] = None, **kwargs) -> Dict[str, Any]:
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
        num_episodes = cfg_get('base_agent.eval.num_episodes', self.cfg_name, 4)
        num_models = cfg_get('base_agent.eval.num_models', self.cfg_name, 1)
        deterministic_eval = cfg_get('base_agent.eval.deterministic', self.cfg_name, True)
        self.logger.info(f"Evaluation begin on {num_episodes} episodes...")

        initial_states = self._get_initial_states()
        model_params = self._get_model_params()

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
        

        if self.model is not None:
            self.model.eval()
        
        for model_idx in range(num_models):
            if should_plot:
                data = RolloutData()

            episode_lengths = []
            episode_pos_err_list = []
            episode_energy_used_list = []
            for episode_idx in range(num_episodes):
                state = initial_states[episode_idx % len(initial_states)]
                options = {
                    "initial_state": state,
                    "model_params": model_params[model_idx],
                    }
                obs, info = eval_env.reset(options=options)
                
                format_params = lambda p : "{" + ", ".join(f"{k}: {v:.4f}" for k, v in p.items()) + "}"
                self.logger.info(f"Model params: {format_params(info.get('model_params', None))}")
                self.logger.info(f"Physical params: {format_params(info.get('phys_params', None))}")

                episode_length = 0
                done = False
                episode_pos_err = 0.0
                episode_energy_used = 0.0
                
                while not done:
                    action, _ = self.predict(obs, deterministic=deterministic_eval)
                    env_action = self._extract_env_action(action)
                    obs, reward, terminated, truncated, info = eval_env.step(env_action)
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
            
            self._log_scalar('eval/mean_pos_err', mean_pos_err)
            self._log_scalar('eval/sum_energy', sum_energy)
            self._log_scalar('eval/mean_length', mean_length)

            if should_plot:
                path = plot_rollout(data, self.output_dir / f'agent_rollout_model{model_idx}_{self.num_timesteps}.png')
                self.logger.info(f'saved agent rollout plot to {path}')

            if self.writer is not None:
                img = Image.open(path)
                img_array = np.array(img)
                self.writer.add_image('eval/rollout_plot', img_array, self.num_timesteps, dataformats='HWC')

            self.save(self.output_dir / "checkpoints" / f"ckp_score={mean_pos_err}_iter={self.num_timesteps}.pth")
            
        return mean_pos_err, std_pos_err
