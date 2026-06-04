from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from agents.base_agent import BaseAgent
from utils.rollout_buffer import RolloutBuffer
from utils.config_manager import cfg_get

class PPOAgent(BaseAgent):
    def __init__(self, env, model, logger, config_name="config_ppo.yaml"):
        super().__init__(env, logger, config_name=config_name)
        
        self.model = model.to(self.device)
        
        self.value_coef = cfg_get('ppo_agent.value_coef', self.cfg_name, default=0.5)
        self.entropy_coef = cfg_get('ppo_agent.entropy_coef', self.cfg_name, default=0.0)
        self.gae_lambda = cfg_get('ppo_agent.gae_lambda', self.cfg_name, default=0.95)
        self.max_grad_norm = cfg_get('ppo_agent.max_grad_norm', self.cfg_name, default=0.5)
        self.n_rollout_steps = cfg_get('ppo_agent.n_rollout_steps', self.cfg_name, default=2048)
        self.n_epochs = cfg_get('ppo_agent.n_epochs', self.cfg_name, default=10)
        self.batch_size = cfg_get('ppo_agent.batch_size', self.cfg_name, default=64)
        self.clip_range = cfg_get('ppo_agent.clip_range', self.cfg_name, default=0.2)
        self.eval_interval = cfg_get('base_agent.eval.interval', self.cfg_name, default=50000)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        obs_shape = self.env.single_observation_space.shape
        act_shape = self.env.single_action_space.shape
        
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_rollout_steps,
            num_envs=self.env.num_envs,
            obs_shape=obs_shape,
            act_shape=act_shape,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        self.ep_info_buffer = deque(maxlen=100)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            mu, var, _ = self.model(obs_tensor)

        if deterministic:
            action = mu
        else:
            std = torch.sqrt(var + 1e-8)
            dist = Normal(mu, std)
            action = dist.sample()
            
        return action.cpu().numpy(), None

    def train(self, total_timesteps: Optional[int] = None, eval_env=None) -> dict:
        self.logger.info("Rozpoczynam trening PPO...")
        last_eval_step = 0
        best_mean_reward = -float('inf')
        obs, _ = self.env.reset()
        
        while self.num_timesteps < self.total_timesteps:
            if self.lr_scheduling_final is not None:
                progress = self.num_timesteps / self.total_timesteps
                current_lr = self.learning_rate + progress * (self.lr_scheduling_final - self.learning_rate)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # 1. FAZA ZBIERANIA DANYCH (Teraz zbieramy tysiące kroków naraz)
            self.model.eval()
            self.rollout_buffer.reset()

            for step in range(self.n_rollout_steps):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    mu, var, value = self.model(obs_tensor)
                    std = torch.sqrt(var + 1e-8)
                    dist = Normal(mu, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)

                action_np = action.cpu().numpy()
                next_obs, rewards, terminated, truncated, infos = self.env.step(action_np)

                # Epizod ucięty przez czas. next_obs to stan zresetowany.
                # Musimy pobrać PRAWDZIWY stan końcowy, zanim nastąpił reset.
                if "_final_observation" in infos:
                    for i, is_final in enumerate(infos["_final_observation"]):
                        if is_final and truncated[i]:
                            true_next_obs = infos["final_observation"][i]
                            true_next_obs_tensor = torch.FloatTensor(true_next_obs).to(self.device).unsqueeze(0)
                            
                            with torch.no_grad():
                                _, _, true_value = self.model(true_next_obs_tensor)
                            rewards[i] += self.gamma * true_value.item()
                            terminated[i] = True
 
                self.rollout_buffer.add(
                    obs, action_np, rewards, terminated, value.flatten(), log_prob
                )

                obs = next_obs
                self.num_timesteps += self.env.num_envs

                if "_episode" in infos:
                    finished_envs = np.where(infos["_episode"])[0]
                    for idx in finished_envs:
                        ep_rew = infos["episode"]["r"][idx]
                        ep_len = infos["episode"]["l"][idx]
                        self.ep_info_buffer.append(ep_rew)

            if len(self.ep_info_buffer) > 0:
                mean_reward = np.mean(self.ep_info_buffer)
                self._log_scalar('rollout/ep_rew_mean', mean_reward)

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                _, _, next_value = self.model(obs_tensor)
                
            self.rollout_buffer.compute_returns_and_advantages(last_values=next_value, terminated=terminated)

            # 2. FAZA AKTUALIZACJI SIECI
            self.model.train()
            epoch_actor_losses, epoch_critic_losses, epoch_grad_norms = [], [], []
            epoch_approx_kls, epoch_clip_fracs = [], []

            for epoch in range(self.n_epochs):
                for batch in self.rollout_buffer.get(batch_size=self.batch_size):
                    mu, var, values = self.model(batch.observations)
                    std = torch.sqrt(var + 1e-8)
                    dist = Normal(mu, std)
                    
                    log_probs = dist.log_prob(batch.actions).sum(dim=-1)
                    entropies = dist.entropy().sum(dim=-1)

                    advantages = batch.advantages
                    
                    # ppo clipping
                    ratio = torch.exp(log_probs - batch.old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    approx_kl = (batch.old_log_probs - log_probs).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()

                    # value clipping
                    v_pred = values.flatten()
                    v_targets = batch.returns.flatten()
                    v_old = batch.old_values.flatten() # Wymaga, aby Twój buffer to zwracał

                    # Standardowa strata MSE
                    v_loss_unclipped = F.mse_loss(v_pred, v_targets, reduction='none')

                    # Strata z obcięciem (wartość nie może odskoczyć dalej niż o clip_range od starej wartości)
                    v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -self.clip_range, self.clip_range)
                    v_loss_clipped = F.mse_loss(v_pred_clipped, v_targets, reduction='none')

                    # Wybieramy gorszy przypadek (max), aby pesymistycznie podejść do estymacji
                    critic_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropies.mean()
                    loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    epoch_actor_losses.append(actor_loss.item())
                    epoch_critic_losses.append(critic_loss.item())
                    epoch_grad_norms.append(float(grad_norm))
                    epoch_approx_kls.append(float(approx_kl.detach()))
                    epoch_clip_fracs.append(float(clip_fraction))

            avg_actor_loss = np.mean(epoch_actor_losses)
            avg_critic_loss = np.mean(epoch_critic_losses)
            avg_grad_norm = np.mean(epoch_grad_norms)
            avg_approx_kl = np.mean(epoch_approx_kls)
            avg_clip_frac = np.mean(epoch_clip_fracs)
            
            self._log_scalar('train/learning_rate', current_lr)
            self._log_scalar('train/actor_loss', avg_actor_loss)
            self._log_scalar('train/value_loss', avg_critic_loss)
            self._log_scalar('train/grad_norm', avg_grad_norm)
            self._log_scalar('train/approx_kl', avg_approx_kl)
            self._log_scalar('train/clip_fraction', avg_clip_frac)
                
            self.logger.info(f"iter: {self.num_timesteps}, lr: {current_lr:.2e}, critic_loss: {avg_critic_loss:.4f}, actor_loss: {avg_actor_loss:.4f}")
            
            # 3. FAZA EWALUACJI I ZAPISU 
            if eval_env is not None and (self.num_timesteps - last_eval_step) >= self.eval_interval:
                mean_reward, std_reward = self.evaluate(eval_env)
                last_eval_step = self.num_timesteps
                
                # if mean_reward > best_mean_reward:
                #     best_mean_reward = mean_reward
                #     best_model_path = str(self.output_dir / "best_model.pth")
                #     self.save(best_model_path)
                #     self.logger.info(f"New best model saved! mean reward: {mean_reward:.2f}")
                
                # for now not needed to save checkpoints
                # checkpoint_path = str(self.output_dir / f"checkpoint_{self.num_timesteps}.pth")
                # self.save(checkpoint_path)

        self.logger.info("Trening zakończony!")
        return {"total_timesteps": self.num_timesteps, "status": "completed"}