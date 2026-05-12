from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import DiscretizeActionWrapper
from utils.rollout import RolloutData, plot_rollout

class EpisodeResetCallback(BaseCallback):
	"""Callback that resets the environment between episodes."""
	def _on_step(self) -> bool:
		dones = self.locals.get('dones')
		if dones is not None and dones[0]:
			# Episode ended, reset the environment
			self.model.env.reset()
		return True


class RewardPlottingCallback(BaseCallback):
    def __init__(self, plot_freq: int = 1000, plot_dir: Path = Path('./plots')):
        super().__init__(verbose=0)
        self.plot_freq = plot_freq
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.step_rewards = []
        self.step_timesteps = []
        self.episode_rewards = []
        self.episode_ids = []
        self._episode_count = 0
    
    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if rewards is not None and len(rewards) > 0:
            self.step_rewards.append(float(rewards[0]))
            self.step_timesteps.append(self.num_timesteps)

        if dones is not None and infos is not None and len(dones) > 0 and bool(dones[0]):
            info = infos[0]
            episode = info.get('episode') if isinstance(info, dict) else None
            if episode is not None:
                self._episode_count += 1
                self.episode_ids.append(self._episode_count)
                self.episode_rewards.append(float(episode['r']))
        
        # Update plot periodically
        if self.num_timesteps % self.plot_freq == 0 and (self.step_rewards or self.episode_rewards):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

            if self.step_rewards:
                axes[0].plot(self.step_timesteps, self.step_rewards, 'b-', linewidth=1)
                axes[0].set_xlabel('Timesteps')
                axes[0].set_ylabel('Reward')
                axes[0].set_title('Reward vs Timesteps')
                axes[0].grid(True, alpha=0.3)

            if self.episode_rewards:
                axes[1].plot(self.episode_ids, self.episode_rewards, 'g-', linewidth=1.5)
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Episode Reward')
                axes[1].set_title('Reward vs Episode')
                axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            plt.savefig(str(self.plot_dir / 'reward_plot.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        return True


class SaveCallback(BaseCallback):
		def __init__(self, freq: int, out: Path):
			super().__init__(verbose=0)
			self.freq = max(0, int(freq))
			self.out = out

		def _on_step(self) -> bool:
			if self.freq and self.num_timesteps and self.num_timesteps % self.freq == 0:
				self.model.save(str(self.out / f"step_{self.num_timesteps}"))
			return True


class RolloutCallback(BaseCallback):
    def __init__(self, env, freq: int, out: Path, seed: int = 123):
        super().__init__(verbose=0)
        self.freq = max(0, int(freq))
        self.out = Path(out)
        self.seed = int(seed)
        self.env = env

    def _on_step(self) -> bool:
        if not (self.freq and self.num_timesteps and self.num_timesteps % self.freq == 0):
            return True
        obs, _ = self.env.reset(seed=self.seed)
        data = RolloutData()
        done = False
        timestep = 0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            data.add(timestep=timestep, action=float(action), reward=reward, info=info)
            done = terminated or truncated
            timestep += 1
        path = plot_rollout(data, self.out / f'agent_rollout_{self.num_timesteps}.png')
        print(f'saved agent rollout plot to {path}')
        return True