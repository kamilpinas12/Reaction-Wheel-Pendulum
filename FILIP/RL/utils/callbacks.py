from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

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
        if rewards is not None and len(rewards) > 0:
            self.step_rewards.append(float(rewards[0]))
            self.step_timesteps.append(self.num_timesteps)

        # Get last episode reward from info
        if len(self.model.ep_info_buffer) > 0:
            last_reward = self.model.ep_info_buffer[-1]["r"]
            if len(self.model.ep_info_buffer) > self._episode_count:
                self._episode_count = len(self.model.ep_info_buffer)
                self.episode_ids.append(self._episode_count)
                self.episode_rewards.append(float(last_reward))
        
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