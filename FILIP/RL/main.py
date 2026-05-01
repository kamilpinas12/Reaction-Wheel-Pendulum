
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

import utils.config_manager as cfgm
from envs.pendulum_env import ReactionWheelEnv
from agents.dqn_agent import DQNAgent


def _normalize_device():
	d = cfgm.get('base_agent.device')
	if d == 'gpu':
		cfgm.CONFIG.setdefault('base_agent', {})['device'] = 'cuda'


def main():
	_normalize_device()

	env = ReactionWheelEnv()
	cfgm.CONFIG.setdefault('base_agent', {})['env'] = env

	training = cfgm.get('training') or {}
	total = training.get('total_timesteps', 100000)
	save_dir = Path(training.get('output_dir', './checkpoints'))
	save_freq = int(training.get('save_interval', 10000) or 0)
	save_dir.mkdir(parents=True, exist_ok=True)

	class SaveCallback(BaseCallback):
		def __init__(self, freq: int, out: Path):
			super().__init__(verbose=0)
			self.freq = max(0, int(freq))
			self.out = out

		def _on_step(self) -> bool:
			if self.freq and self.num_timesteps and self.num_timesteps % self.freq == 0:
				self.model.save(str(self.out / f"step_{self.num_timesteps}"))
			return True

	agent = DQNAgent()
	cb = SaveCallback(save_freq, save_dir)

	agent.train(total_timesteps=total, callback=cb)
	agent.save(str(save_dir / 'final'))
	print('Training finished — artifacts in', save_dir)


if __name__ == '__main__':
	main()
