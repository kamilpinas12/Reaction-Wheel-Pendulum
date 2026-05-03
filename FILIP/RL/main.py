from pathlib import Path

from envs.pendulum_env import ReactionWheelEnv
from utils.config_manager import get as cfg_get
from utils.callbacks import SaveCallback, RewardPlottingCallback
from agents.dqn_agent import DQNAgent



def main():
	training = cfg_get('training')
	total = training.get('total_timesteps')
	save_dir = Path(training.get('output_dir'))
	save_freq = int(training.get('save_interval'))
	save_dir.mkdir(parents=True, exist_ok=True)

	env = ReactionWheelEnv()
	agent = DQNAgent(env)
	cb = SaveCallback(save_freq, save_dir)
	plot_cb = RewardPlottingCallback(plot_freq=1000, plot_dir=save_dir)

	agent.train(total_timesteps=total, callback=[cb, plot_cb])
	agent.save(str(save_dir / 'final'))
	agent.export_matlab_weights(str(save_dir / 'dqn_weights.mat'))
	print('Training finished — artifacts in', save_dir)


if __name__ == '__main__':
	main()
