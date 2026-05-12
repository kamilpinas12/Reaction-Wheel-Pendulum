from pathlib import Path
import os
import shutil

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import DiscretizeActionWrapper
from reac_wheel_sim.reward_wrappers import FilipRewardWrapper
from utils.config_manager import cfg_get
from utils.callbacks import SaveCallback, RewardPlottingCallback, EpisodeResetCallback, RolloutCallback
from utils.custom_paths import MODELS_DIR, CONFIGS_DIR
from agents.dqn_agent import DQNAgent
from utils.common import setup_file_logger



def main():
	config_name = 'config_dqn.yaml'
	training = cfg_get('training', config_name)
	total = training.get('total_timesteps')
	save_freq = int(training.get('save_interval'))
	output_dir = Path(training.get('output_dir'))
	save_dir = output_dir

	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	env = ReactionWheelEnv(config_name=config_name)
	env = DiscretizeActionWrapper(env, 7)
	env = FilipRewardWrapper(env)
	agent = DQNAgent(env)
	cb = SaveCallback(save_freq, save_dir)
	rollout_cb = RolloutCallback(env, save_freq, save_dir)
	plot_cb = RewardPlottingCallback(plot_freq=1000, plot_dir=save_dir)
	reset_cb = EpisodeResetCallback()

	agent.train(total_timesteps=total, callback=[cb, rollout_cb, plot_cb, reset_cb])
	agent.save(str(save_dir / 'final'))
	print('Training finished — artifacts in', save_dir)


if __name__ == '__main__':
	main()
