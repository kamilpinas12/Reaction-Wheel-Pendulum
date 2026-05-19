import numpy as np
from gymnasium.wrappers import RecordVideo

from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from utils.custom_paths import LOGS_DIR


def test_reaction_wheel_env_records_video():
    video_dir = LOGS_DIR / "pytest"
    env = ReactionWheelEnv(config_name="config_ppo.yaml", render_mode="rgb_array")
    wrapped = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode: True,
        name_prefix="reaction_wheel",
    )

    try:
        wrapped.reset(seed=0, options={"initial_state": [0.2, 0.0, 0.0]})
        done = False
        while not done:
            _, _, terminated, truncated, _ = wrapped.step(np.array([-1], dtype=np.float32))
            done = terminated or truncated
    finally:
        wrapped.close()

    video_files = list(video_dir.rglob("*.mp4"))
    assert video_files, "No video files were written"
    assert all(video_file.stat().st_size > 0 for video_file in video_files)