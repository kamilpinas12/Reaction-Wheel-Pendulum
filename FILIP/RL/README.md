**Reaction Wheel — Python Algorithm**

**Overview:**
- The package contains a small RL training pipeline using Stable Baselines 3. The core pieces are the agent implementation ([agents/dqn_agent.py](agents/dqn_agent.py#L1)), the custom pendulum-like environment ([envs/pendulum_env.py](envs/pendulum_env.py#L1)), and the lightweight launcher ([main.py](main.py#L1)).

**Prerequisites:**
- Python 3.8+ with a virtual environment. Ensure `stable-baselines3`, `gymnasium`, `numpy`, and `pyyaml` are installed in `venv`.

**Quick start:**
- Activate the virtualenv and run training:

```bash
source venv/bin/activate
cd python_algorithm
python main.py
```

- Checkpoints are written to the directory defined in `config.yaml` under `training.output_dir` (default `./checkpoints`).

**Configuration:**
- Edit `config.yaml` at the repository root. Key groups:
  - **base_agent**: `learning_rate`, `gamma`, `seed`, `device`.
  - **dqn_agent**: algorithm hyperparameters (buffer size, epsilons, policy etc.).
  - **training**: `total_timesteps`, `save_interval`, `output_dir`, and `n_actions` (for discrete env).

**Environment choices:**
- The environment is in [envs/pendulum_env.py](envs/pendulum_env.py#L1). It currently exposes a discrete action space mapped to a small set of torques (default 7 values spanning -0.9..0.9). To use continuous control, swap the agent and call the continuous env instead.

**Files of interest:**
- Agent: [agents/dqn_agent.py](agents/dqn_agent.py#L1)
- Env: [envs/pendulum_env.py](envs/pendulum_env.py#L1)
- Launcher: [main.py](main.py#L1)
- Config: [config.yaml](config.yaml#L1)
