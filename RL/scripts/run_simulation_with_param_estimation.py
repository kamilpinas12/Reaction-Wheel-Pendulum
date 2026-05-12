import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt

from reac_wheel_sim.reaction_wheel_env import (
    ReactionWheelEnv,
)
from reac_wheel_sim.signal_generator import SquareSignal
from reac_wheel_sim.reaction_wheel_wrappers import *
from nets.parameter_estimator import ParamEstimatorLSTM
from utils.custom_paths import LOGS_DIR, MODELS_DIR
from utils.common import ParameterNormalizer, setup_file_logger


def setup_logger(log_dir: Path) -> logging.Logger:
    """Setup logger with file and console output."""
    return setup_file_logger(
        "run_simulation_with_param_estimation", log_dir / "simulation.log"
    )


def run_simulation_with_estimation(
    model,
    normalizer,
    device="cpu",
    seq_len=128,
    max_steps=1000,
    logger=None,
):
    """
    Run environment simulation and track parameter estimation over time.
    
    Args:
        model: Trained parameter estimator model
        normalizer: ParameterNormalizer instance
        device: Device to run model on
        seq_len: Sequence length for rolling window
        max_steps: Maximum simulation steps
        logger: Logger instance
        
    Returns:
        dict with simulation data and metrics
    """
    if logger is None:
        logger = logging.getLogger("run_simulation_with_param_estimation")

    # Initialize environment
    env = ReactionWheelEnv()
    env = ObservationNoiseWrapper(env, noise_levels=[0.00, 0.00, 0.00, 0.00, 0.00])

    signal_generator = SquareSignal(amplitude_range=(0.2, 0.6))
    action_sequence = signal_generator.generate(max_steps + seq_len, np.random.default_rng())

    # Sample random ground-truth parameters
    # K_sin_range = [-40, -4]
    # K_reac_wheel_range = [-0.5, -0.01]
    # K_pend_vel_range = [-0.5, -0.01]
    base_env = env.unwrapped
    true_params = np.array([
        base_env.K_sin,
        base_env.K_reac_wheel,
        base_env.K_pend_vel,
    ], dtype=np.float32)


    # logger.info(f"Ground-truth parameters: K_sin={true_params[0]:.4f}, "
    #             f"K_reac_wheel={true_params[1]:.4f}, K_pend_vel={true_params[2]:.4f}")

    # Initialize rolling window buffer
    obs_buffer = deque(maxlen=seq_len)
    model.eval()

    # Warm-up: collect first seq_len observations
    obs, info = env.reset()
    for _ in range(seq_len):
        obs_buffer.append(obs.copy())
        action = np.array([action_sequence[_]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    logger.info(f"Warm-up complete with {len(obs_buffer)} observations")

    # Simulation loop
    step_data = {
        "timesteps": [],
        "true_params": [],
        "estimated_params": [],
        "estimation_errors": [],
        "observations": [],
        "actions": [],
        "mae_per_param": [[], [], []],
        "rmse_per_param": [[], [], []],
    }

    for step in range(max_steps):
        # Get prediction from rolling window
        window_array = np.array(list(obs_buffer), dtype=np.float32)  # (seq_len, 5)
        window_tensor = torch.from_numpy(window_array).unsqueeze(0).to(device)  # (1, seq_len, 5)

        # Prepare lengths tensor for LSTM models that expect sequence lengths
        lengths_tensor = torch.tensor([window_tensor.size(1)], dtype=torch.long).to(window_tensor.device)

        with torch.no_grad():
            pred_normalized = model(window_tensor, lengths_tensor).squeeze(0).cpu().numpy()  # (3,)

        # Denormalize prediction
        pred_params = normalizer.denormalize(pred_normalized)

        # Compute error
        error = np.abs(true_params - pred_params)

        # Store step data
        step_data["timesteps"].append(step)
        step_data["true_params"].append(true_params.copy())
        step_data["estimated_params"].append(pred_params.copy())
        step_data["estimation_errors"].append(error)
        step_data["observations"].append(obs_buffer[-1].copy() if len(obs_buffer) > 0 else np.zeros(5))

        # Compute cumulative metrics
        all_errors = np.array(step_data["estimation_errors"])
        for param_idx in range(3):
            mae = np.mean(np.abs(all_errors[:, param_idx]))
            rmse = np.sqrt(np.mean(all_errors[:, param_idx] ** 2))
            step_data["mae_per_param"][param_idx].append(mae)
            step_data["rmse_per_param"][param_idx].append(rmse)

        # Step environment
        action_value = float(action_sequence[seq_len + step])
        action = np.array([action_value], dtype=np.float32)
        step_data["actions"].append(action_value)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_buffer.append(obs.copy())

        if terminated or truncated:
            logger.info(f"Episode terminated at step {step + 1}")
            break

        if (step + 1) % 100 == 0:
            logger.info(f"Step {step + 1}: MAE={step_data['mae_per_param'][0][-1]:.6f}, "
                        f"{step_data['mae_per_param'][1][-1]:.6f}, "
                        f"{step_data['mae_per_param'][2][-1]:.6f}")

    logger.info(f"Simulation completed at step {step + 1}")

    # Convert lists to arrays
    step_data["true_params"] = np.array(step_data["true_params"])
    step_data["estimated_params"] = np.array(step_data["estimated_params"])
    step_data["estimation_errors"] = np.array(step_data["estimation_errors"])
    step_data["observations"] = np.array(step_data["observations"])
    step_data["actions"] = np.array(step_data["actions"])

    return step_data, true_params, env


def visualize_parameter_evolution(step_data, true_params, save_dir: Path, logger=None):
    """
    Create visualization of parameter estimation evolution over time.
    
    Args:
        step_data: Dict with simulation data from run_simulation_with_estimation()
        true_params: Ground-truth parameters array
        save_dir: Directory to save visualization
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("run_simulation_with_param_estimation")

    save_dir.mkdir(parents=True, exist_ok=True)

    param_names = [r"$K_{sin}$", r"$K_{reac}$", r"$K_{vel}$"]
    timesteps = np.array(step_data["timesteps"])
    true_params_array = step_data["true_params"]
    estimated_params_array = step_data["estimated_params"]
    errors_array = step_data["estimation_errors"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Parameter Estimation Evolution During Simulation", fontsize=16, fontweight="bold")

    for param_idx in range(3):
        # Left column: Time-series
        ax_ts = axes[param_idx, 0]
        ax_ts.plot(timesteps, true_params_array[:, param_idx], "b-", linewidth=2, label="True", alpha=0.8)
        ax_ts.plot(timesteps, estimated_params_array[:, param_idx], "r--", linewidth=1.5, label="Estimated", alpha=0.8)
        ax_ts.axhline(y=true_params[param_idx], color="b", linestyle=":", alpha=0.5)
        ax_ts.set_ylabel(param_names[param_idx], fontsize=11, fontweight="bold")
        ax_ts.set_xlabel("Timestep", fontsize=10)
        ax_ts.legend(loc="best")
        ax_ts.grid(True, alpha=0.3)
        if param_idx == 0:
            ax_ts.set_title("Parameter Value Evolution", fontsize=12, fontweight="bold")

        # Right column: Error
        ax_err = axes[param_idx, 1]
        ax_err.plot(timesteps, errors_array[:, param_idx], "g-", linewidth=1.5, label="Absolute Error")
        final_mae = step_data["mae_per_param"][param_idx][-1]
        ax_err.axhline(y=final_mae, color="r", linestyle="--", linewidth=1.5, label=f"Final MAE: {final_mae:.4f}")
        ax_err.set_ylabel("Absolute Error", fontsize=10)
        ax_err.set_xlabel("Timestep", fontsize=10)
        ax_err.legend(loc="best")
        ax_err.grid(True, alpha=0.3)
        if param_idx == 0:
            ax_err.set_title("Estimation Error Over Time", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_path = save_dir / "parameter_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to: {save_path}")
    plt.close()

    # Create convergence summary plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Parameter Estimation Convergence", fontsize=14, fontweight="bold")

    for param_idx in range(3):
        mae_vals = step_data["mae_per_param"][param_idx]
        ax.plot(timesteps, mae_vals, "-o", linewidth=2, label=param_names[param_idx], markersize=3, alpha=0.7)

    ax.set_xlabel("Timestep", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE (Mean Absolute Error)", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    convergence_path = save_dir / "convergence_mae.png"
    plt.savefig(convergence_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved convergence plot to: {convergence_path}")
    plt.close()


def visualize_simulation_observations(step_data, save_dir: Path, logger=None):
    if logger is None:
        logger = logging.getLogger("run_simulation_with_param_estimation")

    save_dir.mkdir(parents=True, exist_ok=True)

    timesteps = np.array(step_data["timesteps"])
    observations = step_data["observations"]
    actions = step_data["actions"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Simulation Observations Trace", fontsize=16, fontweight="bold")

    obs_labels = [r"$\theta$", r"$\dot{\theta}$ (velocity)", r"$\phi$ (wheel)", r"$u_{prev}$ (action)"]
    

    axes[0].plot(timesteps, np.atan2(observations[:, 0], observations[:, 1]), linewidth=1.6)
    axes[0].set_ylabel(obs_labels[0])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps,observations[:, 3], linewidth=1.6)
    axes[1].set_ylabel(obs_labels[1])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timesteps,observations[:, 4], linewidth=1.6)
    axes[2].set_ylabel(obs_labels[2])
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(timesteps, actions, color="tab:orange", linewidth=1.4)
    axes[3].set_ylabel("Action Command")
    axes[3].set_xlabel("Timestep")
    axes[3].grid(True, alpha=0.3)

    obs_path = save_dir / "simulation_observations.png"
    plt.tight_layout()
    plt.savefig(obs_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved observation trace plot to: {obs_path}")
    plt.close()


def main():
    # Setup
    run_log_dir = LOGS_DIR / "run_simulation_with_param_estimation"
    logger = setup_logger(run_log_dir)

    logger.info("=" * 70)
    logger.info("Starting Parameter Estimation Simulation")
    logger.info("=" * 70)

    # Configuration
    seq_len = 128
    max_steps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # HYPERPARAMETERS
    hparams = {
        "model_arch": "LSTM",
        "min_seq_len": 32,
        "max_seq_len": 128,
        "n_episodes": 2000,
        "lr": 1e-3,
        "epochs": 40,
        "hidden_dim": 32,
        "fc_dim": 64,
        "batch_size": 32
    }

    param_noise_pct = 0.1
    mass_pos_pct = 0.1
    noise_levels=[0.0, 0.0, 0.0, 0.0, 0.0]

    env = ReactionWheelEnv()
    randomizer = ParamRandomizationWrapper(
        env,
        param_noise_pct=param_noise_pct,
        mass_pos_pct=mass_pos_pct,
    )
    target_min_vals, target_max_vals = randomizer.get_ground_truth_bounds()
    env = randomizer
    env = ObservationNoiseWrapper(env, noise_levels=noise_levels)

    checkpoint_path =  MODELS_DIR / "best_model.pt"
    logger.info("Loading model from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model
    model = ParamEstimatorLSTM(
        n_features=5, n_params=3, hidden_dim=hparams["hidden_dim"], fc_dim=hparams["fc_dim"]
    )

    # Support multiple checkpoint formats:
    # - {'model_state_dict': ...}
    # - PyTorch Lightning checkpoint with 'state_dict' where keys may be prefixed with 'model.'
    # - plain state_dict
    try:
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict)
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state = checkpoint["state_dict"]
                # Strip 'model.' prefix if present (Lightning stores lit-module params under 'model.*')
                if any(k.startswith("model.") for k in state.keys()):
                    stripped = {k[len("model."):]: v for k, v in state.items()}
                    model.load_state_dict(stripped)
                else:
                    model.load_state_dict(state)
            else:
                # Maybe the checkpoint is itself a state_dict
                model.load_state_dict(checkpoint)
        else:
            # Fallback: assume loaded object is a state_dict
            model.load_state_dict(checkpoint)
    except Exception as e:
        logger.exception("Failed to load checkpoint. Checkpoint keys: %s", list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint))
        raise RuntimeError(f"Could not load model checkpoint: {e}")

    model.to(device)
    logger.info(f"Model loaded successfully | Device: {device}")

    # Initialize normalizer
    normalizer = ParameterNormalizer(
        min_vals=target_min_vals,
        max_vals=target_max_vals,
    )
    logger.info("Normalizer initialized")

    # Run simulation
    logger.info("-" * 70)
    logger.info("Running simulation with rolling window parameter estimation...")
    logger.info("-" * 70)
    step_data, true_params, env = run_simulation_with_estimation(
        model=model,
        normalizer=normalizer,
        device=device,
        seq_len=seq_len,
        max_steps=max_steps,
        logger=logger,
    )

    # Compute final metrics
    final_errors = step_data["estimation_errors"][-1]
    final_mae = np.mean(final_errors)
    final_rmse = np.sqrt(np.mean(final_errors ** 2))

    logger.info("-" * 70)
    logger.info("Final Metrics:")
    logger.info(f"  True Parameters:      K_sin={true_params[0]:.4f}, K_reac={true_params[1]:.4f}, K_vel={true_params[2]:.4f}")
    logger.info(f"  Final Estimate:       K_sin={step_data['estimated_params'][-1][0]:.4f}, "
                f"K_reac={step_data['estimated_params'][-1][1]:.4f}, K_vel={step_data['estimated_params'][-1][2]:.4f}")
    logger.info(f"  Final MAE (per param): {step_data['mae_per_param'][0][-1]:.6f}, "
                f"{step_data['mae_per_param'][1][-1]:.6f}, {step_data['mae_per_param'][2][-1]:.6f}")
    logger.info(f"  Final RMSE:           {final_rmse:.6f}")
    logger.info("-" * 70)

    # Visualize
    logger.info("Creating visualizations...")
    visualize_parameter_evolution(step_data, true_params, run_log_dir, logger=logger)
    visualize_simulation_observations(step_data, run_log_dir, logger=logger)

    logger.info("=" * 70)
    logger.info("Simulation and visualization completed!")
    logger.info(f"Results saved to: {run_log_dir}")
    logger.info("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
