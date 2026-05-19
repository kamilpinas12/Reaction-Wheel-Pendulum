import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np

from utils.custom_paths import LOGS_DIR


def _get_logger(log_dir=None):
    if log_dir is None:
        log_dir = LOGS_DIR / "train_param_estimator"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_param_estimator")
    logger.setLevel(logging.INFO)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    )
    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not has_file_handler:
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

def evaluate_and_visualize_model(
    model,
    test_loader,
    normalizer=None,
    n_params=3,
    device="cpu",
    save_dir=None,
    logger=None,
    show=False,
):
    if logger is None:
        logger = _get_logger()

    if save_dir is None:
        save_dir = LOGS_DIR / "train_param_estimator" / "results"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for x_batch, y_batch, lengths in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch, lengths)
            
            all_predictions.append(y_pred.cpu().numpy())
            all_ground_truth.append(y_batch.cpu().numpy())
    
    predictions_norm = np.concatenate(all_predictions, axis=0)
    ground_truth_norm = np.concatenate(all_ground_truth, axis=0)
    
    # Denormalize if normalizer provided
    if normalizer is not None:
        predictions = normalizer.denormalize(predictions_norm)
        ground_truth = normalizer.denormalize(ground_truth_norm)
    else:
        predictions = predictions_norm
        ground_truth = ground_truth_norm
    
    logger.info("Evaluation on %d test samples", len(ground_truth))
    
    # Compute metrics
    mae_per_param = np.mean(np.abs(predictions - ground_truth), axis=0)
    rmse_per_param = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=0))
    
    # Compute R² for each parameter
    r2_per_param = []
    for i in range(n_params):
        ss_res = np.sum((ground_truth[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((ground_truth[:, i] - np.mean(ground_truth[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_per_param.append(r2)
    r2_per_param = np.array(r2_per_param)
    
    param_names = ["K_sin", "K_reac_wheel", "K_pend_vel"]
    
    # Log metrics
    logger.info("=" * 60)
    logger.info("Evaluation Metrics (Original Scale)")
    logger.info("=" * 60)
    for i, name in enumerate(param_names):
        logger.info(
            "%15s | MAE: %8.4f | RMSE: %8.4f | R2: %7.4f",
            name,
            mae_per_param[i],
            rmse_per_param[i],
            r2_per_param[i],
        )
    
    logger.info("=" * 60)
    logger.info("Overall Metrics")
    logger.info("=" * 60)
    logger.info("Mean MAE:  %.6f", np.mean(mae_per_param))
    logger.info("Mean RMSE: %.6f", np.mean(rmse_per_param))
    logger.info("Mean R2:   %.6f", np.mean(r2_per_param))
    
    # Visualizations
    fig = plt.figure(figsize=(14, 10))
    
    # Scatter plots: predicted vs actual
    for i in range(n_params):
        ax = plt.subplot(2, 3, i + 1)
        ax.scatter(ground_truth[:, i], predictions[:, i], alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(ground_truth[:, i].min(), predictions[:, i].min())
        max_val = max(ground_truth[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")
        
        ax.set_xlabel("Ground Truth", fontsize=10)
        ax.set_ylabel("Prediction", fontsize=10)
        ax.set_title(f"{param_names[i]}\nR² = {r2_per_param[i]:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # Error histograms
    for i in range(n_params):
        ax = plt.subplot(2, 3, i + 4)
        errors = predictions[:, i] - ground_truth[:, i]
        ax.hist(errors, bins=30, alpha=0.8, edgecolor='black', color='steelblue')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label="Zero error")
        
        ax.set_xlabel("Prediction Error", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"{param_names[i]} Errors\nMAE = {mae_per_param[i]:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
    
    fig.suptitle("Model Evaluation: Predictions vs Ground Truth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    # Save figure
    save_path = save_dir / "model_evaluation.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved evaluation plot to: %s", save_path)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "mae": mae_per_param,
        "rmse": rmse_per_param,
        "r2": r2_per_param,
    }

def visualize_dataset_samples(
    dataset, 
    n_samples: int = 4, 
    save_path = None, 
    show: bool = True
):
    """
    Wizualizuje wybrane próbki z PendSimDataset.
    
    Args:
        dataset: Instancja PendSimDataset.
        n_samples: Liczba epizodów do narysowania (domyślnie 4).
        save_path: Ścieżka do zapisu pliku (opcjonalna).
        show: Czy wyświetlić wykres za pomocą plt.show() (domyślnie True).
    """
    if len(dataset) == 0:
        logging.warning("Dataset jest pusty. Przerywam wizualizację.")
        return

    n_samples = min(int(n_samples), len(dataset))
    fig, axes = plt.subplots(
        n_samples, 4, 
        figsize=(16, 3.5 * n_samples), 
        sharex="col", 
        squeeze=False
    )

    for row in range(n_samples):
        # Odbieramy 3 wartości z nowego formatu datasetu
        seq, target, valid_len = dataset[row]
        
        # Konwertujemy do numpy i odcinamy zera z paddingu
        seq_np = seq.numpy()[:valid_len] 
        target_np = target.numpy()
        t = np.arange(int(valid_len))

        # 1. Kolumna: Trig
        axes[row, 0].plot(t, seq_np[:, 0], label="theta")
        axes[row, 0].set_ylabel("Pend pos")
        
        # 2. Kolumna: theta_dot
        axes[row, 1].plot(t, seq_np[:, 1], label="theta_dot")
        axes[row, 1].set_ylabel("theta_dot")
        
        # 3. Kolumna: phi
        axes[row, 2].plot(t, seq_np[:, 2], label="phi")
        axes[row, 2].set_ylabel("phi")
        
        # 4. Kolumna: Control (prev_u)
        axes[row, 3].plot(t, seq_np[:, 3], label="prev_u", color="tab:orange")
        axes[row, 3].set_ylabel("Control")

        # Ustawienia estetyczne
        for col in range(4):
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend(fontsize=8, loc="upper right")
            
        axes[row, 0].set_title(
            f"Epizod {row} (len={valid_len}) | Cel: [{target_np[0]:.3f}, {target_np[1]:.3f}, {target_np[2]:.3f}]",
            fontsize=10
        )

    # Etykiety osi X tylko na samym dole
    for col in range(4):
        axes[-1, col].set_xlabel("Krok symulacji")

    fig.tight_layout()
    
    # Obsługa zapisu
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Zapisano wizualizację do: {save_path}")

    # Obsługa wyświetlania
    if show:
        plt.show()
    else:
        plt.close(fig)