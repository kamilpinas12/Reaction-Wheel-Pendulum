import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from visualizations import *

from reac_wheel_sim.pend_sim_dataset import PendSimDataset
from reac_wheel_sim.signal_generator import SquareSignal, TrapezoidSignal

from nets.parameter_estimator import PhysicalParameterEstimator
from config import LOGS_DIR, MODELS_DIR


class ParameterNormalizer:
    def __init__(self, min_vals, max_vals):
        self.min_vals = np.asarray(min_vals, dtype=np.float32)
        self.max_vals = np.asarray(max_vals, dtype=np.float32)
        self.ranges = self.max_vals - self.min_vals

    def normalize(self, params):
        return (params - self.min_vals) / self.ranges


def setup_training_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logger = logging.getLogger("train_param_estimator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def normalize_dataset_targets(dataset):
    targets = torch.stack(dataset.targets).numpy()
    normalizer = ParameterNormalizer(targets.min(axis=0), targets.max(axis=0))
    dataset.targets = [
        torch.from_numpy(normalizer.normalize(t.numpy())).float()
        for t in dataset.targets
    ]
    return normalizer


def train_parameter_estimator(
    model,
    train_loader,
    val_loader,
    optimizer,
    logger,
    criterion,
    epochs=50,
    checkpoint_dir=None,
    device="cpu",
):
    if checkpoint_dir is None:
        checkpoint_dir = LOGS_DIR / "train_param_estimator" / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Default TensorBoard directory: runs/
    writer = SummaryWriter(
        comment="-param-estimator-GRU"
    )
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float("inf"), 0

    logger.info("Training started")
    logger.info("Device: %s", device)
    logger.info("Epochs: %d", epochs)
    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info("TensorBoard directory: runs/")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss, best_epoch = avg_val_loss, epoch
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )
            logger.info(
                "New best model at epoch %d | val_loss=%.6f | saved=%s",
                epoch,
                best_val_loss,
                best_model_path,
            )

    writer.close()
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    if best_checkpoint_path.exists():
        saved_model_path = MODELS_DIR / "best_model.pt"
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        torch.save(checkpoint, saved_model_path)
        logger.info("Saved best model to: %s", saved_model_path)

    logger.info(
        "Training finished | best_epoch=%d | best_val_loss=%.6f",
        best_epoch,
        best_val_loss,
    )
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def main():
    run_log_dir = LOGS_DIR / "train_param_estimator"
    logger = setup_training_logger(run_log_dir)

    logger.info("Initializing dataset and model")

    signals = [
        SquareSignal(amplitude_range=(0.2, 1.0)),
        TrapezoidSignal(amplitude_range=(0.2, 1.0)),
    ]

    dataset = PendSimDataset(
        n_episodes=500,
        max_seq_len=64,
        range_pct=10.0,
        noise_levels=[0.01, 0.01, 0.01, 0.01, 0.01],
        signals=signals,
        seed=42,
    )
    train_mode, eval_mode = True, True
    normalize_dataset_targets(dataset)

    logger.info("Dataset size: %d", len(dataset))
    logger.info("train_mode=%s, eval_mode=%s", train_mode, eval_mode)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PhysicalParameterEstimator()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    criterion = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = run_log_dir / "checkpoints"

    if train_mode:
        train_parameter_estimator(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            logger=logger,
            criterion=criterion,
            epochs=50,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

    if eval_mode:
        model_checkpoint_path = checkpoint_dir / "best_model.pt"
        if not model_checkpoint_path.exists():
            logger.error("Checkpoint not found: %s", model_checkpoint_path)
            return

        model = PhysicalParameterEstimator(n_features=5, n_params=3, hidden_dim=32)
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint: %s", model_checkpoint_path)
        evaluate_and_visualize_model(model, val_loader, device=device)
        logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
