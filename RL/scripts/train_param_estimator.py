import os
import shutil
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# --- PYTORCH LIGHTNING IMPORTS ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from utils.visualizations import *
from reac_wheel_sim.pend_sim_dataset import PendSimDataset
from reac_wheel_sim.reaction_wheel_env import ReactionWheelEnv
from reac_wheel_sim.reaction_wheel_wrappers import *
from reac_wheel_sim.signal_generator import SquareSignal, TrapezoidSignal

from nets.parameter_estimator import *
from utils.custom_paths import LOGS_DIR, MODELS_DIR, DATASETS_DIR
from utils.common import ParameterNormalizer, setup_file_logger


def setup_training_logger(log_dir: Path) -> logging.Logger:
    return setup_file_logger("train_param_estimator", log_dir / "train.log")


def normalize_dataset_targets(dataset, min_vals, max_vals):
    normalizer = ParameterNormalizer(min_vals, max_vals)
    dataset.targets = [
        torch.from_numpy(normalizer.normalize(t.numpy())).float()
        for t in dataset.targets
    ]
    return normalizer


# 1. TWORZENIE LIGHTNING MODULE
class LitParamEstimator(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, hparams_dict: dict = None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()
        
        if hparams_dict:
            self.save_hyperparameters(hparams_dict, ignore=['model'])

    def forward(self, x, lengths):
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch 
        y_hat = self(x, lengths)
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_hat = self(x, lengths)
        loss = self.criterion(y_hat, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main():
    run_log_dir = LOGS_DIR / "train_param_estimator"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_file_logger("train_param_estimator", LOGS_DIR / "train_param_estimator" / "train.log")
    logger.info("Initializing dataset and model")

    signals = [
        SquareSignal(amplitude_range=(0.2, 1.0)),
        TrapezoidSignal(amplitude_range=(0.2, 1.0)),
    ]
    
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

    param_noise_pct = 0.10
    mass_pos_pct = 1.00
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

    dataset = PendSimDataset(
        n_episodes=hparams["n_episodes"],
        min_seq_len=hparams["min_seq_len"],
        max_seq_len=hparams["max_seq_len"],
        env=env,
        signals=signals,
        seed=42,
        cache_path=DATASETS_DIR / "pend_sim_dataset",
        load_datast=False
    )
    
    train_mode, eval_mode = True, True
    target_normalizer = normalize_dataset_targets(
        dataset,
        min_vals=target_min_vals,
        max_vals=target_max_vals,
    )

    logger.info("Dataset size: %d", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=4)

    base_model = ParamEstimatorLSTM(
        n_features=5, n_params=3, hidden_dim=hparams["hidden_dim"], fc_dim=hparams["fc_dim"]
    )
    
    lit_model = LitParamEstimator(model=base_model, lr=hparams["lr"], hparams_dict=hparams)

    # 2. KONFIGURACJA CALLBACKÓW I LOGGERÓW
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_log_dir / "checkpoints",
        filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    
    tb_logger = TensorBoardLogger(save_dir="runs/", name="ParamEstimator")

    # 3. TWORZENIE TRAINERA
    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        logger=tb_logger,
        callbacks=[checkpoint_callback, RichProgressBar()],
        accelerator="auto", # Automatycznie wybiera cuda/mps/cpu
        devices="auto",
        log_every_n_steps=10,
    )

    if train_mode:
        logger.info(f"Hyperparams: {hparams}")
        logger.info("Starting PyTorch Lightning training...")
        
        # Ta jedna linijka odpala całą pętlę!
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f"Training finished. Best model saved at: {best_model_path}")
        
        # Kopiujemy najlepszy model do głównego folderu
        if best_model_path:
            saved_model_path = MODELS_DIR / "best_model.pt"
            shutil.copy(best_model_path, saved_model_path)
            logger.info(f"Copied best model to: {saved_model_path}")

    if eval_mode:
        logger.info("Evaluation started...")
        
        model_path = MODELS_DIR / "best_model.pt"
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return

        # Wczytywanie z checkpointu (Lightning sam wczytuje wagi)
        lit_model = LitParamEstimator.load_from_checkpoint(
            checkpoint_path=str(model_path), 
            model=base_model
        )
        
        lit_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit_model.to(device)
        
        evaluate_and_visualize_model(
            lit_model.model, # Wyciągamy czysty model PyTorch (bez Lit-otoczki) dla Twojej funkcji wizualizacji
            val_loader,
            normalizer=target_normalizer,
            save_dir=run_log_dir,
            device=str(device),
        )
        logger.info("Evaluation finished")

if __name__ == "__main__":
    main()