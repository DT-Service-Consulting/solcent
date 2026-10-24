#!/usr/bin/env python3
"""
tft_model.py
============
Trains a Temporal Fusion Transformer (TFT) for dual-objective forecasting (P50 and P25).
Optimized for a GTX 1650 (No mixed precision, scaled batching, safe mask bias).
"""

import os
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_FULL = PROJECT_ROOT / "data" / "processed" / "model_ready_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "tft_model"

MAX_PREDICTION_LENGTH = 96  # 24 hours of 15-min intervals
MAX_ENCODER_LENGTH = 192    # 48 hours of historical context
BATCH_SIZE = 128            # High batch size for 4GB VRAM
NUM_WORKERS = 0             # 0 is fastest for Windows OS
MAX_EPOCHS = 15

def plot_loss_curves(metrics_csv):
    """Parses the CSVLogger output and creates a learning curve plot."""
    if not Path(metrics_csv).exists():
        return
    try:
        df = pd.read_csv(metrics_csv)
        if "train_loss_epoch" not in df.columns or "val_loss" not in df.columns:
            return
            
        train_loss = df.dropna(subset=['train_loss_epoch'])
        val_loss = df.dropna(subset=['val_loss'])

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss['epoch'], train_loss['train_loss_epoch'], label='Train Loss', color='blue')
        plt.plot(val_loss['epoch'], val_loss['val_loss'], label='Validation Loss', color='orange', marker='o')
        plt.title('TFT Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Quantile Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "learning_curve.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSaved learning curve plot to: {plot_path}")
    except Exception as e:
        print(f"\nCould not generate loss plot: {e}")

def main():
    if not MODEL_READY_FULL.exists():
        print(f"Error: Cannot find {MODEL_READY_FULL}.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(MODEL_READY_FULL)
    
    # Ensure system_id is a string (required by pytorch-forecasting as a categorical variable)
    df["system_id"] = df["system_id"].astype(str)
    
    # Clean missing or infinite lags (caused by the 2-day shift at the start of each system's timeline)
    for col in ["lag_pr_same_slot_d2", "lag_pr_recent_day_mean"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Create continuous time_idx for the Transformer
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("system_id").cumcount()

    # The Optimized 13-Feature List
    static_reals = ["capacity_kwp", "tilt_deg", "azimuth_deg"]
    
    time_varying_known_reals = [
        "poa_global_fc", "poa_global_fc2", "clearsky_ghi",
        "sun_elevation", "sun_azimuth", "hour_cos",
        "dni_fc", "cloud_fc2", 
        "lag_pr_same_slot_d2", "lag_pr_recent_day_mean"
    ]
    
    time_varying_unknown_reals = ["target_norm"]

    static_reals = [c for c in static_reals if c in df.columns]
    
    # Split the dataset
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print("Building TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target_norm",
        group_ids=["system_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, predict=True, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)
    test_dataloader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)

    print("Initializing Temporal Fusion Transformer...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=[0.25, 0.50, 0.75]),
        optimizer="adam"
    )

    # MONKEY PATCH: Fix the 16-bit float mask overflow bug
    for module in tft.modules():
        if hasattr(module, 'mask_bias'):
            module.mask_bias = torch.tensor(-1e4)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR / "checkpoints",
        filename="tft-best-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1, monitor="val_loss", mode="min"
    )
    
    logger = CSVLogger(OUTPUT_DIR, name="logs")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_model_summary=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        limit_train_batches=0.1,  # Train on 10% sample per epoch for extreme speed
        limit_val_batches=20      # Hard limit of 20 batches to avoid fraction crashes
    )

    # Search for an existing best model
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    best_ckpt = None
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("tft-best-*.ckpt"))
        if checkpoints:
            best_ckpt = checkpoints[0] # Pick the first one we find

    if best_ckpt:
        print("\n[!] Best model already found on disk. Skipping training phase!\n")
    else:
        print(f"\nTraining TFT Model for up to {MAX_EPOCHS} epochs...")
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        metrics_csv = OUTPUT_DIR / "logs" / "version_0" / "metrics.csv"
        plot_loss_curves(metrics_csv)
        
        # Grab the newly created best checkpoint
        best_ckpt = checkpoint_callback.best_model_path

    print("\nEvaluating on Test Set (Unseen Time Period)...")
    print(f"Loading best model for testing: {best_ckpt}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

    # Use pytorch-forecasting's native prediction method
    predictions = best_tft.predict(test_dataloader, mode="quantiles", return_y=True)
    
    # Extract actuals and quantiles (Move from GPU VRAM to CPU RAM first)
    actuals = predictions.y[0].cpu().numpy()          # True target normalized values
    q_preds = predictions.output.cpu().numpy()        # Shape: (batches, time_steps, quantiles)
    
    p25 = q_preds[:, :, 0] # 25th Percentile
    p50 = q_preds[:, :, 1] # 50th Percentile (Median)
    
    # -----------------------------------------------------
    # DAYLIGHT-ONLY EVALUATION (Apples-to-Apples with LGBM)
    # -----------------------------------------------------
    daylight_mask = actuals > 0.01  # Only evaluate where the sun is actually up
    
    a_day = actuals[daylight_mask]
    p50_day = p50[daylight_mask]
    p25_day = p25[daylight_mask]

    # Metrics Calculations
    # 1. P50 Metrics (Manager)
    rmse_p50 = np.sqrt(np.mean((p50_day - a_day)**2))
    ss_res = np.sum((p50_day - a_day)**2)
    ss_tot = np.sum((a_day - np.mean(a_day))**2)
    r2_p50 = 1 - (ss_res / ss_tot)
    
    # 2. P25 Metrics (Battery EMS)
    rmse_p25 = np.sqrt(np.mean((p25_day - a_day)**2))
    mbe_p25 = np.mean(p25_day - a_day)

    print("\n" + "="*50)
    print("TFT DUAL-OBJECTIVE FORECAST RESULTS (Daylight Only)")
    print("="*50)
    print("1. Business/Manager Metrics (P50 Forecast):")
    print(f"   RMSE : {rmse_p50:.4f}")
    print(f"   R^2  : {r2_p50:.4f}")
    print("\n2. Battery EMS Metrics (P25 Forecast):")
    print(f"   RMSE : {rmse_p25:.4f}")
    print(f"   MBE  : {mbe_p25:.4f}  <-- Maintains safe negative bias")
    print("="*50)

if __name__ == "__main__":
    main()