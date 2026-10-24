#!/usr/bin/env python3
"""
tune_lstm_attention.py
======================
Upgrades the baseline Bi-LSTM by injecting a Multi-Head Self-Attention mechanism,
and wraps the entire architecture in an Optuna hyperparameter optimization loop.

Architecture: Input -> Bi-LSTM -> Multi-Head Self-Attention -> Dense Head -> Output
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_baselines import metrics, skill, FEATURES, CLIP

# Suppress warnings for cleaner Optuna output
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
FEATURES_FILE = Path("data/interim/features_all_systems.csv")
OUTPUT_DIR    = Path("models/pv_model")
EXCLUDE       = [12293]
TARGET        = "target_norm"
EPOCHS        = 40         # Reduced slightly for faster Optuna trials
BATCH         = 64
PATIENCE      = 7
N_TRIALS      = 20         # Number of hyperparameter combinations to test
SEED          = 42
# ----------------------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

class AttentionBiLSTM(nn.Module):
    """
    State-of-the-art hybrid architecture. The Bi-LSTM captures the chronological 
    weather sequence, and the Attention layer determines which specific times of day 
    are most critical for predicting the current slot's generation.
    """
    def __init__(self, n_features, hidden=64, num_heads=4, dropout=0.1):
        super().__init__()
        # 1. Sequential Feature Extraction
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        
        # 2. Self-Attention (embed_dim must be hidden * 2 because of bidirectional)
        self.attn = nn.MultiheadAttention(embed_dim=hidden * 2, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        
        # 3. Non-linear Output Head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h, _ = self.lstm(x)                  # Output: (Batch, 96 slots, 2*hidden)
        
        # Self-Attention (Query, Key, Value are all the LSTM output)
        attn_out, _ = self.attn(h, h, h)     
        
        # Residual Connection (helps gradients flow through deep networks)
        h = h + attn_out                     
        
        return self.head(h).squeeze(-1)      # Output: (Batch, 96)

def build_sequences(df, feats):
    """Reshape rows into fixed 96-slot day sequences per (system, date)."""
    df = df.sort_values(["system_id", "timestamp"])
    X, Y, M, keys = [], [], [], []
    for (sid, d), g in df.groupby(["system_id", df["timestamp"].dt.normalize()]):
        if len(g) != 96:
            continue
        X.append(g[feats].values.astype("float32"))
        Y.append(np.nan_to_num(g[TARGET].values.astype("float32"), nan=0.0))
        M.append((g["is_daylight"] & g["is_valid"]).values.astype("float32"))
        keys.append((sid, pd.Timestamp(d)))
    return np.array(X), np.array(Y), np.array(M), keys

def masked_rmse(pred, y, m):
    e = ((pred - y) ** 2) * m
    return float(np.sqrt(e.sum() / m.sum()))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FEATURES_FILE.exists():
        print(f"Error: {FEATURES_FILE} not found.")
        return
        
    print("Loading data and building 24-hour sequences...")
    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
        
    feats = [c for c in FEATURES if c in df.columns]
    df[feats] = df[feats].fillna(0.0)

    X, Y, M, keys = build_sequences(df, feats)
    key_df = pd.DataFrame(keys, columns=["system_id", "date"])

    q70, q85 = key_df["date"].quantile(0.70), key_df["date"].quantile(0.85)
    split = np.where(key_df["date"] <= q70, "train",
                     np.where(key_df["date"] <= q85, "val", "test"))
    
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    flat = X[tr].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xs = (X - mu) / sd

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")
    
    Xt = torch.tensor(Xs).to(dev)
    Yt = torch.tensor(Y).to(dev)
    Mt = torch.tensor(M).to(dev)
    
    X_val_cpu = Xs[va] # Keep a CPU copy for fast numpy evaluation
    
    def batches(idx):
        ii = np.where(idx)[0]; np.random.shuffle(ii)
        for s in range(0, len(ii), BATCH):
            yield ii[s:s + BATCH]

    print(f"\n--- Starting Optuna Tuning ({N_TRIALS} Trials) ---")
    
    def objective(trial):
        # Hyperparameter Search Space
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        model = AttentionBiLSTM(len(feats), hidden=hidden_size, num_heads=num_heads, dropout=dropout).to(dev)
        
        # AdamW is generally superior to Adam when using weight_decay
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning Rate Scheduler (Reduces LR when validation loss plateaus)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

        best_val = np.inf
        wait = 0
        
        for ep in range(1, EPOCHS + 1):
            model.train()
            for b in batches(tr):
                xb, yb, mb = Xt[b], Yt[b], Mt[b]
                opt.zero_grad()
                pred = model(xb)
                loss = (((pred - yb) ** 2) * mb).sum() / mb.sum()
                loss.backward()
                opt.step()
                
            model.eval()
            with torch.no_grad():
                vp = model(Xt[va]).cpu().numpy()
            
            vr = masked_rmse(vp, Y[va], M[va])
            scheduler.step(vr) # Step the scheduler based on val metric
            
            if vr < best_val - 1e-5:
                best_val = vr
                wait = 0
            else:
                wait += 1
                
            if wait >= PATIENCE:
                break
                
        return best_val

    # Run Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print("\n--- Optuna Tuning Complete ---")
    print(f"Best Val RMSE: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # ==========================================
    # Train Final Model with Best Parameters
    # ==========================================
    print("\nTraining Final Champion Model...")
    bp = study.best_params
    
    final_model = AttentionBiLSTM(len(feats), hidden=bp["hidden_size"], 
                                  num_heads=bp["num_heads"], dropout=bp["dropout"]).to(dev)
                                  
    opt = torch.optim.AdamW(final_model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    
    best_final, best_state, wait = np.inf, None, 0
    train_history = []
    val_history = []
    
    for ep in range(1, EPOCHS + 20): # Give it extra epochs for the final run
        final_model.train()
        epoch_loss = 0.0
        n_batches = 0
        for b in batches(tr):
            xb, yb, mb = Xt[b], Yt[b], Mt[b]
            opt.zero_grad()
            pred = final_model(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum()
            loss.backward(); opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        final_model.eval()
        with torch.no_grad():
            vp = final_model(Xt[va]).cpu().numpy()
            
        vr = masked_rmse(vp, Y[va], M[va])
        tr_rmse = float(np.sqrt(epoch_loss / n_batches))
        
        train_history.append(tr_rmse)
        val_history.append(vr)
        
        scheduler.step(vr)
        
        if vr < best_final - 1e-5:
            best_final, best_state, wait = vr, {k: v.clone() for k, v in final_model.state_dict().items()}, 0
        else:
            wait += 1
            
        print(f"  epoch {ep:2d}: val masked RMSE {vr:.4f}{'  *' if wait==0 else ''}")
        if wait >= PATIENCE:
            print(f"  early stop (no improvement for {PATIENCE} epochs)")
            break
            
    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ==========================================
    # Generate Learning Curve Plot
    # ==========================================
    print("\nGenerating Learning Curve Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_history) + 1), train_history, label="Train RMSE", color="#1f77b4", linewidth=2)
    plt.plot(range(1, len(val_history) + 1), val_history, label="Validation RMSE", color="#ff7f0e", linewidth=2)
    plt.title("Attention Bi-LSTM Final Model Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = OUTPUT_DIR / "attention_lstm_learning_curve.png"
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"Saved learning curve to: {curve_path}")

    # ==========================================
    # Evaluate on Unseen Test Set
    # ==========================================
    final_model.eval()
    with torch.no_grad():
        P = np.clip(final_model(Xt[te]).cpu().numpy(), 0, CLIP)
        
    yte, mte = Y[te], M[te]
    kte = key_df[te].reset_index(drop=True)

    cap_map = df.drop_duplicates("system_id").set_index("system_id")["capacity_kwp"].to_dict()
    day_target = {(int(k.system_id), k.date.normalize()): Y[i] for i, k in key_df.iterrows()}
    
    rows = []
    for i in range(len(kte)):
        sid = int(kte.iloc[i]["system_id"])
        dd = kte.iloc[i]["date"]
        prev = day_target.get((sid, (dd - pd.Timedelta(days=1)).normalize()))
        for s in range(96):
            if mte[i, s] < 0.5:
                continue
            rows.append({
                "system_id": sid,
                "actual": yte[i, s], 
                "pred_attn_lstm": P[i, s],
                "pred_persistence": (prev[s] if prev is not None else np.nan),
                "cap": cap_map.get(sid, np.nan)
            })
            
    ev = pd.DataFrame(rows).dropna(subset=["pred_persistence"])
    m_lstm = metrics(ev["actual"], ev["pred_attn_lstm"], ev["cap"])
    rmse_p = metrics(ev["actual"], ev["pred_persistence"], ev["cap"])["RMSE"]
    sk = skill(m_lstm["RMSE"], rmse_p)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ ATTENTION Bi-LSTM TEST METRICS ================")
    print(pd.Series(m_lstm).to_string())
    print(f"\nSKILL vs persistence : {sk:+.3f}")
    
    # Save the PyTorch Model Weights
    model_path = OUTPUT_DIR / "best_attention_lstm.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved optimized PyTorch model to: {model_path}")

if __name__ == "__main__":
    main()