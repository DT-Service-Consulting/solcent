#!/usr/bin/env python3
"""
model_benchmark.py
==================
An automated "Optuna Arena" that tunes several machine-learning models on the 
same data and compares them to see which performs best on the day-ahead 
PV forecasting problem after rigorous hyperparameter optimization.

UPDATES:
  - Added Learning Curve tracking. Automatically plots Train vs. Validation RMSE 
    across boosting rounds for XGBoost and LightGBM, saving them to learning_curves/

Models compared & Tuned:
  * Persistence         (baseline: same slot yesterday)
  * Clear-sky proxy     (baseline: forecast POA / 1000)
  * Linear Regression   (simple linear model - NO TUNING)
  * Ridge Regression    (regularised linear - TUNED)
  * Decision Tree       (single tree - TUNED)
  * Random Forest       (bagged trees - TUNED)
  * Gradient Boosting   (sklearn HistGradientBoosting - TUNED)
  * XGBoost             (if installed - TUNED)
  * LightGBM            (if installed - TUNED)

Inputs : data/processed/model_ready.csv
Outputs: models/pv_model/benchmark_metrics.csv, best_params.json, comparison charts,
         and models/pv_model/learning_curves/
"""

import time
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone

# Suppress annoying sklearn/lightgbm warnings during hundreds of trials
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # We will print our own progress

# Import shared baseline variables
from model_baselines import metrics, skill, FEATURES, TARGET, CLIP

# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY  = PROJECT_ROOT / "data" / "processed" / "model_ready.csv"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "pv_model"
CURVES_DIR   = OUTPUT_DIR / "learning_curves"
MOS_DATA     = PROJECT_ROOT / "data" / "interim" / "mos_training_data.csv"
MOS_PARAMS   = PROJECT_ROOT / "models" / "mos_model" / "best_mos_params.json"
EXCLUDE      = [12293]
N_TRIALS     = 20    # Number of Optuna trials PER MODEL.
# ----------------------------------------------------------------------------

def suggest_model(trial, model_name, is_final=False):
    """
    Defines the hyperparameter search space for each model.
    If is_final=True, returns a beefed-up version of the model using the best params.
    """
    if model_name == "Linear Regression":
        return LinearRegression()
        
    elif model_name == "Ridge Regression":
        alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        return Ridge(alpha=alpha)
        
    elif model_name == "Decision Tree":
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 200)
        return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
        
    elif model_name == "Random Forest":
        max_depth = trial.suggest_int("max_depth", 10, 25)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 100)
        n_est = 250 if is_final else 50
        return RandomForestRegressor(n_estimators=n_est, max_depth=max_depth, 
                                     min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=0)
                                     
    elif model_name == "Gradient Boosting":
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 20, 127)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 20, 100)
        l2_regularization = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
        
        return HistGradientBoostingRegressor(
            max_iter=300 if is_final else 150, learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, random_state=0
        )
        
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        max_depth = trial.suggest_int("max_depth", 4, 12)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        
        return XGBRegressor(
            n_estimators=400 if is_final else 150, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=-1, random_state=0, tree_method="hist"
        )
        
    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        num_leaves = trial.suggest_int("num_leaves", 20, 150)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_child_samples = trial.suggest_int("min_child_samples", 20, 100)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        
        return LGBMRegressor(
            n_estimators=500 if is_final else 150, learning_rate=learning_rate,
            num_leaves=num_leaves, max_depth=max_depth, min_child_samples=min_child_samples,
            subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, random_state=0, n_jobs=-1, verbose=-1
        )

def get_model_list():
    """Returns a dictionary of available models and whether they require NaN imputation."""
    models = {
        "Linear Regression": True,
        "Ridge Regression":  True,
        "Decision Tree":     True,
        "Random Forest":     True,
        "Gradient Boosting": False,
    }
    try:
        import xgboost
        models["XGBoost"] = False
    except ImportError:
        print("  (XGBoost not installed - skipping)")
    try:
        import lightgbm
        models["LightGBM"] = False
    except ImportError:
        print("  (LightGBM not installed - skipping)")
        
    return models

def main():
    if not MODEL_READY.exists():
        print(f"Error: {MODEL_READY} not found.")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(MODEL_READY)
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    df = df.sort_values(["system_id", "timestamp"]).reset_index(drop=True)
    
    # ========================================================================
    # NEW: MOS (LIGHTGBM) PRE-PROCESSING PIPELINE
    # ========================================================================
    print("\n--- APPLYING MOS (LightGBM) CORRECTION TO FEATURES ---")
    if MOS_DATA.exists():
        df_mos = pd.read_csv(MOS_DATA)
        df_mos['timestamp'] = pd.to_datetime(df_mos['timestamp'], utc=True).dt.tz_convert("Europe/Brussels")
        
        # Prepare MOS Targets & Features
        df_mos['error_target'] = df_mos['actual_ghi'] - df_mos['ecmwf_ghi']
        df_mos['hour'] = df_mos['timestamp'].dt.hour
        df_mos['month'] = df_mos['timestamp'].dt.month
        df_mos['day_of_year'] = df_mos['timestamp'].dt.dayofyear
        
        mos_features = ['ecmwf_ghi', 'ecmwf_cloud', 'ecmwf_temp', 'hour', 'month', 'day_of_year']
        X_mos = df_mos[mos_features]
        y_mos = df_mos['error_target']
        
        # Default LightGBM params (will be overwritten if best_mos_params.json exists)
        lgb_params = {'n_estimators': 150, 'learning_rate': 0.05, 'random_state': 42, 'n_jobs': -1}
        if MOS_PARAMS.exists():
            with open(MOS_PARAMS, "r") as f:
                p_dict = json.load(f)
                if "LightGBM" in p_dict:
                    lgb_params = p_dict["LightGBM"]
                    
        from lightgbm import LGBMRegressor
        mos_model = LGBMRegressor(**lgb_params)
        mos_model.fit(X_mos, y_mos)
        
        # Calculate Correction Ratio
        df_mos['predicted_error'] = mos_model.predict(X_mos)
        df_mos['corrected_ghi'] = np.clip(df_mos['ecmwf_ghi'] + df_mos['predicted_error'], 0, None)
        df_mos['mos_ratio'] = df_mos['corrected_ghi'] / (df_mos['ecmwf_ghi'] + 1e-6)
        
        # Merge Ratio into the main DataFrame
        mos_bridge = df_mos[['system_id', 'timestamp', 'mos_ratio']].copy()
        df = pd.merge(df, mos_bridge, on=['system_id', 'timestamp'], how='left')
        df['mos_ratio'] = df['mos_ratio'].fillna(1.0)
        
        # Apply the correction to our physical irradiance features
        if 'poa_global_fc' in df.columns:
            df['poa_global_fc_mos'] = df['poa_global_fc'] * df['mos_ratio']
        if 'poa_global_fc2' in df.columns:
            df['poa_global_fc2_mos'] = df['poa_global_fc2'] * df['mos_ratio']
            
        print(f"-> Successfully trained LightGBM MOS and corrected PV features.")
    else:
        print("-> WARNING: mos_training_data.csv not found. Proceeding with uncorrected raw features.")
    # ========================================================================
    
    # Map the corrected features into the active feature list
    raw_feats = [c for c in FEATURES if c in df.columns]
    feats = []
    for f in raw_feats:
        if f == "poa_global_fc" and "poa_global_fc_mos" in df.columns:
            feats.append("poa_global_fc_mos")
        elif f == "poa_global_fc2" and "poa_global_fc2_mos" in df.columns:
            feats.append("poa_global_fc2_mos")
        else:
            feats.append(f)
    
    print(f"\n==================================================================")
    print(f"OPTUNA BENCHMARK ARENA | {len(df):,} rows | {df.system_id.nunique()} systems | {len(feats)} features")
    print(f"==================================================================\n")

    # Split for Optuna: Train (Fit), Val (Score/Tune), Test (Final Evaluation)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]
    
    # We combine Train + Val for the FINAL fit to maximize data exposure
    tr_va = pd.concat([tr, va])

    # Extract DataFrames
    Xtr, ytr = tr[feats], tr[TARGET]
    Xva, yva = va[feats], va[TARGET]
    Xtr_va, ytr_va = tr_va[feats], tr_va[TARGET]
    Xte, yte, cte = te[feats], te[TARGET], te["capacity_kwp"]
    
    # Create NaN-imputed copies for strict models (Linear, RF, etc.)
    med = Xtr.median()
    Xtr_i, Xva_i = Xtr.fillna(med), Xva.fillna(med)
    Xtr_va_i, Xte_i = Xtr_va.fillna(med), Xte.fillna(med)

    # Baselines on the test set
    prev = df[["system_id", "timestamp", TARGET]].copy()
    prev["timestamp"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={TARGET: "persist"})
    te = te.merge(prev, on=["system_id", "timestamp"], how="left")
    
    mask = te["persist"].notna()               
    yte_m, cte_m = te.loc[mask, TARGET], te.loc[mask, "capacity_kwp"]
    persist = te.loc[mask, "persist"]
    
    csrc = "poa_global_fc" if "poa_global_fc" in te.columns else "clearsky_ghi"
    clearsky = (te.loc[mask, csrc] / 1000.0).clip(0, CLIP)
    rmse_persist = metrics(yte_m, persist, cte_m)["RMSE"]

    results = {}
    best_params = {}
    
    def record(name, pred, secs=np.nan):
        m = metrics(yte_m, np.clip(pred, 0, CLIP), cte_m)
        m["skill"] = skill(m["RMSE"], rmse_persist)
        m["train_s"] = secs
        
        # Calculate nRMSE: RMSE normalized by the mean of the actual generation
        m["nRMSE"] = m["RMSE"] / yte_m.mean() if yte_m.mean() > 0 else np.nan
        
        results[name] = m
        print(f"  [FINAL TEST] {name:18s} RMSE: {m['RMSE']:.4f} | nRMSE: {m['nRMSE']:.4f} | R2: {m['R2']:.3f} | Skill: {m['skill']:+.3f}")

    print("--- BASELINES ---")
    record("Persistence", persist.values, 0.0)
    record("Clear-sky proxy", clearsky.values, 0.0)
    print("\n--- BEGIN TUNING ARENA ---")

    for name, needs_impute in get_model_list().items():
        t0 = time.time()
        
        # Select appropriate dataset version (imputed vs native NaNs)
        X_train_opt = Xtr_i if needs_impute else Xtr
        X_val_opt = Xva_i if needs_impute else Xva
        
        X_train_final = Xtr_va_i if needs_impute else Xtr_va
        X_test_final = Xte_i.loc[mask.values] if needs_impute else Xte.loc[mask.values]

        if name == "Linear Regression":
            print(f"\nEvaluating {name} (No tuning required)...")
            final_model = LinearRegression()
            best_params[name] = {}
            
        else:
            print(f"\n[{name}] Running {N_TRIALS} Optuna Trials...")
            
            def objective(trial):
                model = suggest_model(trial, name, is_final=False)
                model.fit(X_train_opt, ytr)
                preds = np.clip(model.predict(X_val_opt), 0, CLIP)
                return np.sqrt(mean_squared_error(yva, preds))
                
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            
            print(f"  [Tuned] Best Val RMSE: {study.best_value:.4f}")
            final_model = suggest_model(optuna.trial.FixedTrial(study.best_params), name, is_final=True)
            
            saved_params = study.best_params.copy()
            if hasattr(final_model, 'n_estimators'):
                saved_params['n_estimators'] = final_model.n_estimators
            elif hasattr(final_model, 'max_iter'):
                saved_params['max_iter'] = final_model.max_iter
            best_params[name] = saved_params

        # -------------------------------------------------------------
        # GENERATE LEARNING CURVES (For Gradient Boosting frameworks)
        # -------------------------------------------------------------
        if name in ["XGBoost", "LightGBM"]:
            print(f"  -> Generating Learning Curve for {name}...")
            # Use a clone to track Train vs Val without polluting the final fit
            curve_model = clone(final_model)
            
            try:
                if name == "XGBoost":
                    curve_model.fit(X_train_opt, ytr, eval_set=[(X_train_opt, ytr), (X_val_opt, yva)], verbose=False)
                    evals = curve_model.evals_result()
                    t_key, v_key = list(evals.keys())[0], list(evals.keys())[1]
                    m_key = list(evals[t_key].keys())[0]
                    train_metric = evals[t_key][m_key]
                    val_metric = evals[v_key][m_key]
                    
                elif name == "LightGBM":
                    curve_model.fit(X_train_opt, ytr, eval_set=[(X_train_opt, ytr), (X_val_opt, yva)])
                    evals = curve_model.evals_result_
                    t_key, v_key = list(evals.keys())[0], list(evals.keys())[1]
                    m_key = list(evals[t_key].keys())[0]
                    
                    # LightGBM might return L2 (MSE) natively instead of RMSE depending on version
                    train_metric = np.sqrt(evals[t_key][m_key]) if m_key == 'l2' else evals[t_key][m_key]
                    val_metric = np.sqrt(evals[v_key][m_key]) if m_key == 'l2' else evals[v_key][m_key]
                
                # Plot and save
                plt.figure(figsize=(9, 6))
                plt.plot(train_metric, label="Train Error", color="#1f77b4", linewidth=2)
                plt.plot(val_metric, label="Validation Error", color="#ff7f0e", linewidth=2)
                plt.title(f"{name} Learning Curve (Best Parameters)")
                plt.xlabel("Boosting Round (Trees)")
                plt.ylabel("RMSE")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(CURVES_DIR / f"{name.lower().replace(' ', '_')}_learning_curve.png", dpi=120)
                plt.close()
                print(f"  -> Curve saved to learning_curves/")
            except Exception as e:
                print(f"  -> [Warning] Failed to plot learning curve: {e}")

        # Fit final model on combined Train+Val data for maximum accuracy
        final_model.fit(X_train_final, ytr_va)
        
        # Predict on Test Data
        pred = final_model.predict(X_test_final)
        record(name, pred, time.time() - t0)

    res = pd.DataFrame(results).T
    res.index.name = "model"
    res.to_csv(OUTPUT_DIR / "benchmark_metrics.csv")
    
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ FINAL OPTIMIZED BENCHMARK (Test Set, Daylight) ================")
    print(res[["MAE", "RMSE", "nRMSE", "MBE", "R2", "skill", "train_s"]].to_string())
    
    best = res.drop(["Persistence", "Clear-sky proxy"], errors="ignore")["RMSE"].idxmin()
    print(f"\n🏆 CHAMPION MODEL BY RMSE: {best.upper()} 🏆\n")

    # ---------- Charts ----------
    order = res["RMSE"].sort_values(ascending=False).index
    colors = ["#c0392b" if n in ("Persistence", "Clear-sky proxy") else "#1f6fb5" for n in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(order, res.loc[order, "RMSE"], color=colors)
    for i, n in enumerate(order):
        ax.text(res.loc[n, "RMSE"], i, f" {res.loc[n,'RMSE']:.4f}", va="center", fontsize=9)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title(f"Optimized Model Comparison - RMSE (Test Set, Daylight)")
    ax.text(0.98, 0.02, "red = baselines", transform=ax.transAxes, ha="right", color="#c0392b", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_rmse.png", dpi=100)
    plt.close()

    trained = res.drop(["Persistence", "Clear-sky proxy"], errors="ignore").sort_values("skill")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(trained))
    h = 0.4
    ax.barh(y + h/2, trained["skill"], h, label="Skill vs Persistence", color="#1f6fb5")
    ax.barh(y - h/2, trained["R2"], h, label="R\u00b2", color="#e08a1e")
    ax.set_yticks(y)
    ax.set_yticklabels(trained.index)
    ax.set_xlabel("Score (higher is better)")
    ax.set_title("Optimized Model Comparison - Skill & R\u00b2")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_skill_r2.png", dpi=100)
    plt.close()

    metr = ["MAE", "RMSE", "nRMSE", "R2", "skill"]
    tm = res.drop(["Persistence", "Clear-sky proxy"], errors="ignore")[metr].copy()
    norm = tm.copy()
    for c in metr:                              
        v = tm[c]
        norm[c] = (v - v.min()) / (v.max() - v.min() + 1e-9)
        if c in ("MAE", "RMSE", "nRMSE"):
            norm[c] = 1 - norm[c] 
            
    fig, ax = plt.subplots(figsize=(7, 0.6*len(tm)+2))
    im = ax.imshow(norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metr)))
    ax.set_xticklabels(metr)
    ax.set_yticks(range(len(tm)))
    ax.set_yticklabels(tm.index)
    
    for i in range(len(tm)):
        for j, c in enumerate(metr):
            ax.text(j, i, f"{tm.iloc[i][c]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if norm.iloc[i, j] > 0.85 or norm.iloc[i, j] < 0.15 else "black")
            
    ax.set_title("Optimized Model Heatmap (Green = Better)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative (1=Best)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_heatmap.png", dpi=100)
    plt.close()

    print(f"Artifacts saved -> {OUTPUT_DIR.name}/benchmark_metrics.csv, best_params.json and charts.")

if __name__ == "__main__":
    main()