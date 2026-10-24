#!/usr/bin/env python3
"""
model_lstm.py
=============
A SEQUENCE neural network (bidirectional LSTM) as a fair comparison to the LightGBM
model. Goal: check whether modelling the whole day as a sequence captures temporal
structure that per-slot gradient boosting misses. The honest expectation is to
*match* LightGBM, not necessarily beat it -- on tabular problems trees are hard to top.

Why bidirectional is legitimate here: at the noon cutoff you already have the ENTIRE
next-day weather forecast, so the model may look both forward and backward across the
day's 96 slots. It is not forecasting in real time within the day.

Design (deliberately small -- small data rewards small models):
  * Input : each day = a sequence of 96 fifteen-minute slots x features (same features
            as LightGBM). Night slots are included but masked out of the loss.
  * Model : BiLSTM over the day -> per-slot dense head -> 96 outputs.
  * Loss  : masked MSE on daylight slots only.
  * Eval  : daylight slots only, same metrics + skill score as model_baselines.py, so
            the numbers are directly comparable.

Input : pv_features/features_all_systems.csv  (full 96-slot grid, not model_ready)
Output: pv_model/lstm_metrics.csv, pv_model/lstm_predictions_test.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_baselines import metrics, skill, FEATURES, CLIP

# ----------------------------------------------------------------------------
FEATURES_FILE = Path("data/interim/features_all_systems.csv")
OUTPUT_DIR    = Path("pv_model")
EXCLUDE       = [12293]
TARGET        = "target_norm"
EPOCHS        = 60
BATCH         = 64
HIDDEN        = 64
LR            = 1e-3
PATIENCE      = 10
MAX_SYSTEMS   = None       # set an int to subsample systems for a quick trial
SEED          = 32
# ----------------------------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)


class DayLSTM(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.lstm(x)             # (B, 96, 2*hidden)
        return self.head(h).squeeze(-1)  # (B, 96)


def build_sequences(df, feats):
    """Reshape rows into fixed 96-slot day sequences per (system, date)."""
    df = df.sort_values(["system_id", "timestamp"])
    X, Y, M, keys = [], [], [], []
    dropped = 0
    for (sid, d), g in df.groupby(["system_id", df["timestamp"].dt.normalize()]):
        if len(g) != 96:                # skip irregular days (e.g. DST transitions)
            dropped += 1; continue
        X.append(g[feats].values.astype("float32"))
        Y.append(np.nan_to_num(g[TARGET].values.astype("float32"), nan=0.0))
        M.append((g["is_daylight"] & g["is_valid"]).values.astype("float32"))
        keys.append((sid, pd.Timestamp(d)))
    print(f"built {len(X)} day-sequences ({dropped} irregular days skipped)")
    return np.array(X), np.array(Y), np.array(M), keys


def masked_rmse(pred, y, m):
    e = ((pred - y) ** 2) * m
    return float(np.sqrt(e.sum() / m.sum()))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Brussels")
    if EXCLUDE:
        df = df[~df.system_id.isin(EXCLUDE)]
    if MAX_SYSTEMS:
        keep = sorted(df.system_id.unique())[:MAX_SYSTEMS]
        df = df[df.system_id.isin(keep)]
    feats = [c for c in FEATURES if c in df.columns]
    # NN needs no NaNs: fill feature gaps (lags on first days) with 0 (neutral)
    df[feats] = df[feats].fillna(0.0)
    print(f"{len(df):,} rows | {df.system_id.nunique()} systems | {len(feats)} features")

    X, Y, M, keys = build_sequences(df, feats)
    key_df = pd.DataFrame(keys, columns=["system_id", "date"])

    # time-based split by DAY date (same 70/15/15 idea as model_ready)
    q70, q85 = key_df["date"].quantile(0.70), key_df["date"].quantile(0.85)
    split = np.where(key_df["date"] <= q70, "train",
                     np.where(key_df["date"] <= q85, "val", "test"))
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"days: train {tr.sum()} | val {va.sum()} | test {te.sum()}")

    # standardise features on TRAIN slots only
    flat = X[tr].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xs = (X - mu) / sd

    dev = "cpu"
    Xt = torch.tensor(Xs); Yt = torch.tensor(Y); Mt = torch.tensor(M)
    model = DayLSTM(len(feats), HIDDEN).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def batches(idx):
        ii = np.where(idx)[0]; np.random.shuffle(ii)
        for s in range(0, len(ii), BATCH):
            yield ii[s:s + BATCH]

    best, best_state, wait = np.inf, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in batches(tr):
            xb, yb, mb = Xt[b], Yt[b], Mt[b]
            opt.zero_grad()
            pred = model(xb)
            loss = (((pred - yb) ** 2) * mb).sum() / mb.sum()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xt[va]).numpy()
        vr = masked_rmse(vp, Y[va], M[va])
        if vr < best - 1e-5:
            best, best_state, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        print(f"  epoch {ep:2d}: val masked RMSE {vr:.4f}{'  *' if wait==0 else ''}")
        if wait >= PATIENCE:
            print(f"  early stop (no improvement for {PATIENCE} epochs)"); break
    if best_state is not None:
        model.load_state_dict(best_state)

    # predict test days
    model.eval()
    with torch.no_grad():
        P = np.clip(model(Xt[te]).numpy(), 0, CLIP)
    yte, mte = Y[te], M[te]
    kte = key_df[te].reset_index(drop=True)

    # flatten test daylight slots into a tidy frame, attach baselines for skill
    cap_map = df.drop_duplicates("system_id").set_index("system_id")["capacity_kwp"].to_dict()
    day_target = {(int(k.system_id), k.date.normalize()): Y[i]
                  for i, k in key_df.iterrows()}
    rows = []
    for i in range(len(kte)):
        sid = int(kte.iloc[i]["system_id"]); dd = kte.iloc[i]["date"]
        prev = day_target.get((sid, (dd - pd.Timedelta(days=1)).normalize()))
        for s in range(96):
            if mte[i, s] < 0.5:
                continue                       # daylight only
            rows.append({"system_id": sid,
                         "actual": yte[i, s], "pred_lstm": P[i, s],
                         "pred_persistence": (prev[s] if prev is not None else np.nan),
                         "cap": cap_map.get(sid, np.nan)})
    ev = pd.DataFrame(rows).dropna(subset=["pred_persistence"])
    print(f"\nEvaluating on {len(ev):,} test daylight slots (persistence-defined).")

    m_lstm = metrics(ev["actual"], ev["pred_lstm"], ev["cap"])
    rmse_p = metrics(ev["actual"], ev["pred_persistence"], ev["cap"])["RMSE"]
    sk = skill(m_lstm["RMSE"], rmse_p)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ LSTM TEST METRICS (daylight, normalised) ================")
    print(pd.Series(m_lstm).to_string())
    print(f"\nSKILL vs persistence : {sk:+.3f}")
    print(f"\nCompare to the LightGBM result from model_baselines.py "
          f"(skill ~+0.40, RMSE ~0.117, R2 ~0.79).")
    print("If the LSTM is close to or below LightGBM, that is the expected outcome for "
          "this tabular problem and a valid finding to report.")

    pd.DataFrame([{**m_lstm, "skill_vs_persist": sk}]).to_csv(
        OUTPUT_DIR / "lstm_metrics.csv", index=False)
    ev.to_csv(OUTPUT_DIR / "lstm_predictions_test.csv", index=False)
    print(f"\nsaved -> {OUTPUT_DIR/'lstm_metrics.csv'}, {OUTPUT_DIR/'lstm_predictions_test.csv'}")


if __name__ == "__main__":
    main()
