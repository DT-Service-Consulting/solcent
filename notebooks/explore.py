"""
explore.py - Initial EDA for the Solcent day-ahead dataset.

Written in "percent-cell" format: run as a plain script (python notebooks/explore.py)
OR open in VS Code / Jupyter, where each `# %%` becomes a runnable cell.

It works on the PROCESSED, joined table (data/processed/solcent_dataset.csv) produced by
src/dataset/build_daily_blocks.py -- that is the table you will actually model on.

Figures are saved to reports/figures/. Read the printed numbers alongside them.
"""
# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")          # headless; remove this line to show plots interactively
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
DATA = "data/processed/solcent_dataset.csv"

df = pd.read_csv(DATA, index_col=0, parse_dates=True)
df = df.sort_index()
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

# %% [1] INTEGRITY -- can you trust the table at all?
print("\n--- coverage ---")
print("range :", df.index.min(), "->", df.index.max())
print("duplicated timestamps:", int(df.index.duplicated().sum()))
print("\n--- missing %% per column (top 10) ---")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False).head(10))

# %% helper columns: ensemble day-ahead GHI & cloud, and a daylight mask
ghi_cols   = [c for c in df if c.startswith("shortwave_radiation_previous_day1_")]
cloud_cols = [c for c in df if c.startswith("cloud_cover_previous_day1_")]
df["ghi_d1"]   = df[ghi_cols].mean(axis=1)      # multi-model mean GHI, day-ahead lead
df["cloud_d1"] = df[cloud_cols].mean(axis=1)
df["is_day"]   = df["ghi_d1"] > 5               # leakage-free daylight proxy
day = df[df["is_day"]].copy()
print(f"\nDaylight rows: {len(day)} / {len(df)}  ({100*len(day)/len(df):.0f}% of hours)")

# %% [2] THE TARGET -- shape, diurnal cycle, seasonal cycle
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["load_factor"].plot.hist(bins=50, ax=ax[0], title="Load factor — all hours (note night spike at 0)")
df.groupby(df.index.hour)["load_factor"].mean().plot(ax=ax[1], marker="o", title="Mean load factor by hour (UTC)")
df.groupby(df.index.month)["load_factor"].mean().plot(ax=ax[2], marker="o", title="Mean load factor by month")
fig.tight_layout(); fig.savefig(FIG / "target_profiles.png", dpi=110); plt.close(fig)

# %% [3] NON-STATIONARITY -- installed capacity drifts over time
ax = df["monitoredcapacity"].plot(figsize=(9, 3), title="Monitored capacity (MWp) over time")
ax.figure.savefig(FIG / "capacity_drift.png", dpi=110, bbox_inches="tight"); plt.close(ax.figure)
print("\ncapacity range (MWp):", round(df["monitoredcapacity"].min(), 0), "->",
      round(df["monitoredcapacity"].max(), 0))

# %% [4] FEATURE vs TARGET -- the physics should be visible
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(day["ghi_d1"], day["load_factor"], s=5, alpha=.3)
ax[0].set(xlabel="day-ahead GHI (W/m²)", ylabel="load factor", title="GHI vs load factor")
ax[1].scatter(day["cloud_d1"], day["load_factor"], s=5, alpha=.3, c="grey")
ax[1].set(xlabel="day-ahead cloud cover (%)", ylabel="load factor", title="Cloud vs load factor")
fig.tight_layout(); fig.savefig(FIG / "feature_vs_target.png", dpi=110); plt.close(fig)

# %% [5] CORRELATION with the target (daylight only)
corr = day.select_dtypes("number").corr()["load_factor"].sort_values(ascending=False)
print("\n--- correlation with load_factor (daylight) ---")
print(corr.head(12).round(3))

# %% [6] MULTI-MODEL AGREEMENT -- the basis of your uncertainty signal
print("\n--- day-ahead GHI correlation between NWP models ---")
print(day[ghi_cols].corr().round(3))
day["ghi_spread"] = day[ghi_cols].std(axis=1)
print("mean GHI spread across models (W/m²):", round(day["ghi_spread"].mean(), 1))

# %% [7] THE BAR TO BEAT -- Elia's own published day-ahead forecast
day["elia_err"] = day["measured"] - day["dayaheadforecast"]
print("\n--- Elia operational day-ahead (daylight) ---")
print("MAE  (MW):", round(day["elia_err"].abs().mean(), 1))
print("Bias (MW):", round(day["elia_err"].mean(), 1))
cov = ((day["measured"] >= day["dayaheadconfidence10"]) &
       (day["measured"] <= day["dayaheadconfidence90"])).mean()
print("P10-P90 coverage (should be ~0.80):", round(cov, 3))
ax = (day.groupby(day.index.hour)["elia_err"].apply(lambda s: s.abs().mean())
      .plot(figsize=(8, 3), marker="o", title="Elia day-ahead |error| (MW) by hour"))
ax.figure.savefig(FIG / "elia_error_by_hour.png", dpi=110, bbox_inches="tight"); plt.close(ax.figure)

print(f"\nDone. Figures written to {FIG}/")
