"""
explore.py - Quick EDA on the joined dataset (run after build_daily_blocks.py).
Convert to a notebook if you prefer; kept as a script so it lives cleanly in git.
"""
import pandas as pd

df = pd.read_parquet("data/processed/solcent_dataset.parquet")
print(df.shape)
print(df.head())
print(df.describe().T)

# Sanity checks to look at:
#  - GHI (shortwave_radiation_*_ecmwf) vs measured generation: should correlate strongly
#  - Elia dayaheadforecast error: (measured - dayaheadforecast) -- where is it largest?
#  - seasonal coverage: do you have both summers and both winters in the range?
#  - multi-model spread vs realised error: does spread widen on hard days?
