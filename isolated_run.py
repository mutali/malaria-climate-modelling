"""
isolated_run.py
---------------
Test train -> predict pipeline without needing CHAP installed.

Uses sample data from input/ and writes results to output/.
Run with:
    uv run python isolated_run.py
"""

from pathlib import Path
from train import train
from predict import predict

# Paths matching the minimalist_example_uv layout
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_DATA  = INPUT_DIR / "train_data.csv"
FUTURE_DATA = INPUT_DIR / "future_data.csv"
MODEL_PATH  = OUTPUT_DIR / "model.pkl"
PREDS_PATH  = OUTPUT_DIR / "predictions.csv"

print("=" * 55)
print("  Malaria-Climate Kigali -- Isolated Test Run")
print("=" * 55)

print("\n[1/2] Training model on malaria_data + climate_data...")
train(str(TRAIN_DATA), str(MODEL_PATH))

print("\n[2/2] Generating predictions from climate_data forecasts...")
predict(str(MODEL_PATH), str(TRAIN_DATA), str(FUTURE_DATA), str(PREDS_PATH))

print(f"\n  Done. Check {OUTPUT_DIR}/ for:")
print(f"    {MODEL_PATH}    -- trained model")
print(f"    {PREDS_PATH}  -- predicted disease cases")

# Readable preview: posterior mean + 95% interval
import pandas as pd
preds = pd.read_csv(PREDS_PATH)
sample_cols = [c for c in preds.columns if c.startswith("sample_")]
sample_data = preds[sample_cols]
summary = pd.DataFrame({
    "time_period":     preds["time_period"],
    "location":        preds["location"],
    "predicted_mean":  sample_data.mean(axis=1).round(1),
    "lower_95":        sample_data.quantile(0.025, axis=1).round(0).astype(int),
    "upper_95":        sample_data.quantile(0.975, axis=1).round(0).astype(int),
})

print(f"\nPredictions preview ({len(preds)} rows, {len(sample_cols)} samples each):")
print(summary.to_string(index=False))
