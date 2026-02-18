"""
train.py
--------
CHAP entry point: train the malaria-climate model for Rwanda.

Reads a CSV (malaria_data + climate_data columns merged), fits a
Poisson regression of disease_cases on climate covariates,
and saves the trained model to file.

Called by CHAP as:
    python train.py <train_data> <model>

Variable naming:
    malaria_data  -> disease_cases, population  (health surveillance)
    climate_data  -> rainfall, mean_temperature, max_temperature,
                     min_temperature, relative_humidity, ndvi
"""

import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Climate covariates derived from climate_data
CLIMATE_FEATURES = [
    "rainfall",
    "mean_temperature",
    "relative_humidity",
    "ndvi",
]


def train(train_data_path, model_path):
    # Load merged malaria_data + climate_data CSV
    df = pd.read_csv(train_data_path)

    features = df[CLIMATE_FEATURES].fillna(0)

    population = df["population"].clip(lower=1).fillna(1)
    default_population = float(population.mean())

    # disease_cases = malaria_data target variable
    target = df["disease_cases"].fillna(0).clip(lower=0)

    # Poisson GLM with standardised climate_data features.
    # We fit the rate ratio (observed / expected) so the model learns
    # relative risk from climate signals independent of population size.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(max_iter=500, alpha=0.01)),
    ])

    reference_rate = target.sum() / population.sum()
    expected = population * reference_rate
    rate_ratio = (target / expected.clip(lower=1e-6)).clip(lower=1e-6)

    pipeline.fit(features, rate_ratio)

    artefact = {
        "pipeline": pipeline,
        "reference_rate": float(reference_rate),
        "default_population": default_population,
        "features": CLIMATE_FEATURES,
    }
    joblib.dump(artefact, model_path)
    print(f"[train] Model saved to {model_path}")
    print(f"[train] Reference rate: {reference_rate:.6f} cases/person")
    print(f"[train] Trained on {len(df):,} rows from {train_data_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <train_data_path> <model_path>")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2])
