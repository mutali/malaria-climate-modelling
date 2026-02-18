"""
predict.py
----------
CHAP entry point: generate malaria case predictions for Rwanda.

Loads a trained model and generates probabilistic forecasts for future
time periods using future climate_data covariates.

Called by CHAP as:
    python predict.py <model> <historic_data> <future_data> <out_file>

Output CSV format (CHAP standard):
    time_period, location, sample_0[, sample_1, ...]

Variable naming:
    malaria_data  -> disease_cases, population  (health surveillance)
    climate_data  -> rainfall, mean_temperature, max_temperature,
                     min_temperature, relative_humidity, ndvi
"""

import sys
import joblib
import numpy as np
import pandas as pd

# Number of Monte Carlo samples to output (simulates predictive uncertainty)
N_SAMPLES = 200


def predict(model_path, historic_data_path, future_data_path, out_file_path):
    # Load trained model artefact
    artefact = joblib.load(model_path)
    pipeline = artefact["pipeline"]
    reference_rate = artefact["reference_rate"]
    features = artefact["features"]
    default_population = artefact["default_population"]

    # Load future climate_data — no disease_cases column here
    future_df = pd.read_csv(future_data_path)

    X_future = future_df[features].fillna(0)

    # Population: use column if present, else fall back to training mean
    if "population" in future_df.columns:
        population = future_df["population"].clip(lower=1).fillna(default_population).values
    else:
        population = np.full(len(future_df), default_population)

    # Predict rate ratio from climate_data features
    predicted_rate_ratio = pipeline.predict(X_future)

    # Expected cases = population x reference_rate x predicted_rate_ratio
    mu = population * reference_rate * np.clip(predicted_rate_ratio, 1e-6, None)

    # Generate Poisson samples for probabilistic output
    rng = np.random.default_rng(42)
    samples = rng.poisson(mu, size=(N_SAMPLES, len(mu))).T  # (n_rows, N_SAMPLES)
    sample_df = pd.DataFrame(
        samples, columns=[f"sample_{i}" for i in range(N_SAMPLES)]
    )
    output_df = pd.concat(
        [future_df[["time_period", "location"]].reset_index(drop=True), sample_df],
        axis=1,
    )

    output_df.to_csv(out_file_path, index=False)
    print(f"[predict] Predictions saved to {out_file_path}")
    print(f"[predict] {len(output_df):,} rows x {N_SAMPLES} samples")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python predict.py <model_path> <historic_data_path> "
            "<future_data_path> <out_file_path>"
        )
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
