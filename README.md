# Malaria & Climate Modelling — Kigali Workshop (CHAP / uv edition)



---

This repository is the **Python + DHIS2 CHAP** port of the Rwanda malaria–climate model from the Kigali workshop, using [uv](https://docs.astral.sh/uv/) for dependency management.

The model learns the relationship between climate covariates (`climate_data`) and malaria cases (`malaria_data`) at sector level across Rwanda, and generates probabilistic predictions compatible with the DHIS2 CHAP Modeling App.

## Requirements

You need [uv](https://docs.astral.sh/uv/) installed:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Repository structure

```
.
├── MLproject           # CHAP integration configuration
├── train.py            # Training logic
├── predict.py          # Prediction logic
├── pyproject.toml      # Python dependencies (uv reads this)
├── isolated_run.py     # Test without CHAP installed
├── input/              # Sample training and forecast data
│   ├── train_data.csv  # malaria_data + climate_data (historical)
│   └── future_data.csv # climate_data only (for forecasting)
└── output/             # Generated models and predictions (gitignored)
```

No other setup needed — `uv run` automatically creates the virtual environment and installs dependencies on first use.

## Variable naming

This model uses two named data sources that CHAP maps automatically:

| CHAP name | Contents |
|---|---|
| `malaria_data` | `disease_cases`, `population`, `time_period`, `location` |
| `climate_data` | `rainfall`, `mean_temperature`, `max_temperature`, `min_temperature`, `relative_humidity`, `ndvi`, `time_period`, `location` |


## Running without CHAP (for development)

The quickest way to check everything works:

```
uv run python isolated_run.py
```

This trains the model on `input/train_data.csv` and generates predictions into `output/`. After running, check:

- `output/model.pkl` — the trained model
- `output/predictions.csv` — predicted malaria cases (columns: `time_period`, `location`, `sample_0`, ..., `sample_199`)

### Training

`train.py` reads the merged CSV (malaria_data + climate_data columns), fits a Poisson GLM on climate covariates, and saves the model:

```python
def train(train_data_path, model_path):
    df = pd.read_csv(train_data_path)
    features = df[["rainfall", "mean_temperature", "relative_humidity", "ndvi"]].fillna(0)
    # disease_cases = malaria_data target; covariates = climate_data features
    ...
    pipeline.fit(features, rate_ratio)
    joblib.dump(artefact, model_path)
```

### Predicting

`predict.py` loads the trained model and uses future `climate_data` to generate 200 Poisson predictive samples per row:

```python
def predict(model_path, historic_data_path, future_data_path, out_file_path):
    artefact = joblib.load(model_path)
    future_df = pd.read_csv(future_data_path)
    features = future_df[["rainfall", "mean_temperature", "relative_humidity", "ndvi"]].fillna(0)
    predicted_rate_ratio = pipeline.predict(features)
    mu = population * reference_rate * predicted_rate_ratio
    # Poisson samples → sample_0, sample_1, ..., sample_199
    ...
```

## Making changes

### Change the model type

`train.py` uses `PoissonRegressor` from scikit-learn. You can try:

```python
# Gradient Boosting for climate-disease non-linear effects
from sklearn.ensemble import GradientBoostingRegressor
```

Remember to update `predict.py` to match any feature changes.

### Change which climate covariates are used

Both `train.py` and `predict.py` use:

```python
CLIMATE_FEATURES = ["rainfall", "mean_temperature", "relative_humidity", "ndvi"]
```

To add temperature range as a feature, for example:

```python
df["temp_range"] = df["max_temperature"] - df["min_temperature"]
CLIMATE_FEATURES = ["rainfall", "mean_temperature", "temp_range", "relative_humidity", "ndvi"]
```

### Add lagged covariates

To test whether rainfall 1 month prior predicts cases better:

```python
df = df.sort_values(["location", "time_period"])
df["rainfall_lag1"] = df.groupby("location")["rainfall"].shift(1)
CLIMATE_FEATURES = ["rainfall_lag1", "mean_temperature", "relative_humidity", "ndvi"]
```

After any change, always re-run `isolated_run.py` to verify everything works.

## Running through CHAP

After [installing chap-core](https://dhis2-chap.github.io/chap-core/chap-cli/chap-core-cli-setup.html):

```bash
# Evaluate against a standard dataset
chap evaluate \
  --model-name /path/to/malaria-climate-modelling-kigali \
  --dataset-name ISIMIP_dengue_harmonized \
  --dataset-country rwanda \
  --report-filename report.pdf

# Or with a local CSV
chap evaluate \
  --model-name /path/to/malaria-climate-modelling-kigali \
  --dataset-csv input/train_data.csv \
  --report-filename report.pdf
```

The `MLproject` file tells CHAP how to call the model:

```yaml
name: malaria_climate_kigali

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```


## References

- DHIS2 CHAP Platform: [dhis2-chap.github.io/chap-core](https://dhis2-chap.github.io/chap-core/)
- Minimalist CHAP Python example: [dhis2-chap/minimalist_example_uv](https://github.com/dhis2-chap/minimalist_example_uv)
