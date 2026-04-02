# BENCHMARKING.md — Benchmarking Guide

This document explains how to benchmark `uncertainty_flow` models using the built-in benchmarking framework.

---

## Quick Start

```bash
# Run benchmark on weather dataset (auto-tuning enabled by default)
uv run python -m uncertainty_flow.cli benchmark --dataset weather

# Run on a specific dataset
uv run python -m uncertainty_flow.cli benchmark --dataset electricity

# List available datasets
uv run python -m uncertainty_flow.cli list-datasets
```

---

## Available Datasets

The library integrates with [HuggingFace datasets](https://huggingface.co/datasets) and includes **108 datasets** for benchmarking:

| Dataset | Domain | Description |
|---------|--------|-------------|
| `weather` | Climate | Weather time series (ts-arena) |
| `exchange_rate` | Finance | Daily exchange rates |
| `electricity` | Energy | Electricity demand time series |
| `m4_daily` | Mixed | M4 daily forecasting competition |
| `m4_hourly` | Mixed | M4 hourly forecasting competition |
| `m4_weekly` | Mixed | M4 weekly forecasting competition |
| `m4_monthly` | Mixed | M4 monthly forecasting competition |
| `m4_quarterly` | Mixed | M4 quarterly forecasting competition |
| `m4_yearly` | Mixed | M4 yearly forecasting competition |
| `weatherbench_daily` | Climate | WeatherBench daily weather |
| `weatherbench_hourly_temperature` | Climate | WeatherBench hourly temperature |
| `monash_electricity_hourly` | Energy | Australian electricity demand |
| `monash_london_smart_meters` | Energy | London smart meter data |
| `ercot` | Energy | Texas electricity demand |
| `monash_traffic` | Transportation | Traffic flow data |
| `monash_pedestrian_counts` | Transportation | Pedestrian counts |
| `taxi_1h` | Transportation | Taxi trip counts (1h) |
| `monash_hospital` | Healthcare | Hospital admissions |
| `monash_fred_md` | Finance | FRED macroeconomic indicators |
| `m5` | Retail | Walmart sales data |

### Filter by Domain

```bash
# List only energy datasets
uv run python -m uncertainty_flow.cli list-datasets --domain Energy

# List only climate datasets
uv run python -m uncertainty_flow.cli list-datasets --domain Climate
```

---

## CLI Commands

### `benchmark` — Run Benchmark

```bash
uv run python -m uncertainty_flow.cli benchmark --dataset <name> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset`, `-d` | (required) | Dataset name or HuggingFace path |
| `--model`, `-m` | `all` | Models to run: `all`, `quantile-forest`, `conformal-regressor`, `conformal-forecaster` |
| `--n-samples`, `-n` | `1000` | Number of samples to use |
| `--horizon`, `-h` | `3` | Forecast horizon for time series models |
| `--n-estimators`, `-e` | `30` | Number of base estimators |
| `--target`, `-t` | `OT` | Target column name |
| `--auto-tune` | `true` | Enable/disable auto-tuning |
| `--target-coverage`, `-c` | `0.9` | Target coverage level for tuning |
| `--tune-samples` | `500` | Samples to use for tuning |
| `--output`, `-o` | `benchmark_results` | Output file prefix |
| `--json-only` | - | Output only JSON |
| `--csv-only` | - | Output only CSV |

### Examples

```bash
# Run all models with auto-tuning (default)
uv run python -m uncertainty_flow.cli benchmark --dataset weather

# Run specific models
uv run python -m uncertainty_flow.cli benchmark --dataset m4_daily \
    --model quantile-forest,conformal-regressor

# Run without auto-tuning (faster, uses default params)
uv run python -m uncertainty_flow.cli benchmark --dataset weather --no-auto-tune

# Custom coverage target and sample size
uv run python -m uncertainty_flow.cli benchmark --dataset electricity \
    --target-coverage 0.8 --n-samples 2000

# Save results
uv run python -m uncertainty_flow.cli benchmark --dataset weather \
    --output my_results
```

### `list-datasets` — List Available Datasets

```bash
# List all datasets
uv run python -m uncertainty_flow.cli list-datasets

# Filter by domain
uv run python -m uncertainty_flow.cli list-datasets --domain Energy
```

### `download-dataset` — Download Dataset for Offline Use

```bash
# Download a single dataset
uv run python -m uncertainty_flow.cli download-dataset m4_daily

# Download to custom cache directory
uv run python -m uncertainty_flow.cli download-dataset weather --cache-dir /path/to/cache
```

---

## Auto-Tuning

Auto-tuning is **enabled by default** and automatically finds optimal hyperparameters for each model to achieve the target coverage level.

### How It Works

1. For each model, the tuner tests multiple parameter combinations
2. Parameters are scored based on coverage calibration and interval sharpness
3. The best parameters are used for the final benchmark

### Search Space

| Model | Parameters Tested |
|-------|------------------|
| `quantile-forest` | `n_estimators`: [20, 30, 50], `min_samples_leaf`: [3, 5, 10] |
| `conformal-regressor` | supported base-estimator params such as `n_estimators`, plus `calibration_size`: [0.15, 0.20, 0.25, 0.30] |
| `conformal-forecaster` | supported base-estimator params such as `n_estimators`, plus `calibration_size`: [0.15, 0.20, 0.25, 0.30] and `lags`: [1, 2, 3] |

### Disabling Auto-Tuning

```bash
# Faster runs with default parameters
uv run python -m uncertainty_flow.cli benchmark --dataset weather --no-auto-tune
```

---

## Output Format

### JSON Output

```json
{
  "metadata": {
    "run_id": "3d115493",
    "timestamp": "2026-03-31T13:30:22Z",
    "dataset": "weather",
    "domain": "Climate",
    "n_samples": 1000,
    "horizon": 3,
    "auto_tune": true,
    "target_coverage": 0.9
  },
  "results": [
    {
      "model": "conformal-forecaster",
      "coverage_90": 0.9449,
      "coverage_80": 0.8788,
      "sharpness_90": 0.0223,
      "sharpness_80": 0.0148,
      "winkler_90": 0.0260,
      "winkler_80": 0.0197,
      "pinball_loss": 0.0027,
      "train_time_sec": 0.091,
      "n_samples": 997,
      "tuned_params": {"n_estimators": 50, "calibration_size": 0.25, "lags": 1},
      "was_tuned": true
    }
  ]
}
```

### Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| `coverage_90` | Fraction of true values within 90% prediction interval | ~0.90 |
| `coverage_80` | Fraction of true values within 80% prediction interval | ~0.80 |
| `sharpness_90` | Average width of 90% prediction intervals | Lower is better |
| `winkler_90` | Winkler score for 90% intervals | Lower is better |
| `pinball_loss` | Pinball loss at quantile 0.1 | Lower is better |
| `train_time_sec` | Training time in seconds | - |

---

## Using the Library Programmatically

### Python API

```python
from uncertainty_flow.benchmarking import BenchmarkConfig, BenchmarkRunner

# Create config with auto-tuning enabled
config = BenchmarkConfig(
    dataset_name="weather",
    n_samples=1000,
    horizon=3,
    auto_tune=True,
    target_coverage=0.9,
)

# Run benchmark
runner = BenchmarkRunner(config)
runner.load_data()
result = runner.run_all()

# Access results
for model_result in result.models:
    print(f"{model_result.model_name}:")
    print(f"  Coverage @ 90%: {model_result.coverage_90}")
    print(f"  Sharpness @ 90%: {model_result.sharpness_90}")
    print(f"  Tuned params: {model_result.tuned_params}")

# Save results
runner.save_json("results.json")
runner.save_csv("results.csv")
```

### Auto-Tuning Only

```python
from uncertainty_flow.benchmarking.tuning import auto_tune_model, TuningConfig
from uncertainty_flow.benchmarking.datasets import load_dataset

# Load data
df, _ = load_dataset("weather", n_samples=500)
target = "OT"

# Tune a specific model
config = TuningConfig(target_coverage=0.9, n_samples=500)
result = auto_tune_model(
    model_name="conformal-forecaster",
    df=df,
    target=target,
    horizon=3,
    config=config,
)

print(f"Best params: {result.best_params}")
print(f"Coverage: {result.coverage_90}")
```

---

## Best Practices

1. **Use Auto-Tuning** — It significantly improves coverage calibration with minimal performance overhead.

2. **Choose Appropriate Sample Size** — Use at least 500 samples for reliable tuning, 1000+ for final benchmarks.

3. **Match Horizon to Dataset** — Set `--horizon` based on your forecasting needs. Larger horizons require more data.

4. **Compare Multiple Models** — Different models excel on different datasets. Run `all` models to find the best fit.

5. **Consider Coverage vs Sharpness Trade-off** — A model with slightly lower coverage but much tighter intervals may be preferable for some applications.

---

## Troubleshooting

### "Dataset not found"

```bash
# Verify dataset name
uv run python -m uncertainty_flow.cli list-datasets | grep <name>

# Use full HuggingFace path if needed
uv run python -m uncertainty_flow.cli benchmark \
    --dataset autogluon/chronos_datasets/m4_daily
```

### Poor Coverage Results

- Enable auto-tuning to find better hyperparameters
- Increase `tune-samples` for more reliable tuning
- Try a different model — some models work better on certain data patterns

### Slow Benchmark Runs

- Reduce `n-samples` for faster iteration
- Disable auto-tuning for quick experiments
- Reduce model complexity (fewer estimators)
