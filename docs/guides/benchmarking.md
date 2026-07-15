# BENCHMARKING.md — Benchmarking Guide

This document explains how to benchmark `uncertainty_flow` models using the built-in benchmarking framework.

---

## Quick Start

```bash
# Run the canonical pipeline benchmark on a dataset
mkdir -p results
uv run python -m uncertainty_flow.cli benchmark --dataset weather --output results/weather

# Run selected registered providers without tuning
uv run python -m uncertainty_flow.cli benchmark \
  --dataset weather --model quantile-forest,conformal-regressor --no-auto-tune

# Generate a report from saved results
uv run python benchmarks/generate_report.py --results-dir results --output results/report.md
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

Benchmark orchestration is implemented by the Hamilton-backed pipeline
coordinators and exposed through the CLI and typed benchmarking contracts.

Pipeline lifecycle per run:

1. Resolve `RunRequest` and validate provider/configuration metadata.
2. Load through `DatasetRegistry` and derive content-based identity.
3. Materialize Bronze, Silver, and Gold lineage in staging.
4. Fit registered providers and evaluate registered metrics.
5. Verify checksummed evidence and promote the final Platinum manifest last.

### `benchmark` — Run Benchmark

```bash
uv run python -m uncertainty_flow.cli benchmark --dataset <name> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset`, `-d` | (required) | Dataset name or HuggingFace path |
| `--model`, `-m` | `all` | Comma-separated registered model names, or `all` |
| `--samples`, `-s` | `100` | Number of dataset rows to load |
| `--horizon`, `-h` | `3` | Forecast horizon for time series models |
| `--n-estimators`, `-e` | `30` | Number of base estimators |
| `--target`, `-t` | dataset default | Target column name |
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
    --target-coverage 0.8 --samples 2000

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
2. Parameters are scored on validation splits only (no fit/predict on the same rows)
3. The best parameters are used for the final benchmark on a separate untouched test holdout

### Validation Strategy (Leakage-Safe)

- Tabular tuning defaults to random holdout; for small datasets it uses CV.
- Time-series tuning defaults to temporal holdout.
- Optional hybrid validation uses outer holdout + inner out-of-sample CV on outer-train only.
- The selector is deterministic and logs chosen strategy and rationale.

Example strategy logs:

```text
validation_strategy strategy=temporal_holdout reason=time_series task defaults to temporal holdout ...
tuning_validation_plan model=conformal-forecaster strategy=temporal_holdout reason=time_series task defaults...
```

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
  "manifest": {
    "identity": {"run_id": "<content-derived-sha256>"},
    "status": "success",
    "verification_passed": true
  },
  "verification": {"passed": true, "checks": []},
  "artifacts": [],
  "model_results": [
    {
      "model_id": "conformal-forecaster",
      "provider": "conformal-forecaster",
      "status": "success",
      "train_time_sec": 0.091,
      "evaluation_row_count": 200,
      "resolved_parameters": {"n_estimators": 50, "calibration_size": 0.25},
      "metrics": {
        "coverage_90": 0.9449,
        "winkler_90": 0.0260,
        "pinball": 0.0027
      }
    }
  ],
  "reused": false
}
```

Serialized benchmark output is the typed `PipelineRunResult` and uses:

- `manifest`, `verification`, `artifacts`, and `model_results` fields

### Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| `coverage_90` | Fraction of true values within 90% prediction interval | ~0.90 |
| `coverage_80` | Fraction of true values within 80% prediction interval | ~0.80 |
| `sharpness_90` | Average width of 90% prediction intervals | Lower is better |
| `winkler_90` | Winkler score for 90% intervals | Lower is better |
| `pinball` | Mean pinball loss across available quantiles | Lower is better |
| `train_time_sec` | Training time in seconds | - |

---

## Using the Library Programmatically

### Python API

```python
from uncertainty_flow.benchmarking import ModelMatrixCoordinator
from uncertainty_flow.benchmarking.contracts import RunRequest

request = RunRequest(
    dataset={"id": "fixture", "provider": "local_parquet", "uri": "fixture.parquet", "target": "y"},
    validation={"strategy": "temporal_holdout", "test_size": 0.2, "preserve_order": True},
    models=({"id": "conformal-forecaster", "provider": "conformal-forecaster"},),
    evaluation={"metrics": ["coverage", "winkler", "crps"]},
    storage={"provider": "local", "root": "data"},
)
result = ModelMatrixCoordinator(storage_root="data").run_with_lock(request, frame)
print(result.model_results[0].metrics)
```

## Extending Benchmark Models

Use the provider seam for new benchmark model adapters.

- Stable built-in names remain: `quantile-forest`, `conformal-regressor`, `conformal-forecaster`
- Provider contract lives in `providers.py` (`BenchmarkModelProvider`)
- Register provider parameters in `registry.py`; unknown parameters are rejected before execution.
- Use `ModelMatrixCoordinator.run_with_lock()` for custom integrations.

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
