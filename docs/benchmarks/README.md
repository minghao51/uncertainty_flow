# Benchmark Results

This document presents comprehensive benchmark results comparing `uncertainty_flow` models with conventional regression and forecasting baselines.

## Overview

**Benchmark Date:** April 2026  
**Sample Size:** 500 observations per dataset  
**Forecast Horizon:** 3 steps  
**Auto-tuning:** Disabled (default parameters)

Canonical committed artifacts in this repo live in `docs/benchmarks/` and the
matching generated `results/comprehensive_v2_*` files. Older non-`v2` result
snapshots and quick-trial CSVs have been removed to avoid duplicate sources of truth.

## Datasets

| Dataset | Domain | Target | Description |
|---------|--------|--------|-------------|
| `weather` | Climate | OT | Weather time series with 22 features |
| `electricity` | Energy | OT | Electricity demand with 320 features |
| `exchange_rate` | Finance | OT | Daily exchange rates with 8 features |

## Models Compared

### Uncertainty Flow Models
- **quantile-forest** — Quantile Forest Forecaster using sklearn RandomForest with quantile regression
- **conformal-regressor** — Conformal prediction wrapper for regression models
- **conformal-forecaster** — Conformal prediction for time series forecasting with lag features

### Conventional Regression Baselines
- **linear-regression** — Ordinary Least Squares with conformalized intervals
- **ridge-regression** — Ridge Regression (L2 regularization) with conformalized intervals
- **random-forest** — Random Forest with conformalized intervals
- **gradient-boosting** — Gradient Boosting Regressor with conformalized intervals

### Simple Time Series Baselines
- **naive-forecast** — Last observed value with historical error-based intervals
- **moving-average** — Rolling window average with error-based intervals

---

## Results Summary

### Overall Rankings (by average Winkler @ 90%)

| Rank | Model | Avg Winkler @ 90% | Avg Coverage @ 90% | Avg Sharpness @ 90% | Avg Train Time |
|------|-------|-------------------|-------------------|---------------------|----------------|
| 1 | **quantile-forest** | 102.2600 | 0.823 | 81.5945 | 0.255s |
| 2 | **random-forest** | 108.7778 | 0.799 | 98.1789 | 0.108s |
| 3 | **conformal-forecaster** | 114.1700 | 0.792 | 103.7836 | 0.287s |
| 4 | conformal-regressor | 122.1890 | 0.753 | 108.5528 | 0.302s |
| 5 | gradient-boosting | 122.1890 | 0.753 | 108.5528 | 0.356s |
| 6 | linear-regression | 283.6228 | 0.802 | 274.6183 | 0.024s |
| 7 | ridge-regression | 283.8519 | 0.779 | 275.1602 | 0.023s |
| 8 | moving-average | 458.4653 | 0.345 | 239.2003 | 0.002s |
| 9 | naive-forecast | 481.4650 | 0.353 | 256.1676 | 0.000s |

---

## Results by Dataset

### Weather (Climate)

| Model | Coverage @ 90% | Coverage @ 80% | Sharpness @ 90% | Winkler @ 90% | Train Time |
|-------|---------------|---------------|-----------------|---------------|------------|
| **random-forest** | 0.446 | 0.202 | 0.0374 | **0.0585** | 0.066s |
| linear-regression | 0.934 | 0.904 | 0.0636 | 0.0685 | 0.007s |
| conformal-forecaster | 0.540 | 0.295 | 0.0301 | 0.0701 | 0.054s |
| quantile-forest | 0.764 | 0.718 | 0.0151 | 0.0809 | 0.049s |
| ridge-regression | 0.816 | 0.750 | 0.0609 | 0.0815 | 0.006s |
| conformal-regressor | 0.370 | 0.244 | 0.0322 | 0.1025 | 0.059s |
| gradient-boosting | 0.370 | 0.244 | 0.0322 | 0.1025 | 0.055s |
| naive-forecast | 0.298 | 0.286 | 0.0314 | 0.4382 | 0.000s |
| moving-average | 0.280 | 0.222 | 0.0251 | 0.4859 | 0.002s |

**Best Model:** `random-forest` (Winkler: 0.0585)

---

### Electricity (Energy)

| Model | Coverage @ 90% | Coverage @ 80% | Sharpness @ 90% | Winkler @ 90% | Train Time |
|-------|---------------|---------------|-----------------|---------------|------------|
| **quantile-forest** | 0.928 | 0.856 | 244.70 | **306.01** | 0.686s |
| random-forest | 0.970 | 0.942 | 293.94 | 325.71 | 0.204s |
| conformal-forecaster | 0.958 | 0.874 | 310.99 | 342.08 | 0.779s |
| conformal-regressor | 0.932 | 0.856 | 325.03 | 365.85 | 0.819s |
| gradient-boosting | 0.932 | 0.856 | 325.03 | 365.85 | 0.984s |
| linear-regression | 0.980 | 0.960 | 823.44 | 849.81 | 0.062s |
| ridge-regression | 0.980 | 0.960 | 825.04 | 850.54 | 0.060s |
| naive-forecast | 0.724 | 0.618 | 768.29 | 1435.84 | 0.000s |
| moving-average | 0.718 | 0.564 | 717.43 | 1366.59 | 0.002s |

**Best Model:** `quantile-forest` (Winkler: 306.01)

---

### Exchange Rate (Finance)

| Model | Coverage @ 90% | Coverage @ 80% | Sharpness @ 90% | Winkler @ 90% | Train Time |
|-------|---------------|---------------|-----------------|---------------|------------|
| **conformal-forecaster** | 0.880 | 0.753 | 0.3262 | **0.3613** | 0.029s |
| random-forest | 0.982 | 0.960 | 0.5559 | 0.5665 | 0.055s |
| conformal-regressor | 0.958 | 0.862 | 0.5986 | 0.6145 | 0.029s |
| gradient-boosting | 0.958 | 0.862 | 0.5986 | 0.6145 | 0.028s |
| quantile-forest | 0.778 | 0.728 | 0.0681 | 0.6894 | 0.030s |
| linear-regression | 0.492 | 0.420 | 0.3522 | 0.9911 | 0.004s |
| ridge-regression | 0.540 | 0.464 | 0.3833 | 0.9315 | 0.004s |
| naive-forecast | 0.038 | 0.038 | 0.1779 | 8.1133 | 0.000s |
| moving-average | 0.038 | 0.038 | 0.1411 | 8.3211 | 0.002s |

**Best Model:** `conformal-forecaster` (Winkler: 0.3613)

---

## Key Findings

### 1. Best Overall Model
**quantile-forest** achieves the best overall performance with an average Winkler score of 102.26, demonstrating excellent balance between coverage calibration and interval sharpness.

### 2. Best Baseline Model
**random-forest** with conformalized intervals performs competitively, ranking 2nd overall with an average Winkler score of 108.78.

### 3. Coverage Analysis

| Model | Avg Coverage @ 90% | Deviation from Target |
|-------|-------------------|----------------------|
| quantile-forest | 0.823 | -0.077 |
| linear-regression | 0.802 | -0.098 |
| random-forest | 0.799 | -0.101 |
| conformal-forecaster | 0.792 | -0.108 |

### 4. Training Efficiency

| Model | Avg Train Time | Speed Rank |
|-------|---------------|------------|
| naive-forecast | 0.000s | 1 |
| moving-average | 0.002s | 2 |
| ridge-regression | 0.023s | 3 |
| linear-regression | 0.024s | 4 |
| conformal-forecaster | 0.029s | 5 |
| random-forest | 0.108s | 6 |
| quantile-forest | 0.255s | 7 |

---

## Metrics Explained

For metric definitions (coverage, sharpness, Winkler score, pinball loss), see [../guides/calibration.md](../guides/calibration.md).

---

## How to Reproduce

### Run Comprehensive Benchmark

```bash
# Run on all default datasets
uv run python scripts/comprehensive_benchmark.py --all-datasets --n-samples 500 --output results/comprehensive_v2

# Run on specific dataset
uv run python scripts/comprehensive_benchmark.py --dataset weather --n-samples 500 --output results/comprehensive_v2_weather
```

### Generate Comparison Report

```bash
uv run python scripts/generate_report.py --output results/benchmark_report.md
```

### Using the CLI

```bash
# Quick benchmark without auto-tuning
uv run python -m uncertainty_flow.cli benchmark --dataset weather --n-samples 500 --no-auto-tune

# With auto-tuning (slower but better calibrated)
uv run python -m uncertainty_flow.cli benchmark --dataset weather --auto-tune
```

---

## Files

| File | Description |
|------|-------------|
| `comprehensive_v2_weather.json` | Full results for weather dataset |
| `comprehensive_v2_electricity.json` | Full results for electricity dataset |
| `comprehensive_v2_exchange_rate.json` | Full results for exchange_rate dataset |
| `comparison_table.csv` | Combined comparison table for all datasets |

---

## Recommendations

1. **For general use:** Start with `quantile-forest` — it provides the best overall balance of coverage and sharpness.

2. **For time series:** Use `conformal-forecaster` which explicitly models temporal dependencies through lag features.

3. **For fast inference:** `linear-regression` or `ridge-regression` with conformalized intervals provide reasonable coverage with minimal training time.

4. **For high-dimensional data:** `random-forest` handles many features well and provides competitive performance.

5. **Avoid simple baselines:** `naive-forecast` and `moving-average` show poor coverage calibration and should only be used as sanity checks.
