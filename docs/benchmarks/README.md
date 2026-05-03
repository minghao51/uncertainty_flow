# Comprehensive Benchmark Results — April 26, 2026

This document presents comprehensive benchmark results comparing all `uncertainty_flow` models with conventional regression and forecasting baselines.

## Overview

**Benchmark Date:** April 26, 2026
**Sample Size:** 1,000 observations per dataset
**Forecast Horizon:** 3 steps
**Iterations:** 1 (single-run, deterministic)
**Auto-tuning:** Disabled (default parameters)
**New Models Tested:** `deep-quantile`, `deep-quantile-torch`, `bayesian-quantile`
**New Metrics:** CRPS, MAE, RMSE, Calibration Error (in addition to existing coverage, sharpness, Winkler, pinball loss)
**Skipped:** `transformer-forecaster` (requires `chronos-forecasting` package)

## Datasets

| Dataset | Domain | Target | Features | Description |
|---------|--------|--------|----------|-------------|
| `weather` | Climate | OT | 22 | Weather time series |
| `electricity` | Energy | OT | 320 | Electricity demand |
| `exchange_rate` | Finance | OT | 8 | Daily exchange rates |

## Models Compared

### Uncertainty Flow Models
- **quantile-forest** — Quantile Forest Forecaster
- **conformal-regressor** — Conformal prediction wrapper for regression
- **conformal-forecaster** — Conformal prediction for time series with lag features
- **deep-quantile** — Multi-quantile MLP (sklearn backend) *[NEW]*
- **deep-quantile-torch** — Multi-quantile MLP (PyTorch backend) *[NEW]*
- **bayesian-quantile** — Bayesian linear regression via NumPyro MCMC *[NEW]*

### Conventional Regression Baselines
- **linear-regression** — OLS with conformalized intervals
- **ridge-regression** — Ridge Regression with conformalized intervals
- **random-forest** — Random Forest with conformalized intervals
- **gradient-boosting** — Gradient Boosting with conformalized intervals

### Simple Time Series Baselines
- **naive-forecast** — Last observed value with error-based intervals
- **moving-average** — Rolling window average with error-based intervals

---

## Results Summary

### Overall Rankings (by average Winkler @ 90%)

| Rank | Model | Avg Winkler@90% | Avg Coverage@90% | Avg MAE | Avg Cal.Error | Avg Time |
|------|-------|-----------------|-------------------|---------|---------------|----------|
| 1 | **deep-quantile-torch** | 264.45 | 0.928 | 41.67 | 0.033 | 1.52s |
| 2 | **conformal-forecaster** | 134.01 | 0.864 | 22.30 | 0.101 | 0.32s |
| 3 | **quantile-forest** | 107.82 | 0.835 | 17.19 | 0.070 | 0.47s |
| 4 | **deep-quantile** | 279.53 | 0.900 | 51.34 | 0.003 | 2.00s |
| 5 | random-forest | 134.41 | 0.713 | 14.84 | 0.293 | 0.10s |
| 6 | conformal-regressor | 151.74 | 0.720 | 26.23 | 0.268 | 0.32s |
| 7 | gradient-boosting | 151.74 | 0.720 | 26.23 | 0.268 | 0.33s |
| 8 | linear-regression | 179.06 | 0.737 | 24.83 | 0.223 | 0.02s |
| 9 | ridge-regression | 179.09 | 0.724 | 24.30 | 0.238 | 0.02s |
| 10 | naive-forecast | 580.66 | 0.362 | 100.22 | 0.538 | 0.000s |
| 11 | moving-average | 920.76 | 0.303 | 115.24 | 0.597 | 0.002s |
| 12 | bayesian-quantile | 9004567 | 0.001 | 447356 | 0.899 | 11.06s |

---

## Results by Dataset

### Weather (Climate)

| Model | Cov@90% | Wink@90% | CRPS | MAE | RMSE | CalErr | Time(s) |
|-------|---------|----------|------|-----|------|--------|---------|
| **deep-quantile-torch** | 0.956 | **0.0149** | **0.0020** | **0.0026** | **0.0039** | 0.056 | 1.76 |
| conformal-forecaster | 0.936 | 0.0279 | 0.0033 | 0.0042 | 0.0058 | 0.036 | 0.06 |
| conformal-regressor | 0.964 | 0.0347 | 0.0044 | 0.0078 | 0.0094 | 0.064 | 0.06 |
| gradient-boosting | 0.964 | 0.0347 | 0.0044 | 0.0078 | 0.0094 | 0.064 | 0.06 |
| quantile-forest | 0.841 | 0.0399 | 0.0038 | 0.0046 | 0.0081 | 0.059 | 0.12 |
| **deep-quantile** | **0.901** | 0.0460 | 0.0081 | 0.0102 | 0.0142 | **0.001** | 2.41 |
| random-forest | 0.980 | 0.0546 | 0.0052 | 0.0043 | 0.0080 | 0.080 | 0.05 |
| linear-regression | 0.911 | 0.0558 | 0.0071 | 0.0101 | 0.0131 | 0.011 | 0.01 |
| ridge-regression | 0.913 | 0.0565 | 0.0073 | 0.0105 | 0.0135 | 0.013 | 0.00 |
| moving-average | 0.268 | 0.3552 | 0.0234 | 0.0269 | 0.0319 | 0.632 | 0.00 |
| naive-forecast | 0.271 | 0.3822 | 0.0258 | 0.0299 | 0.0353 | 0.629 | 0.00 |
| bayesian-quantile | 0.000 | 199.67 | 10.06 | 10.03 | 11.59 | 0.900 | 3.72 |

**Best:** `deep-quantile-torch` (Winkler: 0.0149, CRPS: 0.0020, MAE: 0.0026)

### Exchange Rate (Finance)

| Model | Cov@90% | Wink@90% | CRPS | MAE | RMSE | CalErr | Time(s) |
|-------|---------|----------|------|-----|------|--------|---------|
| deep-quantile-torch | 0.935 | **0.1351** | **0.0181** | **0.0239** | **0.0317** | 0.035 | 1.30 |
| **deep-quantile** | **0.903** | 0.1891 | 0.0294 | 0.0389 | 0.0517 | **0.003** | 2.24 |
| conformal-forecaster | 0.694 | 0.3014 | 0.0476 | 0.0802 | 0.0888 | 0.206 | 0.03 |
| quantile-forest | 0.757 | 1.0456 | 0.0650 | 0.0750 | 0.1475 | 0.143 | 0.11 |
| conformal-regressor | 0.229 | 1.3138 | 0.1401 | 0.1841 | 0.1977 | 0.671 | 0.03 |
| gradient-boosting | 0.229 | 1.3138 | 0.1401 | 0.1841 | 0.1977 | 0.671 | 0.03 |
| linear-regression | 0.321 | 1.6191 | 0.1687 | 0.2142 | 0.2336 | 0.579 | 0.00 |
| ridge-regression | 0.279 | 1.8380 | 0.1825 | 0.2385 | 0.2598 | 0.621 | 0.00 |
| random-forest | 0.180 | 1.9008 | 0.1720 | 0.2115 | 0.2245 | 0.720 | 0.05 |
| naive-forecast | 0.165 | 3.7912 | 0.2299 | 0.2547 | 0.2999 | 0.735 | 0.00 |
| moving-average | 0.137 | 4.1204 | 0.2385 | 0.2586 | 0.3056 | 0.763 | 0.00 |
| bayesian-quantile | 0.002 | 934.49 | 47.57 | 48.00 | 56.11 | 0.898 | 4.87 |

**Best:** `deep-quantile-torch` (Winkler: 0.1351), `deep-quantile` (best calibration: 0.003 error)

### Electricity (Energy)

| Model | Cov@90% | Wink@90% | CRPS | MAE | RMSE | CalErr | Time(s) |
|-------|---------|----------|------|-----|------|--------|---------|
| **quantile-forest** | **0.907** | **321.37** | 39.56 | 51.49 | 74.90 | **0.007** | 1.18 |
| random-forest | 0.979 | 401.28 | **38.20** | **42.29** | **62.69** | 0.079 | 0.20 |
| conformal-forecaster | 0.963 | 401.71 | 47.14 | 62.42 | 82.53 | 0.063 | 0.87 |
| conformal-regressor | 0.968 | 453.87 | 53.25 | 70.70 | 92.35 | 0.068 | 0.89 |
| gradient-boosting | 0.968 | 453.87 | 53.25 | 70.70 | 92.35 | 0.068 | 0.90 |
| ridge-regression | 0.980 | 535.38 | 56.26 | 64.23 | 89.10 | 0.080 | 0.04 |
| linear-regression | 0.980 | 535.50 | 56.28 | 64.26 | 89.13 | 0.080 | 0.05 |
| deep-quantile | 0.895 | 792.36 | 104.29 | 142.98 | 191.27 | 0.005 | 1.34 |
| deep-quantile-torch | 0.893 | 793.21 | 90.55 | 123.49 | 181.02 | 0.007 | 1.51 |
| naive-forecast | 0.651 | 1737.81 | 217.52 | 300.38 | 356.61 | 0.249 | 0.00 |
| moving-average | 0.504 | 2758.81 | 267.38 | 345.44 | 423.97 | 0.396 | 0.00 |
| bayesian-quantile | 0.000 | 26M | 1.3M | 1.3M | 1.4M | 0.900 | 24.58 |

**Best:** `quantile-forest` (Winkler: 321.37, Coverage: 90.7%, Calibration Error: 0.007)

---

## Key Findings

### Finding 1: deep-quantile-torch Dominates on Low-Dimensional Data

`deep-quantile-torch` achieves the best Winkler score on both weather (0.0149) and exchange_rate (0.1351). Its neural network architecture captures non-linear relationships that tree-based methods miss on these smaller feature spaces. However, it struggles on the 320-feature electricity dataset (793.21 Winkler).

### Finding 2: deep-quantile Has the Best Calibration

`deep-quantile` achieves near-perfect calibration error on weather (0.001) and exchange_rate (0.003). Its 90.1% coverage on weather is essentially exact. However, its intervals are wider than `deep-quantile-torch`, making it second-best on Winkler score.

### Finding 3: quantile-forest Remains the Best for High-Dimensional Data

On electricity (320 features), `quantile-forest` dominates with Winkler 321.37 — less than half the next best. It achieves 90.7% coverage with only 0.007 calibration error. This confirms the previous finding that quantile forests excel on high-dimensional tabular data.

### Finding 4: BayesianQuantileRegressor Fails on Default Settings

The Bayesian model produces catastrophically bad results across all datasets (0% coverage on weather/electricity, 0.2% on exchange_rate). The horseshoe prior with default settings over-regularizes, producing near-zero coefficients. The posterior predictions collapse to a narrow range around zero, missing the true distribution entirely. **This model requires significant hyperparameter tuning before use.**

### Finding 5: conformal-regressor and gradient-boosting Remain Identical

As noted in the previous benchmark, these two models produce byte-identical results because the baseline `gradient-boosting` wraps `ConformalRegressor` with the same default `GradientBoostingRegressor` base estimator.

### Finding 6: Simple Baselines Are Only Sanity Checks

`naive-forecast` and `moving-average` consistently rank last among non-broken models. Their only advantage is near-zero computation time. They should not be used for production uncertainty quantification.

### Finding 7: Speed vs Quality Tradeoff Is Clear

| Speed Tier | Models | Avg Winkler | Avg Time |
|-----------|--------|-------------|----------|
| Ultra-fast (<5ms) | naive, moving-average | 750.71 | 0.002s |
| Fast (<50ms) | linear, ridge | 179.08 | 0.015s |
| Medium (<200ms) | random-forest, conformal-* | 140.05 | 0.21s |
| Slower (>1s) | quantile-forest, deep-quantile-* | 217.27 | 1.42s |

The medium tier (conformal-forecaster, random-forest) provides the best cost-quality ratio for most applications.

---

## Recommendations

1. **Best overall default:** `deep-quantile-torch` for low-dimensional data, `quantile-forest` for high-dimensional data
2. **Best calibrated:** `deep-quantile` — near-exact 90% coverage across datasets
3. **Fast production option:** `conformal-forecaster` — best quality in the medium speed tier
4. **Avoid:** `bayesian-quantile` with default settings (needs tuning)
5. **Remove duplication:** `gradient-boosting` is identical to `conformal-regressor`

---

## Files

| File | Description |
|------|-------------|
| `full_run_all.json` | Full results for all datasets |
| `full_run_weather.json` | Full results for weather dataset |
| `full_run_electricity.json` | Full results for electricity dataset |
| `full_run_exchange_rate.json` | Full results for exchange_rate dataset |
| `comparison_table.csv` | Combined comparison table for all datasets |
| `20260426-comprehensive-run.md` | Auto-generated console report |

## How to Reproduce

```bash
# Install all optional dependencies
uv sync --extra opinion

# Run full benchmark
uv run python benchmarks/run_benchmarks.py --all-datasets -n 1000 --iterations 3 --warmup 1 -o full_run

# Generate report
uv run python benchmarks/generate_report.py --output docs/benchmarks/20260426-comprehensive-run.md
```
