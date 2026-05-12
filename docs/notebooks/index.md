# Interactive Notebooks

Quarto `.qmd` notebooks are the **source of truth** for all examples. Each notebook
is rendered to static HTML via `quarto render` with `freeze: auto` caching.

| Notebook | Description |
|----------|-------------|
| [Quick Start: End-to-End Workflow](01-quick-start.md) | Complete lifecycle — conformal regression, classification, DistributionPrediction API, persistence, and validation strategies |
| [Time Series Forecasting with Uncertainty](02-time-series.md) | Compare ConformalForecaster, QuantileForestForecaster, and AdaptiveConformalForecaster on real weather data |
| [Diagnostics & Calibration](03-diagnostics-calibration.md) | Understand and improve your predictions — uncertainty decomposition, PIT diagnostics, recalibration, and SHAP attribution |
| [Risk & Multivariate Uncertainty](04-risk-copulas.md) | Risk-aware decision making with conformal risk control, copula-based multivariate modeling, and joint distribution analysis |
| [Methods & Benchmarks](05-benchmarks.md) | Every model in uncertainty_flow, when to use it, how they compare, and how to run your own benchmarks |

## Run Locally

```bash
uv sync --extra opinion
make notebooks
```
