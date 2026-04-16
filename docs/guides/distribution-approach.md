# Distribution-First Concepts

`uncertainty_flow` is built around one choice: `predict()` returns a `DistributionPrediction` object, not a point estimate. Every model family — conformal wrappers, tree models, neural nets, Bayesian, causal — converges to the same downstream contract.

## The Two Workflows

**Tabular regression:** Wrap an existing sklearn estimator with `ConformalRegressor` for calibrated intervals on unordered feature tables.

**Forecasting:** Use `ConformalForecaster`, `QuantileForestForecaster`, or `TransformerForecaster` for ordered time series with temporal splits and multivariate joint uncertainty.

## What DistributionPrediction Gives You

Every `predict()` call returns the same interface: `quantile()`, `interval()`, `mean()`, `sample()`, `plot()`. This means model switching is cheap, evaluation code is reusable, and downstream consumers ask for intervals, samples, or quantiles the same way every time.

## Beyond Core Prediction

Once you have a fitted model, additional modules build on the same surface:

- **Calibration:** `.calibration_report()` checks whether stated confidence matches empirical coverage. See [./calibration.md](./calibration.md).
- **Decomposition:** `EnsembleDecomposition` separates aleatoric from epistemic uncertainty via bootstrap refits. See [./models.md](./models.md).
- **Feature leverage:** `FeatureLeverageAnalyzer` scores features by their impact on interval width. See [./models.md](./models.md).
- **Risk control:** `ConformalRiskControl` calibrates intervals to control expected loss rather than coverage. See [./models.md](./models.md).
- **Counterfactuals:** `UncertaintyExplainer` finds minimal feature changes that reduce uncertainty. See [./models.md](./models.md).

## Where To Go Next

- [../architecture/overview.md](../architecture/overview.md) — package structure and data flow
- [./models.md](./models.md) — model selection, guarantee tradeoffs, and choosing quickly
- [./calibration.md](./calibration.md) — interpreting calibration reports and miscalibration response
- [./charting.md](./charting.md) — plotting, intervals, samples, and visual inspection
- [../api/spec.md](../api/spec.md) — full API specification
