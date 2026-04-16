# Architecture Overview

This document is the source of truth for how the package is organized and how predictions flow through the system.

## Package Structure

```text
uncertainty_flow/
├── analysis/                  # Feature leverage analysis for uncertainty attribution
├── bayesian/                  # Bayesian quantile regression (NumPyro, optional)
├── benchmarking/              # Benchmark datasets, runner, auto-tuning
├── calibration/               # Calibration reports, residual analysis, SHAP helpers
├── causal/                    # Treatment effect estimation with conformal uncertainty
├── core/                      # Base classes, config, DistributionPrediction, persistence, shared types
├── counterfactual/            # Counterfactual explanations for uncertainty reduction
├── decomposition/             # Ensemble-based aleatoric/epistemic decomposition
├── metrics/                   # Pinball, Winkler, coverage
├── models/                    # Native models (QuantileForest, DeepQuantile, Transformer)
├── multimodal/                # Cross-feature group uncertainty aggregation
├── multivariate/              # Copula families and multivariate support
├── risk/                      # Conformal risk control with user-defined risk functions
├── utils/                     # Polars bridge, split strategies, exceptions, helpers
├── viz/                       # Interactive Streamlit dashboard (optional)
├── wrappers/                  # Conformal wrappers around estimators
└── cli.py                     # Project CLI entrypoint
```

## Model Families

`uncertainty_flow` has multiple model families:

- **Wrappers:** `ConformalRegressor` and `ConformalForecaster` adapt existing estimators and add calibrated uncertainty.
- **Native models:** `QuantileForestForecaster`, `DeepQuantileNet`, optional `DeepQuantileNetTorch`, and optional `TransformerForecaster`.
- **Bayesian:** `BayesianQuantileRegressor` provides posterior inference via NumPyro MCMC (optional dependency).
- **Causal:** `CausalUncertaintyEstimator` estimates treatment effects with conformal uncertainty (doubly robust, S-learner, T-learner).
- **Multi-modal:** `CrossModalAggregator` combines predictions from separate feature groups using product, copula, or independent aggregation.
- **Risk control:** `ConformalRiskControl` wraps conformal prediction around user-defined risk functions (asymmetric loss, inventory cost, VaR, threshold penalty).
- **Counterfactual:** `UncertaintyExplainer` finds minimal feature changes to reduce prediction interval width.
- **Analysis:** `FeatureLeverageAnalyzer` scores features by their impact on interval width (aleatoric vs epistemic).
- **Decomposition:** `EnsembleDecomposition` separates aleatoric from epistemic uncertainty via bootstrap refits.
- **Visualization:** `launch_dashboard()` starts an interactive Streamlit dashboard for calibration exploration (optional dependency).

All models return a `DistributionPrediction` object so downstream code can use the same access pattern regardless of the training backend.

## Data Flow

```text
Polars DataFrame / LazyFrame
        |
        v
utils/polars_bridge.py
  - validate columns
  - materialize LazyFrame only when needed
  - convert to NumPy for computation
        |
        v
BaseUncertaintyModel / model-specific fit-predict logic
        |
        v
Optional calibration + multivariate dependence modeling
        |
        v
DistributionPrediction
  - quantile()
  - interval()
  - mean()
  - sample()
  - plot()
        |
        v
Polars outputs for callers
```

## Core Contracts

### `BaseUncertaintyModel`

All public models implement a common `fit()` / `predict()` interface and expose calibration helpers through the same base contract.

### `DistributionPrediction`

`DistributionPrediction` is the unifying output layer. It stores quantile predictions internally and exposes:

- `quantile()` for extracting one or more quantile columns
- `interval()` for symmetric prediction intervals
- `mean()` for the median-style point estimate
- `sample()` for Monte Carlo-style downstream simulation
- `plot()` for visual inspection of predictive spread

This distribution-first contract replaces model-specific output types and keeps downstream consumers simple.

### Polars Boundary

The Polars to NumPy seam is intentionally centralized in `utils/polars_bridge.py`. User-facing APIs remain Polars-native, while internal compute stays compatible with NumPy, SciPy, and scikit-learn style backends.

## Calibration and Splits

The package supports distinct split strategies depending on the problem shape:

- `RandomHoldoutSplit` for i.i.d. tabular problems
- `TemporalHoldoutSplit` for ordered forecasting data
- cross-conformal style workflows where supported

Small calibration sets are guarded explicitly so models fail loudly when interval estimates would be unreliable.

## Multivariate Uncertainty

For multi-target forecasting, the package combines per-target marginals with a copula layer to approximate joint behavior. The current multivariate module supports:

- `GaussianCopula` for general correlation structure
- `ClaytonCopula` for lower-tail dependence
- `GumbelCopula` for upper-tail dependence
- `FrankCopula` for symmetric non-tail dependence
- auto-selection paths where the model supports choosing the family programmatically

That separation keeps marginal prediction logic independent from joint dependence modeling.

## Benchmarking Layer

The `benchmarking/` package and CLI support:

- loading benchmark datasets
- running comparable model evaluations
- optional parameter tuning
- exporting structured benchmark results

Those tools are intentionally separate from the model APIs so the library can stay usable both as a package and as an evaluation harness.

## Analysis, Decomposition, and Risk Layer

Three post-hoc analysis modules sit between prediction and decision-making:

- **FeatureLeverageAnalyzer** scores features by how much they influence interval width, separating aleatoric (irreducible) from epistemic (model-knowledge) contributions. Multivariate mode reports per-target scores.
- **EnsembleDecomposition** refits a model on bootstrap samples and decomposes total interval width into aleatoric (average width) and epistemic (variance across refits).
- **ConformalRiskControl** accepts a user-defined risk function and calibrates prediction intervals to control expected risk rather than just coverage.

These modules consume a fitted model and optionally test data. They do not modify the model itself.

## Causal and Counterfactual Modules

- **CausalUncertaintyEstimator** wraps outcome and propensity models to produce treatment effect estimates (CATE) with conformal confidence intervals. Supports doubly-robust, S-learner, and T-learner methods.
- **UncertaintyExplainer** searches for minimal feature perturbations that reduce interval width, using evolutionary search for tree models and gradient descent for differentiable models.
