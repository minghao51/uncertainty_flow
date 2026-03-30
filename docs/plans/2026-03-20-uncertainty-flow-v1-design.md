# uncertainty_flow v1 Implementation Design

**Date:** 2026-03-20
**Status:** Approved
**Approach:** Full v1 Foundation, Bottom-up Layers, Strict TDD

---

## Overview

Implement the `uncertainty_flow` library from scratch - a probabilistic forecasting and uncertainty quantification package with a distribution-first API. All models return a `DistributionPrediction` object instead of point predictions.

---

## Project Structure

```
uncertainty_flow/
├── pyproject.toml          # uv-managed dependencies
├── tests/                  # Test directory (mirrors package)
│   ├── conftest.py         # Shared pytest fixtures
│   ├── core/
│   ├── metrics/
│   ├── utils/
│   ├── calibration/
│   ├── wrappers/
│   └── models/
└── uncertainty_flow/
    ├── __init__.py         # Public API surface
    ├── core/
    │   ├── base.py         # BaseUncertaintyModel ABC
    │   ├── distribution.py # DistributionPrediction
    │   └── types.py        # Type aliases, enums
    ├── metrics/
    │   ├── pinball.py
    │   ├── winkler.py
    │   └── coverage.py
    ├── utils/
    │   ├── polars_bridge.py
    │   ├── split.py
    │   └── warnings.py
    ├── calibration/
    │   ├── report.py
    │   └── residual_analysis.py
    ├── multivariate/
    │   ├── copula.py
    │   └── marginal.py
    ├── wrappers/
    │   ├── conformal.py
    │   └── conformal_ts.py
    └── models/
        └── quantile_forest.py
```

---

## Implementation Order (Bottom-up)

1. **Project Setup**
   - Initialize uv project with pyproject.toml
   - Create directory structure
   - Configure pytest, ruff

2. **Core Layer** (no dependencies)
   - `types.py` - Type aliases, enums, constants
   - `base.py` - BaseUncertaintyModel ABC
   - `distribution.py` - DistributionPrediction class

3. **Metrics Layer** (standalone)
   - `pinball.py` - Quantile loss
   - `winkler.py` - Interval score
   - `coverage.py` - Empirical coverage

4. **Utils Layer** (core infrastructure)
   - `polars_bridge.py` - Polars ↔ NumPy conversion
   - `split.py` - Calibration split strategies
   - `warnings.py` - Standardized error/warning codes

5. **Calibration Layer** (diagnostics)
   - `residual_analysis.py` - Uncertainty driver detection
   - `report.py` - Calibration report generation

6. **Multivariate Layer** (joint intervals)
   - `marginal.py` - Per-target CDFs
   - `copula.py` - Gaussian copula

7. **Models & Wrappers** (concrete implementations)
   - `conformal.py` - ConformalRegressor (tabular)
   - `conformal_ts.py` - ConformalForecaster (time series)
   - `quantile_forest.py` - QuantileForestForecaster

---

## Key Components

### DistributionPrediction
The core output object. Stores quantiles as NumPy arrays internally, exposes Polars interface.

**Methods:**
- `quantile(q)` - Extract specific quantile levels
- `interval(confidence)` - Get prediction interval
- `mean()` - Return median (0.5 quantile)
- `plot(actuals)` - Fan chart visualization

### BaseUncertaintyModel
ABC that all models inherit from. Enforces `fit()` / `predict()` contract.

**Required methods:**
- `fit(data, target)` - Train the model
- `predict(data)` - Return DistributionPrediction

**Provided methods:**
- `calibration_report()` - Generate diagnostics
- `uncertainty_drivers_` - Residual correlation analysis

### Metrics
Three standalone metrics for evaluation:
- `pinball_loss` - Quantile loss
- `winkler_score` - Interval score (width + accuracy)
- `coverage_score` - Fraction of actuals within interval

### Polars Bridge
Single conversion point between Polars and NumPy:
- `to_numpy()` - Convert Polars → NumPy, materializes LazyFrame if needed
- `to_polars()` - Convert NumPy → Polars, preserves index

---

## Testing Strategy

**Strict TDD:** Write tests before implementation for every module.

**Test structure mirrors package:**
```
tests/core/test_types.py
tests/core/test_base.py
tests/core/test_distribution.py
tests/metrics/test_pinball.py
tests/utils/test_polars_bridge.py
...
```

**Per-module workflow:**
1. Write failing tests
2. Run `uv run pytest` - confirm failures
3. Implement to pass tests
4. Run `uv run pytest` - confirm all pass
5. Run `uv run ruff check .` and `uv run ruff format .`
6. Commit

**Shared fixtures** (conftest.py):
- Sample polars DataFrames
- Mock sklearn models
- Random state for reproducibility

---

## Dependencies

**Core:**
- polars >= 0.20.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0

**Dev:**
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- ruff >= 0.1.0
- matplotlib >= 3.7.0 (soft dependency for plotting)

**Python version:** >= 3.11

---

## Design Principles

1. **Distribution-first** - Every model returns DistributionPrediction
2. **Polars I/O, NumPy spine** - User-facing Polars, internal NumPy
3. **Honest guarantees** - Document which models have mathematical vs empirical coverage
4. **Single conversion point** - All Polars ↔ NumPy in polars_bridge.py
5. **Clean errors** - Standardized UF-W/UF-E codes

---

## Non-Goals (v1)

Explicitly out of scope:
- Async/streaming inference
- `.sample()` method on DistributionPrediction
- PyTorch backend
- Quantile SHAP
- Rich copula families (Clayton, Frank)
- Bayesian/MCMC methods

---

## Success Criteria

✅ All tests pass
✅ ruff check passes with no errors
✅ Can run quickstart examples from README
✅ Calibration report generates valid Polars DataFrame
✅ ConformalRegressor wraps any sklearn model
✅ ConformalForecaster handles multivariate time series
✅ QuantileForestForecaster produces non-crossing quantiles
