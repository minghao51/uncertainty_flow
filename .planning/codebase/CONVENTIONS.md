# Code Conventions

## Style & Formatting

- **Tool**: ruff (E, F, I, N, W rules)
- **Line length**: 100 characters
- **Python**: 3.11+
- **Imports**: absolute within package (`from uncertainty_flow.utils...`)
- **Future imports**: `from __future__ import annotations` at top of all files

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `DistributionPrediction`, `BaseUncertaintyModel` |
| Functions/methods | snake_case | `calibration_report`, `warn_quantile_crossing` |
| Constants | SCREAMING_SNAKE | `MAX_SAMPLE_CHUNK_SIZE`, `PLOT_MAX_SAMPLES` |
| Private attrs | underscore prefix | `_fitted`, `_metadata` |
| Error codes | UF-EXXX | `UF-E001`, `UF-E002` |
| Warning codes | UF-WXXX | `UF-W001`, `UF-W002` |

## Project Structure

```
uncertainty_flow/
├── core/           # Base classes, types, distribution output
├── decisions/      # Decision strategies
├── models/         # Quantile models (forest, deep, transformer)
├── metrics/        # Evaluation metrics (pinball, coverage, winkler)
├── calibration/    # Calibration utilities
├── risk/           # Risk functions
├── utils/          # Shared utilities
├── viz/            # Dashboard visualization
├── causal/         # Causal estimation
├── counterfactual/ # Counterfactual explanations
├── decomposition/  # Ensemble decomposition
├── multivariate/   # Copula modeling
├── multimodal/     # Multi-modal aggregation
├── bayesian/       # Bayesian models (numpyro)
└── benchmarking/   # Benchmarking utilities
```

## Error Handling

### Exception Hierarchy (`uncertainty_flow/utils/exceptions.py`)

```
UncertaintyFlowError (ValueError)
├── ModelError
│   └── ModelNotFittedError
├── DataError
│   └── InvalidDataError
├── CalibrationError
│   └── CalibrationSizeError
└── ConfigurationError
    └── QuantileError
```

### Error Helpers

Functions raise specific exceptions:
- `error_model_not_fitted(model_name)` → `ModelNotFittedError`
- `error_invalid_data(reason)` → `InvalidDataError`
- `error_calibration_too_small(n_samples, min_size)` → `CalibrationSizeError`
- `error_quantile_invalid(reason)` → `QuantileError`

All errors include error codes (e.g., `UF-E002`) for programmatic handling.

### Warnings

Custom warnings in `exceptions.py`:
- `UncertaintyFlowWarning` (UserWarning subclass)
- `warn_calibration_size()` - UF-W001
- `warn_quantile_crossing()` - UF-W002
- `warn_coverage_gap()` - UF-W003
- `warn_no_uncertainty_drivers()` - UF-W004
- `warn_lazyframe_materialized()` - UF-W005
- `warn_copula_auto_selection_ndim()` - UF-W006

## Code Patterns

### Base Model Interface

All uncertainty models inherit `BaseUncertaintyModel` (abstract):
- `fit(data, target, **kwargs)` → returns self
- `predict(data)` → `DistributionPrediction`
- `calibration_report(data, target, quantile_levels)` → `pl.DataFrame`
- `save(path)` / `load(cls, path)` → .uf archive
- `metadata` property
- `uncertainty_drivers_` property

### DistributionPrediction

Core output object storing:
- `quantile_matrix`: NumPy array (n_samples, n_targets * n_quantiles)
- `quantile_levels`: list of quantile levels
- `target_names`: list of target names
- Optional: `posterior`, `group_predictions`, `copula`

### Input Handling

- Accept `PolarsInput` = `pl.DataFrame | pl.LazyFrame`
- Use `materialize_lazyframe()` from `utils.polars_bridge` to normalize
- Validate with error helpers on construction

### Decision Strategies

Implement `DecisionStrategy` ABC:
- `decide(distribution)` → `DecisionResult`
- `DecisionResult` is a dataclass with `optimal_value`, `strategy`, `metadata`

## Polars Conventions

- Use `pl.DataFrame` / `pl.LazyFrame` for data exchange
- Internally convert to NumPy for efficiency via `to_numpy_series_zero_copy`
- Avoid `collect()` calls until necessary
- Use Polars expression API for transformations

## Testing Markers

Automatic markers via `conftest.py` pytest hook:
- `unit`: fast, single-module tests
- `integration`: multi-module workflows
- `slow`: expensive tests (deep quantile torch, numpyro, conformal_ts, counterfactual, persistence)
- `optional`: tests requiring optional deps (torch, numpyro)
- `network`: tests requiring internet
- `smoke`: critical path for CI
