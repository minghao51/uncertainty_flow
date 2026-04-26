# Testing Conventions

## Framework

- **Test runner**: pytest 9.0.2+
- **Coverage**: pytest-cov
- **Type checking**: mypy (python_version 3.11)
- **Linting**: ruff

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and auto-markers
├── test_utils.py              # Test utilities (create_test_distribution, etc.)
├── test_exceptions.py         # Exception hierarchy tests
├── test_config.py
├── test_decisions.py
├── test_package_integration.py
├── test_base_quantile.py
├── core/
│   ├── test_base.py
│   ├── test_types.py
│   ├── test_distribution.py
│   └── test_persistence.py
├── models/
│   ├── test_quantile_forest.py
│   ├── test_base_quantile.py
│   ├── test_deep_quantile.py
│   ├── test_deep_quantile_torch.py  # optional (torch)
│   └── test_transformer.py
├── metrics/
│   ├── test_pinball.py
│   ├── test_coverage.py
│   └── test_winkler.py
├── calibration/
│   ├── test_shap.py           # optional (shap)
│   └── test_residual_analysis.py
├── risk/
│   ├── test_risk_functions.py
│   └── test_control.py
├── utils/
│   ├── test_polars_bridge.py
│   ├── test_calibration_report.py
│   └── test_split.py
├── viz/
│   └── test_dashboard.py
├── causal/
│   └── test_estimator.py
├── counterfactual/
│   └── test_counterfactual.py
├── decomposition/
│   └── test_ensemble.py
├── multivariate/
│   ├── test_copula.py
│   └── ...
├── multimodal/
│   └── test_aggregator.py
├── bayesian/
│   └── test_numpyro_model.py  # optional (numpyro, jax)
├── wrappers/
│   ├── test_conformal.py
│   └── test_conformal_ts.py
└── analysis/
    └── test_leverage.py
```

## Fixtures (conftest.py)

### Auto-Markers

`pytest_collection_modifyitems` hook automatically applies markers:
- `unit` vs `integration` based on file path
- `slow` for slow-running test files
- `optional` for files requiring optional dependencies

### Shared Fixtures

```python
@pytest.fixture
def time_series_data():
    """Create extended time series DataFrame for testing (150 rows)."""
    # Returns pl.DataFrame with date, price, volume

@pytest.fixture
def univariate_time_series():
    """Create univariate time series DataFrame (150 rows)."""
    # Returns pl.DataFrame with date, target
```

## Test Utilities (test_utils.py)

```python
create_test_distribution(n_samples, n_targets, quantile_levels)
    # → DistributionPrediction with synthetic quantile data

assert_interval_properties(interval, target_name, min_width, max_width)
    # → Asserts lower <= upper, valid widths, finite values

compute_empirical_coverage(y_true, lower, upper)
    # → float fraction of values within interval

create_bivariate_residuals(n_samples, correlation)
    # → np.ndarray (n_samples, 2) with specified correlation

create_time_series_with_pattern(n, trend, seasonality, noise_std)
    # → pl.DataFrame with trend, seasonality, noise
```

## Testing Patterns

### Exception Testing

```python
def test_raises_error():
    with pytest.raises(ModelNotFittedError) as exc_info:
        error_model_not_fitted("TestModel")
    assert "UF-E002" in str(exc_info.value)
```

### Metric Testing

```python
def test_numpy_arrays():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.5])
    loss = pinball_loss(y_true, y_pred, 0.5)
    assert loss == 0.25
```

### Distribution Testing

```python
def test_quantile_matrix_shape():
    pred = create_test_distribution(n_samples=100, n_targets=2)
    assert pred.quantile_matrix.shape == (100, 2 * len(pred.quantile_levels))
```

### Polars Input Testing

Tests handle both DataFrame and LazyFrame via `materialize_lazyframe`.

## Pytest Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--strict-markers", "-ra", "--durations=10", "--import-mode=importlib"]
```

### Markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, focused tests for a single module |
| `integration` | Multi-module workflows |
| `slow` | Expensive tests to exclude during iteration |
| `optional` | Tests requiring optional dependency stacks |
| `network` | Tests requiring internet access |
| `smoke` | Critical path tests for CI gating |

### Filtered Warnings

```toml
filterwarnings = [
    "ignore:Stochastic Optimizer",
    "ignore::uncertainty_flow.utils.exceptions.UncertaintyFlowWarning",
    # ... several numeric warnings ignored
]
```

## Coverage

Run with: `pytest --cov=uncertainty_flow --cov-report=html`

Target: core modules, metrics, and utilities should have high coverage.
Optional dependency modules (torch, numpyro, shap) are exccluded from coverage requirements.

## Mocking

No mocking framework is heavily used. Tests primarily use:
- Synthetic data fixtures
- `create_test_distribution()` utility
- Subpackage test data (e.g., small CSV datasets)
