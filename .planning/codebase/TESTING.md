# Testing

## Overview
Uncertainty Flow uses pytest for testing with a focus on unit tests, fixture-based test data, and coverage targets. The test suite mirrors the package structure and emphasizes reproducibility.

## Testing Framework

### Tool: pytest
**Primary testing framework**: pytest

**Configuration** (from `pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

**Version**: pytest >=7.4.0 (dev), >=9.0.2 (main dependencies)

**Note**: Version conflict in pyproject.toml should be resolved

## Test Structure

### Directory Layout
```
tests/
├── conftest.py              # Shared fixtures
├── core/                    # Core module tests
│   ├── test_base.py
│   ├── test_distribution.py
│   └── test_types.py
├── models/                  # Model tests
│   ├── test_deep_quantile.py
│   └── test_deep_quantile_torch.py
├── metrics/                 # Metrics tests
│   ├── test_pinball.py
│   ├── test_coverage.py
│   └── test_winkler.py
├── utils/                   # Utilities tests
│   ├── test_polars_bridge.py
│   └── test_split.py
├── wrappers/                # Wrapper tests
└── calibration/             # Calibration tests
```

**Pattern**: Test structure mirrors source structure

## Fixtures

### Shared Fixtures (tests/conftest.py)

#### sample_dataframe
```python
@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "target": [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5],
    })
```

#### sample_time_series
```python
@pytest.fixture
def sample_time_series():
    """Create a sample time series DataFrame for testing."""
    return pl.DataFrame({
        "date": range(20),
        "price": [10 + i + 0.5 * np.sin(i / 3) for i in range(20)],
        "volume": [100 + 10 * i + 5 * np.cos(i / 2) for i in range(20)],
    })
```

#### random_state
```python
@pytest.fixture
def random_state():
    """Default random state for reproducibility."""
    return 42
```

### Fixture Usage
**Pattern**: Use fixtures instead of creating test data inline

```python
def test_fit_with_sample_data(sample_dataframe):
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")
    assert model.is_fitted
```

## Test Patterns

### Unit Tests

#### Model Testing
```python
def test_fit_returns_self(sample_dataframe):
    """Test that fit() returns self for method chaining."""
    model = DeepQuantileNet()
    result = model.fit(sample_dataframe, target="target")
    assert result is model

def test_predict_returns_distribution(sample_dataframe):
    """Test that predict() returns DistributionPrediction."""
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")
    pred = model.predict(sample_dataframe)
    assert isinstance(pred, DistributionPrediction)
```

#### Metric Testing
```python
def test_pinball_loss_scalar():
    """Test pinball loss with scalar input."""
    loss = pinball_loss(y_true=0.5, y_pred=0.3, quantile=0.5)
    assert loss == 0.2  # |0.5 - 0.3| * 0.5

def test_coverage_score_perfect():
    """Test coverage score with perfect predictions."""
    score = coverage_score(
        y_true=np.array([1, 2, 3]),
        y_pred_lower=np.array([0.5, 1.5, 2.5]),
        y_pred_upper=np.array([1.5, 2.5, 3.5]),
    )
    assert score == 1.0
```

### Integration Tests

#### End-to-End Testing
```python
def test_full_workflow(sample_dataframe):
    """Test complete fit-predict-evaluate workflow."""
    # Split data
    split = RandomHoldoutSplit(calibration_size=0.2)
    train, calib = split.split(sample_dataframe, calibration_size=0.2)

    # Fit model
    model = ConformalRegressor(base_model=GradientBoostingRegressor())
    model.fit(train, target="target")

    # Predict
    pred = model.predict(calib)

    # Evaluate
    report = model.calibration_report(calib, target="target")
    assert isinstance(report, pl.DataFrame)
```

### Property-Based Testing

#### Invariant Testing
```python
def test_quantile_monotonicity(sample_dataframe):
    """Test that quantiles are monotonically increasing."""
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")
    pred = model.predict(sample_dataframe)

    quantiles = pred.quantile([0.1, 0.5, 0.9])
    for col in quantiles.columns:
        assert (quantiles[col].diff() >= 0).all()
```

## Test Data Generation

### Reproducibility
**Always set random seed**:

```python
def test_with_random_data():
    np.random.seed(42)
    data = pl.DataFrame({
        "x": np.random.randn(100),
        "y": np.random.randn(100),
    })
    # ... test logic ...
```

### Synthetic Data
**Pattern**: Create synthetic data for edge cases

```python
def test_with_constant_target():
    """Test behavior when target is constant."""
    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [5, 5, 5, 5, 5],  # Constant
    })
    model = DeepQuantileNet()
    model.fit(df, target="y")
    # Should handle gracefully
```

## Coverage

### Targets
- **Core logic**: >90% coverage
- **Models**: >80% coverage
- **Utilities**: >70% coverage

### Running Coverage

```bash
# Full coverage report
pytest --cov=uncertainty_flow --cov-report=html

# Terminal summary
pytest --cov=uncertainty_flow --cov-report=term-missing

# Minimum coverage threshold
pytest --cov=uncertainty_flow --cov-fail-under=80
```

### Coverage Report
**HTML report**: `htmlcov/index.html`

**Key metrics**:
- Line coverage percentage
- Missing lines by file
- Branch coverage (if configured)

## Test Execution

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/models/test_deep_quantile.py

# Run specific test
pytest tests/models/test_deep_quantile.py::test_fit_returns_self

# Run with coverage
pytest --cov=uncertainty_flow

# Stop on first failure
pytest -x

# Run failed tests only
pytest --lf
```

### Parallel Execution
**Not currently configured** - can be added with pytest-xdist

```bash
# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Mocking

### Current State
**Limited mocking** in current test suite

### Recommended Use Cases
- External API calls (none currently)
- File I/O operations
- Time-dependent tests
- Expensive computations

### Example Pattern
```python
from unittest.mock import patch, MagicMock

def test_with_mocked_model():
    with patch('uncertainty_flow.wrappers.conformal.GradientBoostingRegressor') as mock:
        mock.return_value = MagicMock()
        # ... test logic ...
```

## Parametrization

### Pattern
**Use pytest.mark.parametrize** for data-driven tests:

```python
@pytest.mark.parametrize("quantile,expected_range", [
    (0.1, (0.0, 2.0)),
    (0.5, (1.0, 3.0)),
    (0.9, (2.0, 4.0)),
])
def test_quantile_ranges(quantile, expected_range, sample_dataframe):
    """Test that quantiles fall within expected ranges."""
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")
    pred = model.predict(sample_dataframe)
    q = pred.quantile([quantile])
    # ... assertions ...
```

## Edge Cases to Test

### Input Validation
- Empty DataFrames
- Single-row DataFrames
- Missing values (null)
- Invalid column names
- Wrong data types

### Numerical Edge Cases
- All zeros
- All same values
- Very large values
- Very small values
- NaN/Inf values

### Model Behavior
- Predict before fit
- Fit with empty data
- Calibration set too small
- Invalid quantile levels

## Test Organization

### Test Classes
**Group related tests**:

```python
class TestDeepQuantileNet:
    def test_fit(self, sample_dataframe):
        ...

    def test_predict(self, sample_dataframe):
        ...

    def test_calibration_report(self, sample_dataframe):
        ...
```

### Test Naming
**Pattern**: `test_<function>_<scenario>`

```python
test_fit_returns_self()
test_fit_with_empty_dataframe()
test_fit_with_missing_values()
test_predict_returns_distribution()
```

## Continuous Integration

### Recommended CI Setup
**Not currently configured** - but should include:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv sync
    uv run pytest --cov=uncertainty_flow --cov-fail-under=80

- name: Upload coverage
  run: |
    uv run pytest --cov=uncertainty_flow --cov-report=xml
    # Upload to codecov or similar
```

## Performance Testing

### Benchmarking
**Located in**: `scripts/benchmark.py`

**Purpose**: Performance regression testing

**Usage**:
```bash
uv run python scripts/benchmark.py
```

## Debugging Tests

### pdb Integration
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on error
pytest --pdb --trace
```

### Print Debugging
```python
def test_with_debugging(sample_dataframe):
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")

    # Debug: print prediction structure
    pred = model.predict(sample_dataframe)
    print(pred)  # Will show in pytest output
```

## Test Maintenance

### Keeping Tests Updated
1. **When adding features**: Add corresponding tests
2. **When fixing bugs**: Add regression tests
3. **When refactoring**: Update tests to match new structure
4. **Regular review**: Remove outdated or redundant tests

### Test Smells
- **Fragile tests**: Break with unrelated changes
- **Slow tests**: Take too long to run
- **Complex tests**: Hard to understand
- **Duplicated logic**: Same setup in multiple tests

## Best Practices

### DO
- Use fixtures for shared test data
- Set random seeds for reproducibility
- Test edge cases and error conditions
- Keep tests independent
- Use descriptive test names
- Test behavior, not implementation

### DON'T
- Don't hardcode test data in tests
- Don't skip tests without good reason
- Don't make tests depend on each other
- Don't test private methods directly
- Don't ignore failing tests

## Current Test Coverage Gaps

### Missing Tests
- `wrappers/` directory - No tests yet
- `calibration/` directory - No tests yet
- Multivariate functionality - Limited coverage
- Edge cases - Insufficient coverage

### Priority Areas for More Tests
1. Conformal prediction wrappers
2. Calibration reports
3. Residual analysis
4. Time series forecasting
5. Error handling paths
