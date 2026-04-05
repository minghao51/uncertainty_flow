# Testing Documentation

## Test Framework and Config

### Framework
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **Python 3.11+**: Target test version

### Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
filterwarnings = [
    "ignore:Stochastic Optimizer",
    "ignore:Degrees of freedom <= 0 for slice:RuntimeWarning",
    "ignore:divide by zero encountered in divide:RuntimeWarning",
    "ignore:invalid value encountered in multiply:RuntimeWarning",
    "ignore:invalid value encountered in log:RuntimeWarning",
    "ignore:An input array is constant",
    "ignore::uncertainty_flow.utils.exceptions.UncertaintyFlowWarning",
]
```

## Test Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_package_integration.py # Package-level tests
├── test_base_quantile.py       # Base functionality tests
├── core/
│   ├── test_base.py           # Base model tests
│   ├── test_distribution.py   # Distribution tests
│   └── test_types.py          # Type system tests
├── utils/
│   ├── test_split.py          # Split strategy tests
│   └── test_polars_bridge.py  # Polars integration tests
├── metrics/
│   ├── test_coverage.py       # Coverage score tests
│   ├── test_pinball.py        # Pinball loss tests
│   └── test_winkler.py        # Winkler score tests
├── wrappers/
│   ├── test_conformal.py      # Conformal wrapping tests
│   └── test_conformal_ts.py   # Time series conformal tests
├── calibration/
│   ├── test_residual_analysis.py # Calibration analysis tests
│   └── test_shap.py          # SHAP-based calibration tests
├── causal/
│   └── test_estimator.py      # Causal estimation tests
├── multivariate/
│   └── test_copula.py        # Copula model tests
└── models/
    # Additional model tests...
```

## Test File Naming Patterns

- Test files must start with `test_` prefix
- Test classes must start with `Test` prefix
- Test functions must start with `test_` prefix
- Mirror source directory structure in test directory

Examples:
- `test_core.py` → tests core module
- `test_split.py` → tests utils.split module
- `TestRandomHoldoutSplit` → tests class in utils.split

## Test Patterns

### Shared Fixtures (conftest.py)
- `sample_dataframe`: Basic DataFrame for testing
- `sample_time_series`: Time series DataFrame with patterns
- `random_state`: Reproducible random seed (42)

### Common Test Patterns

#### 1. Error Testing
```python
def test_raises_error_invalid_data(self):
    """Test that invalid input raises proper error."""
    with pytest.raises(ValueError, match="Invalid data"):
        function(invalid_input)
```

#### 2. Warning Testing
```python
def test_warns_small_calibration(self):
    """Test that warning is issued for small calibration set."""
    with pytest.warns(UserWarning, match="Calibration set has only"):
        splitter.split(df, 0.4)  # 40 samples
```

#### 3. Parametrized Tests
```python
@pytest.mark.parametrize("quantile_levels", [[0.5], [0.8, 0.9], [0.1, 0.5, 0.9]])
def test_quantile_levels(quantile_levels):
    """Test function with multiple quantile levels."""
    result = function(quantile_levels)
    assert isinstance(result, float)
```

#### 4. Abstract Class Testing
```python
def test_cannot_instantiate_base_class(self):
    """Base classes should not be instantiable."""
    with pytest.raises(TypeError):
        BaseUncertaintyModel()  # type: ignore
```

#### 5. Type Checking Tests
```python
def test_return_types(self):
    """Test that function returns correct types."""
    result = function(input_data)
    assert isinstance(result, pl.DataFrame)
```

#### 6. Reproducibility Tests
```python
def test_reproducibility_with_random_state(self):
    """Test same random state produces same results."""
    result1 = function(random_state=42)
    result2 = function(random_state=42)
    assert result1 == result2
```

### Mocking Strategy
- Use minimal mocking - prefer real implementations
- Use `unittest.mock` for external dependencies
- Test edge cases with mocked data

### Test Data
- Use Polars DataFrames consistently
- Small, predictable datasets
- Include edge cases (empty data, single row, etc.)
- Use fixed random states for reproducibility

## Coverage Configuration

### Coverage Tool
- **pytest-cov**: Coverage reporting
- **Coverage Threshold**: Not explicitly set in config
- **Branch Coverage**: Enabled

### Running Tests with Coverage
```bash
# Run tests with coverage
pytest --cov=uncertainty_flow --cov-report=html

# Run tests with coverage report
pytest --cov=uncertainty_flow --cov-report=term-missing

# Run specific test with coverage
pytest tests/core/test_base.py --cov=uncertainty_flow.core.base
```

### Coverage Goals
- Aim for 80%+ coverage for core modules
- 90%+ coverage for critical paths
- 100% coverage for utilities

## How to Run Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with stop on first failure
pytest -x

# Run tests with specific markers
pytest -m "not slow"
```

### Running Specific Tests
```bash
# Run specific test file
pytest tests/core/test_base.py

# Run specific test function
pytest tests/core/test_base.py::TestBaseUncertaintyModel::test_cannot_instantiate_base_class

# Run specific test class
pytest tests/core/test_base.py::TestBaseUncertaintyModel
```

### Running with Options
```bash
# Run tests with coverage
pytest --cov=uncertainty_flow --cov-report=term-missing

# Run tests in parallel
pytest -n auto

# Run tests with custom Python path
PYTHONPATH=. pytest

# Run tests without capturing output
pytest -s
```

### Development Workflow
1. Write tests before implementation (TDD)
2. Run tests frequently during development
3. Ensure all tests pass before commit
4. Add tests for new features and bug fixes
5. Maintain test coverage standards

### CI/CD Integration
- Tests run on Python 3.11+
- Run all tests on each commit
- Coverage report generated in CI
- Fail build if tests fail
