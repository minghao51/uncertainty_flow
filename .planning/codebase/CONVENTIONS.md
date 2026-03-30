# Code Conventions

## Overview
Uncertainty Flow follows modern Python best practices with Polars-first data handling, strong typing, and consistent code patterns throughout the codebase.

## Code Style

### Linting & Formatting
**Tool**: Ruff (fast Python linter/formatter)

**Configuration** (from `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]  # Errors, Pyflakes, Import sorting, Naming, Warnings
ignore = []
```

**Style Guide**:
- **Line length**: 100 characters (soft limit)
- **Target Python**: 3.11+
- **Import sorting**: Automatic (isort-style)
- **Naming**: PEP 8 compliant

### Import Conventions

#### Import Order
1. Standard library imports
2. Third-party imports (polars, numpy, sklearn, scipy)
3. Local imports (uncertainty_flow modules)

#### Import Style
```python
# Standard library
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Third-party
import polars as pl
import numpy as np
from sklearn.base import RegressorMixin

# Local
from ..core.base import BaseUncertaintyModel
```

#### TYPE_CHECKING Pattern
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distribution import DistributionPrediction
```

**Purpose**: Avoid circular imports while maintaining type hints

#### Relative Imports
- Use relative imports within package: `from ..core import base`
- Absolute imports for external packages: `import polars as pl`

### Naming Conventions

#### Classes
**Pattern**: `PascalCase` with descriptive names

```python
class BaseUncertaintyModel:
class DistributionPrediction:
class DeepQuantileNet:
class ConformalRegressor:
```

**Base Classes**: Prefix with `Base`
- `BaseUncertaintyModel`
- `BaseSplit`

#### Functions & Methods
**Pattern**: `snake_case` with descriptive verbs

```python
def calibration_report():
def fit(data, target):
def predict(data):
def pinball_loss(y_true, y_pred):
```

**Private Methods**: Prefix with `_`
```python
def _validate_calibration_size():
def _extract_trunk_features():
```

#### Variables
**Pattern**: `snake_case` for local variables

```python
quantile_matrix = ...
n_samples = ...
calibration_data = ...
```

**Constants**: `UPPER_SNAKE_CASE`
```python
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
MIN_CALIBRATION_SIZE = 20
```

#### Type Parameters
**Pattern**: Descriptive names with `_t` suffix or full words

```python
PolarsInput: PolarsInput  # Type alias
TargetSpec: TargetSpec    # Type alias
```

### Type Hints

#### Mandatory Type Hints
**All public functions and methods** must have type hints:

```python
def fit(
    self,
    data: PolarsInput,
    target: TargetSpec,
    **kwargs,
) -> "BaseUncertaintyModel":
    ...
```

#### Type Aliases
**Defined in `core/types.py`**:
```python
PolarsInput = pl.DataFrame | pl.LazyFrame
TargetSpec = str | list[str]
```

#### Return Types
**Always explicit**:
```python
def predict(self, data: PolarsInput) -> "DistributionPrediction":
def calibration_report(...) -> pl.DataFrame:
def _validate(...) -> None:
```

#### Optional Types
**Use `| None` syntax** (Python 3.11+):
```python
quantile_levels: list[float] | None = None
```

### Docstring Conventions

#### Format
**Google-style docstrings** (preferred):

```python
def calibration_report(
    self,
    data: PolarsInput,
    target: TargetSpec,
    quantile_levels: list[float] | None = None,
) -> pl.DataFrame:
    """
    Generate calibration diagnostics.

    Default implementation - can be overridden by subclasses.

    Args:
        data: Validation data
        target: Target column name(s)
        quantile_levels: Quantile levels to evaluate (default: [0.8, 0.9, 0.95])

    Returns:
        Polars DataFrame with calibration metrics
    """
```

#### Docstring Elements
1. **Summary**: One-line description
2. **Extended description** (optional): Additional context
3. **Args**: Parameter descriptions with types
4. **Returns**: Return value description
5. **Raises**: Exceptions that may be raised (if applicable)

#### Class Docstrings
```python
class BaseUncertaintyModel(ABC):
    """
    Base class for all uncertainty quantification models.

    All models must implement fit() and predict() methods.
    Calibration reports are provided via default implementation.
    """
```

## Code Patterns

### Error Handling

#### Exception Hierarchy
**Use specific exception types** from `utils/exceptions.py`:

```python
from ..utils.exceptions import (
    ModelNotFittedError,
    InvalidDataError,
    CalibrationSizeError,
    QuantileError,
    error_model_not_fitted,
    error_invalid_data,
    error_calibration_too_small,
)

# Example: Model not fitted
if not self._fitted:
    error_model_not_fitted("ClassName")

# Example: Invalid data
if missing:
    error_invalid_data(f"Columns not found: {missing}")

# Example: Calibration too small
if n_calib < 20:
    error_calibration_too_small(n_calib)
```

#### Exception Types
- `UncertaintyFlowError`: Base error (inherits from ValueError)
- `ModelNotFittedError`: Model methods called before fitting
- `InvalidDataError`: Invalid input data
- `CalibrationSizeError`: Calibration set too small
- `QuantileError`: Invalid quantile configuration

All exceptions inherit from `ValueError` for backward compatibility.

#### Validation Pattern
**Validate inputs early** with descriptive errors:

```python
def _validate_calibration_size(
    self,
    n_total: int,
    n_calib: int,
) -> None:
    """Validate calibration set size."""
    if n_calib < MIN_CALIBRATION_SIZE:
        error_calibration_too_small(n_calib)
```

#### Bare Except
**Avoid bare except blocks**:
```python
# ❌ BAD
try:
    ...
except Exception:
    pass

# ✅ GOOD
try:
    ...
except SpecificError as e:
    logger.warning(f"Expected error: {e}")
```

**Note**: One instance found in `ConformalRegressor` (line 161) - should be fixed

### Logging

#### Current State
**No structured logging** currently implemented

#### Recommendations
- Use Python's `logging` module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Log model training progress
- Log warnings for edge cases

### Abstract Base Classes

#### Implementation Pattern
```python
from abc import ABC, abstractmethod

class BaseUncertaintyModel(ABC):
    @abstractmethod
    def fit(self, data, target, **kwargs):
        ...

    @abstractmethod
    def predict(self, data):
        ...
```

**Rules**:
1. Inherit from `ABC`
2. Use `@abstractmethod` decorator
3. Use `...` for abstract methods
4. Document interface in docstring

### Property Methods

#### Pattern
```python
@property
def uncertainty_drivers_(self) -> pl.DataFrame | None:
    """
    Return residual correlation analysis results.

    Returns None if model has not been fitted.
    """
    return self._uncertainty_drivers
```

**Convention**: Trailing underscore for fitted attributes (sklearn style)

### Method Chaining

#### Pattern
**Return `self` from mutator methods**:

```python
def fit(self, data, target, **kwargs) -> "BaseUncertaintyModel":
    # ... fitting logic ...
    return self  # Enable method chaining
```

**Usage**: `model.fit(X, y).predict(X_test)`

## Polars Conventions

### DataFrame Handling

#### Accept Both DataFrame and LazyFrame
```python
def fit(self, data: PolarsInput, target: TargetSpec):
    # PolarsInput = pl.DataFrame | pl.LazyFrame
    if isinstance(data, pl.LazyFrame):
        data = data.collect()  # Materialize if needed
```

#### Lazy Evaluation
**Prefer LazyFrame for large datasets**:
```python
lazy_df = pl.scan_csv("large_file.csv")  # Lazy
result = lazy_df.filter(col > 0).collect()  # Materialize
```

#### Column Naming
**Use descriptive column names**:
```python
df = pl.DataFrame({
    "feature1": [...],
    "feature2": [...],
    "target": [...]
})
```

### Polars Expressions

#### Method Chaining
```python
result = (
    df
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg(pl.col("value").mean())
    .sort("category")
)
```

#### Type Safety
**Specify dtypes when creating DataFrames**:
```python
df = pl.DataFrame({
    "x": [1, 2, 3],
    "y": [1.0, 2.0, 3.0]
}, schema={"x": pl.Int32, "y": pl.Float64})
```

## Testing Conventions

### Test Structure

#### Fixtures
**Defined in `tests/conftest.py`**:
```python
@pytest.fixture
def sample_dataframe():
    return pl.DataFrame({...})

@pytest.fixture
def random_state():
    return 42
```

#### Test Functions
**Pattern**: `test_<function>_<scenario>`

```python
def test_fit_returns_self(sample_dataframe):
    model = DeepQuantileNet()
    result = model.fit(sample_dataframe, target="target")
    assert result is model

def test_predict_returns_distribution(sample_dataframe):
    model = DeepQuantileNet()
    model.fit(sample_dataframe, target="target")
    pred = model.predict(sample_dataframe)
    assert isinstance(pred, DistributionPrediction)
```

#### Test Classes
**Pattern**: `Test<ClassName>`

```python
class TestDeepQuantileNet:
    def test_fit(self, ...):
        ...

    def test_predict(self, ...):
        ...
```

### Test Data

#### Reproducibility
**Always set random seed**:
```python
np.random.seed(42)
```

#### Fixtures for Shared Data
**Use pytest fixtures** instead of creating data in each test

### Test Coverage

#### Goals
- **Core logic**: >90% coverage
- **Models**: >80% coverage
- **Utilities**: >70% coverage

#### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=uncertainty_flow

# Specific file
pytest tests/models/test_deep_quantile.py
```

## Code Organization

### Module Structure

#### Public vs Private
**Public API**: Exported in `__init__.py`
```python
# uncertainty_flow/__init__.py
from .core import BaseUncertaintyModel, DistributionPrediction

__all__ = ["BaseUncertaintyModel", "DistributionPrediction"]
```

**Private**: Not exported, prefixed with `_`

#### File Size
**Aim for**: <300 lines per file
**Maximum**: <500 lines (split if larger)

### Circular Dependencies

#### Avoidance Strategies
1. **TYPE_CHECKING imports**: Import for type hints only
2. **Lazy imports**: Import inside functions/methods
3. **Interface segregation**: Split large interfaces

**Example**:
```python
# In base.py
def calibration_report(self, ...):
    from ..calibration.report import calibration_report  # Lazy import
    return calibration_report(self, ...)
```

## Performance Conventions

### Vectorization
**Prefer vectorized operations** over loops:
```python
# ❌ BAD
for i in range(len(data)):
    result[i] = data[i] * 2

# ✅ GOOD
result = data * 2
```

### Memory Management
**Avoid unnecessary copies**:
```python
# ❌ BAD - creates copy
df2 = df.clone()

# ✅ GOOD - shares memory
df2 = df
```

### Lazy Evaluation
**Use LazyFrame for large datasets**:
```python
lazy_df = pl.scan_csv("large.csv")
result = lazy_df.filter(...).collect()
```

## Security Conventions

### Input Validation
**Validate all user inputs**:
```python
if not 0 <= calibration_size <= 1:
    raise ValueError("calibration_size must be between 0 and 1")
```

### No Hardcoded Secrets
**Never commit credentials**:
- No API keys in code
- No passwords in code
- Use environment variables for secrets

### Dependencies
**Keep dependencies updated**:
- Regular security audits
- Pin critical versions
- Use `uv sync` for reproducible installs

## Documentation Conventions

### README
**Sections**:
1. Project description
2. Key features
3. Installation
4. Quickstart examples
5. Coverage guarantees table
6. Contributing guide
7. License

### Code Comments
**When to add comments**:
- Complex algorithms
- Non-obvious optimizations
- Workarounds for bugs
- References to papers/implementation details

**When NOT to add comments**:
- Obvious code (self-documenting)
- Redundant information
- Outdated information

### Type Documentation
**Document complex types**:
```python
PolarsInput: Polars DataFrame or LazyFrame containing features
TargetSpec: Target column name (str) or list of names for multivariate
```

## Version Control Conventions

### Commit Messages
**Format**: Conventional Commits (when using git-commit skill)

```
feat: add DeepQuantileNetTorch model
fix: resolve circular dependency in calibration_report
docs: update ARCHITECTURE.md with new patterns
test: add coverage for edge cases in split strategies
```

### Branching
**Not strictly enforced** - but recommended:
- `main`: Production code
- `feature/<name>`: New features
- `fix/<name>`: Bug fixes
- `docs/<name>`: Documentation updates

## Development Workflow

### Environment Setup
```bash
# Create virtual environment
uv venv

# Install dependencies
uv sync

# Activate
source .venv/bin/activate
```

### Running Commands
```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .

# Testing
uv run pytest

# Type checking (if mypy added)
uv run mypy uncertainty_flow
```

### Code Review
**Before committing**:
1. Run linter: `ruff check .`
2. Run tests: `pytest`
3. Check coverage: `pytest --cov`
4. Verify imports: `python -c "import uncertainty_flow"`
