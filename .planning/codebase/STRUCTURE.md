# Directory Structure

## Overview
Uncertainty Flow follows a standard Python package structure with clear separation between source code, tests, documentation, and configuration files.

## Top-Level Directory Layout

```
uncertainty_flow/
├── uncertainty_flow/          # Main package source
├── tests/                     # Test suite
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── data/                      # Data files (for development/testing)
├── .claude/                   # Claude Code configuration
├── .planning/                 # Planning and codebase documentation
├── .venv/                     # Virtual environment (generated)
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
├── README.md                  # Project readme
├── CLAUDE.md                  # Claude Code instructions
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore patterns
```

## Package Structure (`uncertainty_flow/`)

### Core Module (`uncertainty_flow/core/`)
**Purpose**: Base classes, type definitions, and core abstractions

```
core/
├── __init__.py              # Public exports: BaseUncertaintyModel, DistributionPrediction
├── base.py                  # BaseUncertaintyModel abstract class
├── distribution.py          # DistributionPrediction result container
└── types.py                 # Type definitions (PolarsInput, TargetSpec, DEFAULT_QUANTILES)
```

**Key Files**:
- `base.py`: All models inherit from `BaseUncertaintyModel`
- `distribution.py`: `DistributionPrediction` class for prediction results
- `types.py`: Type aliases and constants

### Models Module (`uncertainty_flow/models/`)
**Purpose**: Concrete model implementations

```
models/
├── __init__.py              # Exports: DeepQuantileNet, DeepQuantileNetTorch, QuantileForestForecaster
├── deep_quantile.py         # sklearn-based neural network (MLPRegressor backend)
├── deep_quantile_torch.py   # PyTorch-based neural network
└── quantile_forest.py       # Quantile regression forest forecaster
```

**Key Files**:
- `deep_quantile.py`: Shared trunk architecture with sklearn backend
- `deep_quantile_torch.py`: PyTorch implementation (requires torch extra)
- `quantile_forest.py`: Time series forecasting model

### Wrappers Module (`uncertainty_flow/wrappers/`)
**Purpose**: Wrapper classes for adding uncertainty quantification

```
wrappers/
├── __init__.py              # Exports: ConformalRegressor, ConformalForecaster
├── conformal.py             # Conformal prediction wrapper for tabular data
└── conformal_ts.py          # Conformal prediction wrapper for time series
```

**Key Files**:
- `conformal.py`: Wraps any sklearn regressor with conformal prediction
- `conformal_ts.py`: Time series conformal prediction with temporal correction

### Metrics Module (`uncertainty_flow/metrics/`)
**Purpose**: Evaluation metrics for probabilistic predictions

```
metrics/
├── __init__.py              # Exports: pinball_loss, coverage_score, winkler_score
├── pinball.py               # Pinball/quantile loss implementation
├── coverage.py              # Coverage score (empirical coverage)
└── winkler.py               # Winkler score for interval evaluation
```

**Key Files**:
- `pinball.py`: Quantile loss function
- `coverage.py`: Empirical coverage calculation
- `winkler.py`: Interval scoring metric

### Calibration Module (`uncertainty_flow/calibration/`)
**Purpose**: Model calibration and diagnostic tools

```
calibration/
├── __init__.py              # Module initialization
├── report.py                # calibration_report() function
└── residual_analysis.py     # Feature-residual correlation analysis
```

**Key Files**:
- `report.py`: Generates calibration diagnostics
- `residual_analysis.py`: Identifies uncertainty drivers

### Utils Module (`uncertainty_flow/utils/`)
**Purpose**: Utility functions and helpers

```
utils/
├── __init__.py              # Exports: to_numpy, RandomHoldoutSplit, TemporalHoldoutSplit
├── polars_bridge.py         # Polars ↔ NumPy conversion utilities
├── split.py                 # Train/calibration split strategies
└── warnings.py              # Custom warning classes
```

**Key Files**:
- `polars_bridge.py`: `to_numpy()` function for backend compatibility
- `split.py`: `RandomHoldoutSplit` and `TemporalHoldoutSplit` classes
- `warnings.py`: Custom warning types

### Multivariate Module (`uncertainty_flow/multivariate/`)
**Purpose**: Multivariate uncertainty quantification

```
multivariate/
├── __init__.py              # Module initialization
└── copula.py                # Gaussian copula for joint distributions
```

**Key Files**:
- `copula.py`: Copula-based joint distribution modeling (in development)

## Test Structure (`tests/`)

**Purpose**: Comprehensive test suite mirroring package structure

```
tests/
├── __init__.py              # Test package marker
├── conftest.py              # Shared pytest fixtures
├── core/                    # Tests for core module
│   ├── test_base.py         # BaseUncertaintyModel tests
│   ├── test_distribution.py # DistributionPrediction tests
│   └── test_types.py        # Type utilities tests
├── models/                  # Tests for models
│   ├── test_deep_quantile.py        # DeepQuantileNet tests
│   └── test_deep_quantile_torch.py  # DeepQuantileNetTorch tests
├── metrics/                 # Tests for metrics
│   ├── test_pinball.py      # Pinball loss tests
│   ├── test_coverage.py     # Coverage score tests
│   └── test_winkler.py      # Winkler score tests
├── utils/                   # Tests for utilities
│   ├── test_polars_bridge.py # Polars bridge tests
│   └── test_split.py        # Split strategy tests
├── wrappers/                # Tests for wrappers
│   └── (currently empty)
└── calibration/             # Tests for calibration
    └── (currently empty)
```

**Key Files**:
- `conftest.py`: Shared fixtures (`sample_dataframe`, `sample_time_series`, `random_state`)

## Documentation (`docs/`)

**Purpose**: Project documentation and guides

```
docs/
├── guides/                  # User guides
│   └── models.md           # Model selection guide
└── project/                # Project documentation
    ├── roadmap.md          # Feature roadmap
    └── contributing.md     # Contributing guidelines
```

## Scripts (`scripts/`)

**Purpose**: Development and utility scripts

```
scripts/
├── benchmark.py            # Performance benchmarking
└── (other utility scripts)
```

## Configuration Files

### `pyproject.toml`
**Purpose**: Modern Python project configuration

**Key Sections**:
- `[project]`: Metadata (name, version, dependencies)
- `[project.optional-dependencies]`: dev and torch extras
- `[build-system]`: Hatchling build backend
- `[tool.ruff]`: Linter configuration
- `[tool.pytest.ini_options]`: Test configuration

### `CLAUDE.md`
**Purpose**: Claude Code (AI assistant) instructions

**Contents**:
- Development workflow guidelines
- Python environment management (uv)
- Frontend verification commands
- Documentation naming conventions

### `.gitignore`
**Purpose**: Git ignore patterns

**Ignores**:
- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv/`
- `*.pyc`
- `*.egg-info/`

### `uv.lock`
**Purpose**: Dependency lock file for reproducible builds

**Generated by**: uv package manager

## Key Locations

### Entry Points
- **Package**: `uncertainty_flow/__init__.py` - Public API exports
- **Main**: No CLI entry points (library-only package)

### Configuration
- **Project**: `pyproject.toml`
- **Claude**: `CLAUDE.md`
- **Git**: `.gitignore`
- **Tests**: `tests/conftest.py`

### Core Logic
- **Base Classes**: `uncertainty_flow/core/base.py`
- **Predictions**: `uncertainty_flow/core/distribution.py`
- **Models**: `uncertainty_flow/models/`
- **Wrappers**: `uncertainty_flow/wrappers/`

### Testing
- **Fixtures**: `tests/conftest.py`
- **Test Root**: `tests/`
- **Run Command**: `pytest` or `uv run pytest`

### Documentation
- **README**: `README.md` - Main project documentation
- **Guides**: `docs/guides/`
- **Internal**: `.planning/codebase/` - This documentation

## Naming Conventions

### Files
- **Modules**: `snake_case.py` (e.g., `deep_quantile.py`)
- **Tests**: `test_<module>.py` (e.g., `test_deep_quantile.py`)
- **Private**: `_leading_underscore.py` (not currently used)

### Directories
- **Packages**: `snake_case` (e.g., `uncertainty_flow/`)
- **Tests**: Mirror source structure

### Classes
- **Public**: `PascalCase` (e.g., `BaseUncertaintyModel`)
- **Base Classes**: `Base<Purpose>` (e.g., `BaseSplit`)
- **Private**: `_LeadingUnderscore` (not currently used)

### Functions
- **Public**: `snake_case` (e.g., `calibration_report`)
- **Private**: `_leading_underscore` (internal helpers)

### Constants
- **UPPER_CASE**: `DEFAULT_QUANTILES`
- **Private**: `_LEADING_UNDERSCORE`

## File Organization Patterns

### Module `__init__.py` Pattern
Each package module has an `__init__.py` that:
1. Imports from submodules
2. Defines `__all__` for public API
3. Provides clean public interface

### Test Organization
- Tests mirror source structure
- Each source module has corresponding test file
- Shared fixtures in `conftest.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function>_<scenario>`

### Documentation Pattern
- User-facing: `docs/guides/`
- Project docs: `docs/project/`
- Codebase docs: `.planning/codebase/` (this file)

## Generated Files

### Cache Directories
- `.pytest_cache/`: Pytest cache
- `.ruff_cache/`: Ruff linter cache
- `__pycache__/`: Python bytecode cache

### Virtual Environment
- `.venv/`: Virtual environment (created by uv)

### Lock Files
- `uv.lock`: Dependency lock file (managed by uv)

## Data Files (`data/`)

**Purpose**: Development and test data

```
data/
└── (development datasets)
```

**Usage**: Local development and testing only (not packaged)

## Planning Directory (`.planning/`)

**Purpose**: Project planning and codebase documentation

```
.planning/
└── codebase/               # This documentation
    ├── STACK.md
    ├── INTEGRATIONS.md
    ├── ARCHITECTURE.md
    ├── STRUCTURE.md
    ├── CONVENTIONS.md
    ├── TESTING.md
    └── CONCERNS.md
```
