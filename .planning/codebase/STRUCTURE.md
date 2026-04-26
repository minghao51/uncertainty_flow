# Uncertainty Flow Directory Structure

## Full Directory Tree

```
uncertainty_flow/
├── __init__.py              # Public API surface
├── cli.py                   # CLI entry point
├── py.typed                 # PEP 561 typing marker
├── analysis/                # Analysis utilities
├── bayesian/                # Bayesian approaches (NumPyro optional)
├── benchmarking/             # Benchmarking framework and datasets
├── calibration/             # Calibration utilities
├── causal/                   # Causal inference with uncertainty
├── core/                     # Base classes, types, configuration
│   ├── __init__.py
│   ├── base.py              # BaseUncertaintyModel abstract class
│   ├── config.py            # Configuration management
│   ├── distribution.py      # DistributionPrediction class
│   └── types.py             # Type aliases and constants
├── counterfactual/           # Counterfactual reasoning
├── decomposition/            # Uncertainty decomposition
├── decisions/                # Decision-making utilities
├── metrics/                  # Evaluation metrics
├── models/                  # Native uncertainty models
├── multimodal/               # Multi-modal aggregation
├── multivariate/            # Multivariate distributions
├── risk/                     # Risk assessment
├── utils/                    # Common utilities
├── viz/                      # Visualization utilities
└── wrappers/                 # Adapter wrappers (conformal prediction)

data/
├── electricity.parquet
├── exchange_rate.parquet
├── weather.parquet
└── README.md

docs/
├── README.md
├── api/spec.md
├── architecture/overview.md
├── archive/guides/, plans/
├── assets/
├── benchmarks/
├── guides/
├── plans/
├── project/
└── technical_roadmap.md

results/
├── benchmark_report.md
├── comparison_table.csv
└── comprehensive_v2_*.json, csv

scripts/
├── comprehensive_benchmark.py
├── generate_report.py
├── ingest_datasets.py
└── trial_benchmark.py

tests/
├── conftest.py               # Shared test fixtures
├── test_base_quantile.py
├── test_config.py
├── test_decisions.py
├── test_exceptions.py
├── test_package_integration.py
├── test_pytest_markers.py
├── test_utils.py
├── analysis/
├── bayesian/
├── calibration/
├── causal/
├── counterfactual/
├── core/
├── decomposition/
├── metrics/
├── models/
├── multimodal/
├── multivariate/
├── risk/
├── utils/
├── viz/
└── wrappers/
```

## Key Locations

### Core Modules
- **`uncertainty_flow/core/`**: Base classes, types, and configuration
  - `base.py`: Abstract base class for all models
  - `distribution.py`: DistributionPrediction class (core output)
  - `types.py`: Type aliases and constants
  - `config.py`: Configuration management

### Model Implementations
- **`uncertainty_flow/models/`**: Native uncertainty quantification models
  - `deep_quantile.py`: Deep learning quantile regression
  - `quantile_forest.py`: Random forest quantile regression
  - `deep_quantile_torch.py`: PyTorch implementation (optional)
  - `transformer_forecaster.py`: Transformer-based forecaster (optional)

### Wrappers
- **`uncertainty_flow/wrappers/`**: Adapters for existing models
  - `conformal.py`: Conformal prediction for regression
  - `conformal_ts.py`: Conformal prediction for time series

### Specialized Modules
- **`uncertainty_flow/bayesian/`**: Bayesian approaches (NumPyro optional)
- **`uncertainty_flow/causal/`**: Causal inference with uncertainty
- **`uncertainty_flow/multimodal/`**: Multi-modal aggregation
- **`uncertainty_flow/multivariate/`**: Multivariate distributions
- **`uncertainty_flow/counterfactual/`**: Counterfactual reasoning
- **`uncertainty_flow/decomposition/`**: Uncertainty decomposition
- **`uncertainty_flow/decisions/`**: Decision-making utilities
- **`uncertainty_flow/risk/`**: Risk assessment utilities

### Evaluation
- **`uncertainty_flow/metrics/`**: Evaluation metrics
- **`uncertainty_flow/calibration/`**: Calibration utilities
- **`uncertainty_flow/benchmarking/`**: Benchmarking framework
- **`uncertainty_flow/analysis/`**: Analysis utilities

### Utilities
- **`uncertainty_flow/utils/`**: Common utilities
  - `data_splitters.py`: Train/validation splitters
  - `calibration_utils.py`: Calibration diagnostics
  - `validation.py`: Input validation

### Visualization
- **`uncertainty_flow/viz/`**: Visualization utilities

## Entry Points

### Main Package Entry
- **`uncertainty_flow/__init__.py`**: Public API surface
  - Exports core classes, metrics, models, utilities
  - Conditional imports for optional dependencies

### CLI Entry
- **`uncertainty_flow/cli.py`**: Command-line interface
  - `main()`: Entry point for CLI
  - Commands: benchmark, tune, list-datasets, download-dataset

### Package Configuration
- **`pyproject.toml`**:
  - Defines entry point: `uncertainty-flow = uncertainty_flow.cli:main`
  - Optional dependencies: torch, transformers, shap, bench, numpyro

## Naming Conventions

### Files and Directories
- **Snake case** for Python files: `quantile_forest.py`
- **Pascal case** for classes: `BaseUncertaintyModel`
- **Snake case** for functions: `pinball_loss`
- **UPPER_CASE** for constants: `DEFAULT_QUANTILES`

### Module Organization
- **Feature-based grouping**: Each major capability gets its own subdirectory
- **Clear boundaries**: Core vs Implementation vs Support layers
- **Optional modules**: Separated with conditional imports

### Public API
- **Clean separation**: Public exports in `__init__.py`
- **Consistent naming**: Similar patterns across modules
- **Type hints**: Strong typing throughout with Polars integration

## Test Structure

### Test Organization
- **Parallel structure**: Mirrors source code in `tests/`
- **Module-specific tests**: Each module has its own test directory
- **Integration tests**: Root-level tests for package integration

### Key Test Locations
- **`tests/core/`**: Tests for base classes and types
- **`tests/models/`**: Tests for specific model implementations
- **`tests/wrappers/`**: Tests for wrapper classes
- **`tests/metrics/`**: Tests for evaluation metrics
- **`tests/utils/`**: Tests for utility functions

### Test Configuration
- **pytest** as test runner
- **conftest.py**: Shared test fixtures
- **Coverage collection**: Configured for comprehensive coverage
- **Warning filters**: Suppress common warnings from dependencies

## Data and Results

### Data Directory
- **`data/`**: Sample datasets in Parquet format
- **`results/`**: Benchmark results and reports
- **`scripts/`**: Data processing and benchmarking scripts

### Documentation
- **`docs/`**: Comprehensive documentation
  - API specifications
  - Architecture guides
  - Usage tutorials
  - Benchmarking results
