# Uncertainty Flow Directory Structure

## Full Directory Tree

```
uncertainty_flow/
├── README.md
├── LICENSE
├── pyproject.toml
├── CLAUDE.md
├── AGENTS.md
├── data/
│   ├── electricity.parquet
│   ├── exchange_rate.parquet
│   ├── weather.parquet
│   └── README.md
├── docs/
│   ├── README.md
│   ├── api/
│   │   └── spec.md
│   ├── architecture/
│   │   └── overview.md
│   ├── archive/
│   │   ├── guides/
│   │   ├── plans/
│   │   │   ├── 2026-03-20-final-summary.md
│   │   │   ├── 2026-03-20-implementation-progress.md
│   │   │   └── 2026-03-20-uncertainty-flow-v1-design.md
│   │   └── README.md
│   ├── assets/
│   │   ├── charting-forecast-fan-chart.png
│   │   └── charting-regression-fan-chart.png
│   ├── benchmarks/
│   │   ├── comparison_table.csv
│   │   ├── comprehensive_v2_all.json
│   │   ├── comprehensive_v2_electricity.json
│   │   ├── comprehensive_v2_exchange_rate.json
│   │   ├── comprehensive_v2_weather.json
│   │   └── README.md
│   ├── guides/
│   │   ├── benchmarking.md
│   │   ├── calibration.md
│   │   ├── charting.md
│   │   ├── design.md
│   │   ├── distribution-approach.md
│   │   └── models.md
│   ├── plans/
│   │   └── 20260401-v6-plus-design.md
│   ├── project/
│   │   ├── changelog.md
│   │   ├── contributing.md
│   │   └── roadmap.md
│   ├── technical_roadmap.md
│   └── README.md
├── results/
│   ├── benchmark_report.md
│   ├── comparison_table.csv
│   ├── comprehensive_v2_all.json
│   ├── comprehensive_v2_electricity.csv
│   ├── comprehensive_v2_electricity.json
│   ├── comprehensive_v2_exchange_rate.csv
│   ├── comprehensive_v2_exchange_rate.json
│   ├── comprehensive_v2_weather.csv
│   └── comprehensive_v2_weather.json
├── scripts/
│   ├── comprehensive_benchmark.py
│   ├── generate_report.py
│   ├── ingest_datasets.py
│   ├── README.md
│   └── trial_benchmark.py
└── uncertainty_flow/
    ├── __init__.py
    ├── cli.py
    ├── py.typed
    ├── bayesian/
    │   ├── __init__.py
    │   └── ... (bayesian implementation files)
    ├── benchmarking/
    │   ├── __init__.py
    │   ├── ... (benchmarking implementation files)
    │   └── datasets/
    │       ├── __init__.py
    │       └── ... (dataset definitions)
    ├── calibration/
    │   ├── __init__.py
    │   ├── report.py
    │   ├── residual_analysis.py
    │   └── shap_values.py
    ├── causal/
    │   ├── __init__.py
    │   └── estimator.py
    ├── core/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── config.py
    │   ├── distribution.py
    │   └── types.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── coverage.py
    │   ├── pinball.py
    │   └── winkler.py
    ├── models/
    │   ├── __init__.py
    │   ├── deep_quantile.py
    │   ├── deep_quantile_torch.py
    │   ├── quantile_forest.py
    │   └── transformer_forecaster.py
    ├── multimodal/
    │   ├── __init__.py
    │   └── aggregator.py
    ├── multivariate/
    │   ├── __init__.py
    │   └── copula.py
    ├── utils/
    │   ├── __init__.py
    │   ├── calibration_utils.py
    │   ├── data_splitters.py
    │   ├── exceptions.py
    │   ├── polars_bridge.py
    │   └── validation.py
    └── wrappers/
        ├── __init__.py
        ├── conformal.py
        └── conformal_ts.py

├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_base_quantile.py
│   ├── test_config.py
│   ├── test_exceptions.py
│   ├── test_package_integration.py
│   ├── bayesian/
│   │   ├── __init__.py
│   │   └── test_numpyro_model.py
│   ├── calibration/
│   │   ├── test_residual_analysis.py
│   │   └── test_shap.py
│   ├── causal/
│   │   ├── __init__.py
│   │   └── test_estimator.py
│   ├── core/
│   │   ├── test_base.py
│   │   ├── test_distribution.py
│   │   └── test_types.py
│   ├── metrics/
│   │   ├── test_coverage.py
│   │   ├── test_pinball.py
│   │   └── test_winkler.py
│   ├── models/
│   │   ├── test_deep_quantile_torch.py
│   │   ├── test_quantile_forest.py
│   │   ├── test_deep_quantile.py
│   │   └── test_transformer.py
│   ├── multimodal/
│   │   ├── __init__.py
│   │   └── test_aggregator.py
│   ├── multivariate/
│   │   └── test_copula.py
│   ├── utils/
│   │   ├── test_split.py
│   │   ├── test_polars_bridge.py
│   │   └── test_calibration_report.py
│   └── wrappers/
│       ├── test_conformal.py
│       └── test_conformal_ts.py
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

### Evaluation
- **`uncertainty_flow/metrics/`**: Evaluation metrics
- **`uncertainty_flow/calibration/`**: Calibration utilities
- **`uncertainty_flow/benchmarking/`**: Benchmarking framework

### Utilities
- **`uncertainty_flow/utils/`**: Common utilities
  - `data_splitters.py`: Train/validation splitters
  - `calibration_utils.py`: Calibration diagnostics
  - `validation.py`: Input validation

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
