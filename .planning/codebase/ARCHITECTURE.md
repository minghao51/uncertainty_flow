# Uncertainty Flow Architecture

## Overall Pattern

The uncertainty_flow library follows a **layered architecture** with a clear separation of concerns. It's designed as a probabilistic forecasting and uncertainty quantification library with extensibility in mind.

### Architecture Layers

1. **Core Layer** (`core/`)
   - Base abstractions and interfaces
   - Type system and configuration management
   - Core data structures (DistributionPrediction)

2. **Implementation Layer** 
   - `models/`: Native uncertainty quantification models
   - `wrappers/`: Adapters for existing models (conformal prediction)
   - `bayesian/`: Bayesian approaches (optional - requires NumPyro)
   - `causal/`: Causal inference with uncertainty
   - `multimodal/`: Multi-modal uncertainty aggregation
   - `multivariate/`: Multivariate uncertainty handling

3. **Support Layer**
   - `metrics/`: Evaluation metrics (coverage, pinball loss, Winkler score)
   - `calibration/`: Calibration utilities
   - `utils/`: Common utilities and helpers
   - `benchmarking/`: Benchmarking framework and datasets

4. **Entry Points**
   - `__init__.py`: Public API surface
   - `cli.py`: Command-line interface for benchmarking

## Module Responsibilities

### Core Module (`core/`)
- **BaseUncertaintyModel**: Abstract base class defining the interface
- **DistributionPrediction**: Core output object with quantile predictions
- **Configuration**: Centralized configuration system
- **Type System**: Type aliases and constants

### Models Module (`models/`)
- **DeepQuantileNet**: Neural network-based quantile regression
- **QuantileForestForecaster**: Random forest-based quantile regression
- **DeepQuantileNetTorch**: PyTorch version (optional)
- **TransformerForecaster**: Transformer-based forecaster (optional)

### Wrappers Module (`wrappers/`)
- **ConformalRegressor**: Conformal prediction wrapper for regression
- **ConformalForecaster**: Conformal prediction wrapper for time series

### Bayesian Module (`bayesian/`)
- **BayesianQuantileRegressor**: Bayesian quantile regression with MCMC
- Provides posterior sampling and uncertainty intervals

### Causal Module (`causal/`)
- **CausalUncertaintyEstimator**: Treatment effect estimation with uncertainty
- Handles heterogeneity and average treatment effects

### Multi-Modal Module (`multimodal/`)
- **CrossModalAggregator**: Aggregates predictions across multiple modalities
- Handles cross-modal correlations

### Multivariate Module (`multivariate/`)
- **Copula**: Copula-based multivariate distributions
- Models dependencies between multiple targets

### Metrics Module (`metrics/`)
- **Coverage Score**: Measures prediction interval coverage
- **Pinball Loss**: Quantile regression loss function
- **Winkler Score**: Sharpness-aware coverage metric

## Data Flow Between Components

```
Input Data → BaseUncertaintyModel.fit() → Model Training
                        ↓
Model.predict() → DistributionPrediction → Post-processing
                        ↓
Metrics evaluation (calibration, coverage, sharpness)
```

### Key Abstractions

1. **BaseUncertaintyModel**
   - Abstract base class defining `fit()` and `predict()` methods
   - Provides default `calibration_report()` implementation
   - Enables method chaining across models

2. **DistributionPrediction**
   - Core output object holding quantile predictions
   - Supports univariate and multivariate cases
   - Provides methods for:
     - Extracting quantiles and intervals
     - Sampling from predicted distributions
     - Plotting fan charts
     - Bayesian posterior analysis
     - Multi-modal aggregation
     - Causal treatment effects

3. **Configuration System**
   - Centralized configuration via `get_config()`, `set_config()`
   - Default quantiles and model parameters
   - Type-safe configuration with Pydantic

## Entry Points and Public API

### Main Entry Points
1. **Package-level** (`__init__.py`)
   - Exports core classes: `BaseUncertaintyModel`, `DistributionPrediction`
   - Exports metrics: `pinball_loss`, `coverage_score`, `winkler_score`
   - Exports models: `DeepQuantileNet`, `QuantileForestForecaster`
   - Exports utilities: `RandomHoldoutSplit`, `TemporalHoldoutSplit`
   - Exports wrappers: `ConformalRegressor`, `ConformalForecaster`
   - Optional modules (lazy import): Bayesian, Causal, Multi-Modal

2. **Command Line Interface** (`cli.py`)
   - `uncertainty-flow benchmark`: Run benchmarks on datasets
   - `uncertainty-flow tune`: Auto-tune hyperparameters
   - `uncertainty-flow list-datasets`: List available datasets
   - `uncertainty-flow download-dataset`: Download datasets

### Public API Surface
```python
# Core
BaseUncertaintyModel
DistributionPrediction

# Metrics
pinball_loss
coverage_score
winkler_score

# Models
DeepQuantileNet
QuantileForestForecaster
DeepQuantileNetTorch  # optional
TransformerForecaster  # optional

# Wrappers
ConformalRegressor
ConformalForecaster

# Utilities
RandomHoldoutSplit
TemporalHoldoutSplit

# Optional modules (conditional import)
BayesianQuantileRegressor
CausalUncertaintyEstimator
CrossModalAggregator
```

## Design Principles

1. **Extensibility**: Easy to add new models by inheriting from BaseUncertaintyModel
2. **Modularity**: Optional dependencies (torch, transformers, numpyro)
3. **Type Safety**: Strong typing with Polars integration
4. **Performance**: Efficient NumPy backend with Polars interface
5. **Usability**: Rich API with calibration reports and diagnostics
6. **Benchmarking**: Built-in benchmarking framework with real datasets
