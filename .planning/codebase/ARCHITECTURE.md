# Architecture

## Overview
Uncertainty Flow follows a **distribution-first architecture** where all models return probabilistic predictions rather than point estimates. The design emphasizes Polars-native data handling, modular components, and clear separation of concerns.

## Core Architectural Pattern

### Distribution-First Design
The central architectural principle is that every model returns a `DistributionPrediction` object, not a scalar value. This enforces uncertainty quantification as a first-class concern.

```python
# All models follow this pattern:
model.fit(data, target="y")
prediction = model.predict(new_data)  # Returns DistributionPrediction
prediction.interval(confidence=0.9)   # Extract uncertainty intervals
prediction.quantile([0.1, 0.5, 0.9]) # Extract quantiles
```

## Architectural Layers

### 1. Base Layer (`core/`)

**BaseUncertaintyModel** - Abstract interface
- All models inherit from this base class
- Enforces `fit()` and `predict()` methods
- Provides default `calibration_report()` implementation
- Defines contract for uncertainty quantification models

**DistributionPrediction** - Result container
- Immutable container for probabilistic predictions
- Methods: `quantile()`, `interval()`, `mean()`, `sample()`, `plot()`
- Handles both univariate and multivariate predictions
- Stores quantile matrix internally

**Configuration System** - Pydantic-based settings
- `QuantileConfig`: Centralized configuration for quantile levels and thresholds
- Environment variable support via `UNCERTAINTY_FLOW_*` prefix
- Validation for quantile ranges and calibration sizes
- Single source of truth for `DEFAULT_QUANTILES`

**Type System** - Type definitions
- `PolarsInput`: DataFrame | LazyFrame
- `TargetSpec`: str | list[str] (single or multiple targets)
- Strong typing throughout with TYPE_CHECKING

### 2. Model Layer (`models/`)

**BaseQuantileNeuralNet** - Abstract base for neural quantile models
- Provides common functionality for neural quantile regression
- Data preparation (Polars and numpy support)
- Feature scaling with StandardScaler
- Monotonicity enforcement
- Consistent fit/predict interface
- Abstract methods: `_fit_backend()`, `_predict_backend()`

**Concrete Implementations**:
- `DeepQuantileNet`: sklearn-based neural network
- `DeepQuantileNetTorch`: PyTorch-based neural network
- `QuantileForestForecaster`: Time series forecasting

**Pattern**: Neural models inherit from BaseQuantileNeuralNet
- Shared trunk architecture for quantile models
- Post-prediction sorting to ensure non-crossing quantiles
- Backend-specific implementations via abstract methods
- Both sklearn and torch backends have identical interfaces

### 3. Wrapper Layer (`wrappers/`)

**ConformalRegressor** - Conformal prediction wrapper
- Wraps any sklearn regressor
- Adds statistical coverage guarantees
- Split-conformal or cross-conformal strategies
- Calibration set splitting strategies

**ConformalForecaster** - Time series conformal wrapper
- Temporal adaptation of conformal prediction
- Handles time series cross-validation
- Corrects for exchangeability violations

### 4. Metrics Layer (`metrics/`)

**Evaluation Functions**:
- `pinball_loss()`: Quantile loss
- `coverage_score()`: Empirical coverage
- `winkler_score()`: Interval scoring

**Pattern**: Stateless functions taking predictions + true values
- Compatible with DistributionPrediction objects
- Return scalar or Polars DataFrame results

### 5. Calibration Layer (`calibration/`)

**Report Generation**:
- `calibration_report()`: Comprehensive calibration metrics
- `residual_analysis()`: Feature-residual correlations
- Returns Polars DataFrame for easy analysis

**Uncertainty Driver Detection**:
- Automatic correlation analysis
- Identifies which features drive prediction uncertainty
- Accessible via `model.uncertainty_drivers_` property

### 6. Utilities Layer (`utils/`)

**Exception Hierarchy**: Custom error types
- `UncertaintyFlowError`: Base error class
- `ModelNotFittedError`: Model methods called before fitting
- `InvalidDataError`: Invalid input data
- `CalibrationSizeError`: Calibration set too small
- `QuantileError`: Invalid quantile configuration
- Helper functions: `error_model_not_fitted()`, `error_invalid_data()`, etc.

**Calibration Utilities**: Shared calibration functions
- `calibration_report()`: Extracted to avoid circular dependencies
- Re-exported from `calibration/report.py` for backward compatibility

**Polars Bridge**: Convert between Polars and NumPy
**Split Strategies**: Random and temporal holdout splits
**Warnings**: Custom warning classes

## Data Flow

### Training Flow
```
Polars DataFrame → Model.fit() → Internal State
                 ↓
            Calibration Set
                 ↓
         Quantile Estimation
                 ↓
         Trained Model
```

### Prediction Flow
```
Polars DataFrame → Model.predict() → DistributionPrediction
                                      ↓
                              quantile_matrix
                                      ↓
                        interval() / quantile() / mean()
```

### Calibration Flow
```
Model + Validation Data → calibration_report() → Metrics DataFrame
                                                    ↓
                                         Coverage, Sharpness, Scores
```

## Key Design Patterns

### 1. Abstract Base Class Pattern
- `BaseUncertaintyModel` defines interface
- All models implement fit/predict
- Enforces consistency across implementations

### 2. Immutable Result Container
- `DistributionPrediction` is immutable
- Prevents accidental modification
- Clear separation between model and predictions

### 3. Strategy Pattern
- Calibration split strategies: `BaseSplit` hierarchy
- Random holdout vs temporal holdout
- Pluggable strategies for different use cases

### 4. Wrapper Pattern
- `ConformalRegressor` wraps any sklearn estimator
- Adds uncertainty quantification to point predictors
- Enables conformal prediction for existing models

### 5. Bridge Pattern
- `to_numpy()` bridges Polars and NumPy ecosystems
- Maintains Polars-first approach while enabling sklearn compatibility

## Abstractions

### Model Abstraction
All models expose:
- `fit(data, target, **kwargs)`: Training interface
- `predict(data)`: Prediction interface
- `calibration_report(data, target)`: Evaluation interface
- `uncertainty_drivers_`: Feature importance (optional)

### Prediction Abstraction
`DistributionPrediction` provides:
- `quantile(q)`: Quantile extraction
- `interval(confidence)`: Prediction intervals
- `mean()`: Point estimate
- `sample(n)`: Monte Carlo samples
- `plot()`: Visualization

### Data Abstraction
- `PolarsInput`: Accepts DataFrame or LazyFrame
- Lazy evaluation supported throughout
- Materialization only when necessary

## Entry Points

### Public API (`__init__.py`)
```python
# Core
from uncertainty_flow import BaseUncertaintyModel, DistributionPrediction

# Models
from uncertainty_flow import DeepQuantertaintyNet, QuantileForestForecaster

# Wrappers
from uncertainty_flow import ConformalRegressor, ConformalForecaster

# Metrics
from uncertainty_flow import pinball_loss, coverage_score, winkler_score

# Utilities
from uncertainty_flow import RandomHoldoutSplit, TemporalHoldoutSplit, to_numpy
```

### CLI Entry Points
Currently none - library-only package

## Coupling & Cohesion

### High Cohesion
- Each module has single, well-defined responsibility
- `core/`: Base classes and types
- `models/`: Concrete model implementations
- `metrics/`: Evaluation functions
- `calibration/`: Calibration and analysis

### Low Coupling
- Models don't depend on each other
- Metrics are standalone functions
- Wrappers depend only on sklearn, not on internal models
- Polars used throughout for consistent data interface

### Known Coupling Issues
- ~~**Circular dependency**: `BaseUncertaintyModel.calibration_report()` imports calibration module lazily~~ **RESOLVED**: `calibration_report()` moved to `utils/calibration_utils.py`
- **Metric imports**: Some models import metrics directly (tight coupling)
- **Type checking**: Uses TYPE_CHECKING to avoid circular imports

## Multivariate Architecture

### Marginal Approach
- Each target gets marginal CDF
- Quantiles computed independently per target
- Simple but ignores correlations

### Copula Approach (Roadmap)
- Gaussian copula for joint distributions
- Captures cross-target correlations
- Implemented in `multivariate/copula.py`

## Time Series Architecture

### Horizon-Based Forecasting
- Models predict multiple steps ahead
- `QuantileForestForecaster`: Direct multi-step forecasting
- `ConformalForecaster`: Conformal prediction for time series

### Temporal Splitting
- `TemporalHoldoutSplit`: Time-aware cross-validation
- Respects temporal ordering
- Prevents look-ahead bias

## Performance Considerations

### Lazy Evaluation
- LazyFrame support throughout
- Materialization only when needed
- `.collect()` called strategically

### Memory Management
- Potential issue: Large sample generation
- Multiple `.collect()` calls in fit/predict
- Room for optimization in sampling methods

### Computational Efficiency
- Vectorized operations via Polars
- NumPy bridge for sklearn compatibility
- Post-prediction sorting (not during optimization)

## Extension Points

### Adding New Models
1. **For neural quantile models**: Inherit from `BaseQuantileNeuralNet`
   - Implement `_fit_backend(x, y)` for training logic
   - Implement `_predict_backend(x)` for prediction logic
   - Data preparation and scaling handled by base class
2. **For other models**: Inherit from `BaseUncertaintyModel`
   - Implement `fit()` and `predict()` methods
   - Return `DistributionPrediction` from predict
3. Optionally override `calibration_report()`

### Adding New Metrics
1. Create function in `metrics/`
2. Accept DistributionPrediction + true values
3. Return scalar or Polars DataFrame
4. Export in `metrics/__init__.py`

### Adding New Split Strategies
1. Inherit from `BaseSplit` in `utils/split.py`
2. Implement `split()` method
3. Validate calibration size requirements using `error_calibration_too_small()`
4. Export in utils module

### Adding New Exception Types
1. Add to `utils/exceptions.py` hierarchy
2. Inherit from appropriate base class (`ModelError`, `DataError`, etc.)
3. Add error code (e.g., "UF-E005")
4. Create helper function if commonly used
5. Export in `utils/__init__.py`
