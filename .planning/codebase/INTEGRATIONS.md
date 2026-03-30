# External Integrations

## Overview
Uncertainty Flow has minimal external dependencies and integrations, focusing on local computation with Polars and scikit-learn as the primary interfaces.

## Data Sources & Formats

### Polars Integration
- **Primary Data Format**: Polars DataFrame and LazyFrame
- **Lazy Evaluation**: Full support for lazy computation
- **I/O Operations**:
  - CSV reading via `pl.read_csv()`
  - Native Polars data structures throughout
- **Type Safety**: Strong typing with `PolarsInput` (DataFrame | LazyFrame)

### NumPy Bridge
- **Utility**: `to_numpy()` function for converting Polars to NumPy
- **Purpose**: Backend compatibility for scikit-learn and PyTorch models
- **Location**: `uncertainty_flow/utils/polars_bridge.py`

## Machine Learning Frameworks

### Scikit-learn Integration
- **Wrapper Models**: `ConformalRegressor` wraps any sklearn estimator
- **Base Classes**: Uses `RegressorMixin` for model interfaces
- **Preprocessing**: `StandardScaler` for feature normalization
- **Supported Models**: Any sklearn-compatible regressor (RandomForest, GradientBoosting, etc.)

### PyTorch Integration (Optional)
- **Model**: `DeepQuantileNetTorch` for PyTorch backend
- **Purpose**: Alternative to sklearn-based DeepQuantileNet
- **GPU Support**: Via PyTorch (when torch extra installed)
- **Installation**: `pip install uncertainty-flow[torch]`

## Statistical & Mathematical Libraries

### SciPy
- **Optimization**: Used in quantile regression optimization
- **Distributions**: Statistical distribution functions
- **Numerical Routines**: Advanced mathematical operations

### NumPy
- **Array Operations**: Core numerical computations
- **Random Generation**: Reproducible random states
- **Statistical Functions**: Mean, std, quantile operations

## Visualization

### Matplotlib (Optional)
- **Plotting**: DistributionPrediction.plot() method
- **Fan Charts**: Uncertainty visualization
- **Calibration Overlays**: Model performance visualization
- **Installation**: Part of dev extra

## Testing Frameworks

### Pytest
- **Test Discovery**: Automatic test discovery
- **Fixtures**: Shared test fixtures in `conftest.py`
- **Coverage**: Integration with pytest-cov

## No External APIs

Uncertainty Flow does **NOT** integrate with:
- ❌ Cloud services (AWS, GCP, Azure)
- ❌ Databases (PostgreSQL, MongoDB, etc.)
- ❌ Auth providers (OAuth, JWT libraries)
- ❌ Monitoring services (Datadog, New Relic)
- ❌ External APIs (OpenAI, weather, financial data)

## File System Operations

### Local File Access
- **CSV Files**: Reading training/test data
- **Model Persistence**: Future consideration (pickle/joblib)
- **Configuration**: pyproject.toml for project settings

## Package Management

### UV (Recommended)
- **Dependency Management**: Modern Python package manager
- **Virtual Environment**: `.venv` directory
- **Commands**: `uv run`, `uv sync`, `uv add`

### pip (Standard)
- **Alternative Installation**: Standard pip installation supported
- **PyPI Distribution**: Published to PyPI

## Development Tools Integration

### Ruff
- **Linting**: Fast Python linting
- **Formatting**: Code formatting (configured in pyproject.toml)
- **Import Sorting**: Automatic import organization

### Git
- **Version Control**: Standard Git workflow
- **.gitignore**: Python-specific patterns included

## Time Series Support

### Native Time Series
- **No External TS Libraries**: Built from first principles
- **Models**: `QuantileForestForecaster`, `ConformalForecaster`
- **Features**: Temporal splitting, horizon-based forecasting
- **Multivariate**: Gaussian copula for joint predictions

## Future Integrations (Roadmap)

### Potential Future Additions
- **Joblib/Pickle**: Model persistence
- **PyTorch Lightning**: Training utilities
- **Weights & Biases**: Experiment tracking
- **Optuna**: Hyperparameter optimization
- **SHAP**: Explainability (Quantile SHAP)

## Integration Philosophy

- **Minimal Dependencies**: Keep core functionality self-contained
- **Polars-Native**: Embrace Polars as primary data format
- **Sklearn-Compatible**: Work with existing sklearn ecosystem
- **Optional Extras**: Extended features via optional dependencies
- **No Vendor Lock-in**: No proprietary cloud dependencies
