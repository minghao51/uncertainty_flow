# Tech Stack

## Languages & Runtime
- **Python**: 3.11+ (3.11, 3.12, 3.13 tested in CI)

## Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| polars | >=0.20.0 | Data frames, lazy evaluation |
| numpy | >=1.24.0 | Numerical computing |
| pyarrow | >=23.0 | Arrow format interoperability |
| scikit-learn | >=1.3.0 | RandomForest for quantile regression |
| scipy | >=1.11.0 | Scientific computing |
| click | >=8.0.0 | CLI framework |
| pydantic | >=2.0.0 | Data validation |
| pydantic-settings | >=2.13.1 | Environment-based settings |

## Optional Dependencies
| Package | Purpose |
|---------|---------|
| torch | Deep quantile networks |
| chronos-forecasting | Transformer-based forecasting |
| shap | SHAP value analysis |
| datasets | HuggingFace dataset loading |
| numpyro + jax | Bayesian quantile regression |
| streamlit | Interactive dashboard |
| matplotlib | Plotting |

## Dev Dependencies
- pytest, pytest-cov, ruff, matplotlib

## Configuration Files
- `pyproject.toml` - Project metadata, dependencies, build config
- `pyproject.toml [tool.ruff]` - Linting rules (E, F, I, N, W)
- `pyproject.toml [tool.pytest.ini_options]` - Test markers, paths, warnings
- `pyproject.toml [tool.mypy]` - Type checking (Python 3.11)

## Build System
- **hatchling** - PEP 517 build backend
- **uv** - Package manager (used in CI and local dev)

## CLI Entry Point
- `uncertainty-flow` command via `uncertainty_flow.cli:main`
