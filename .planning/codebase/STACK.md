# Stack

## Language
- **Python 3.11+** (required minimum)

## Build System
- **hatchling** — build backend
- **uv** — package manager (per project conventions)
- Standard `pyproject.toml` configuration

## Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| polars | >=0.20.0 | DataFrame operations |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | ML models, metrics |
| scipy | >=1.11.0 | Statistical functions |
| click | >=8.0.0 | CLI framework |
| pydantic | >=2.0.0 | Data validation |
| pydantic-settings | >=2.13.1 | Settings management |

## Optional Dependencies
| Extra | Package | Purpose |
|-------|---------|---------|
| torch | torch>=2.0.0 | Deep learning quantile models |
| transformers | chronos-forecasting>=2.0 | Time series transformer models |
| shap | shap>=0.44.0 | Model explainability |
| bench | datasets>=2.0.0 | HuggingFace benchmark datasets |
| numpyro | numpyro>=0.14.0, jax>=0.4.0 | Bayesian modeling |

## Dev Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=9.0.2 | Testing framework |
| pytest-cov | >=4.1.0 | Coverage reporting |
| ruff | >=0.1.0 | Linting + formatting |
| matplotlib | >=3.7.0 | Visualization |

## Configuration Files
| File | Tool | Notes |
|------|------|-------|
| pyproject.toml | hatch, ruff, pytest, mypy | Central config |
| uncertainty_flow/py.typed | PEP 561 | Marks package as typed |

## CLI Entry Point
- `uncertainty-flow` → `uncertainty_flow.cli:main` (via Click)

## Type Checking
- **mypy** configured (py311, warn_return_any, warn_unused_configs)
- PEP 561 typed package marker present
