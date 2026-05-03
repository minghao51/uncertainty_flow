# Interactive Notebooks

Quarto `.qmd` notebooks are the **source of truth** for all examples. Each notebook
is rendered to static HTML via `quarto render` with `freeze: auto` caching.

| Notebook | Description |
|----------|-------------|
| [Quick Start: Conformal Regression](01-quick-start.md) | Wrap any scikit-learn model with statistically rigorous coverage guarantees using ConformalRegressor |
| [Time Series Forecasting with Uncertainty](02-time-series.md) | Forecast temperature with conformal prediction bands and compare ConformalForecaster vs QuantileForestForecaster |
| [Uncertainty Decomposition: Aleatoric vs. Epistemic](03-uncertainty-decomposition.md) | Decompose prediction uncertainty into aleatoric (data noise) and epistemic (model) components using bootstrap ensembles. |
| [Risk-Aware Decision Making](04-risk-aware-decisions.md) | Use ConformalRiskControl with custom risk functions for cost-sensitive predictions. |
| [Multivariate Copulas & Cross-Modal Aggregation](05-multivariate-copulas.md) | Copula families, auto-selection, joint sampling, and multivariate forecasting |
| [Methods & Benchmarks Overview](06-methods-benchmarks.md) | Every model in uncertainty_flow, when to use it, and how they compare on real benchmarks |
| [Split Strategies: Sensible Defaults for Validation](07-split-strategies.md) | Auto-selecting the right validation strategy based on task type and data size |

## Run Locally

```bash
uv sync --extra opinion
make notebooks
```
