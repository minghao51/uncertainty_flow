# Uncertainty Flow — Overview

**Version 0.5.0** — A Python library for probabilistic forecasting and uncertainty quantification. Provides conformal prediction wrappers, quantile regression models, multivariate copulas, causal uncertainty estimation, and benchmarking tooling — all built on Polars DataFrames with a `DistributionPrediction` core abstraction.

## Architecture

**Pattern:** Layered library — core abstractions → models/wrappers → higher-level analysis → CLI/benchmarking/docs

```
┌─────────────────────────────────────────────────────────┐
│  CLI (click)          Benchmarking / Tuning             │
│  uncertainty_flow/cli.py   benchmarking/                │
├─────────────────────────────────────────────────────────┤
│  Higher-Level Analysis                                  │
│  causal/  counterfactual/  decomposition/               │
│  multimodal/  analysis/  risk/  bayesian/               │
├─────────────────────────────────────────────────────────┤
│  Models & Wrappers                                      │
│  models/  (QuantileForest, DeepQuantile, Transformer)   │
│  wrappers/ (ConformalRegressor, ConformalForecaster,    │
│             AdaptiveConformal, EnbPI, Classifier)       │
│  calibration/  multivariate/  viz/                      │
├─────────────────────────────────────────────────────────┤
│  Core                                                   │
│  core/base.py  core/distribution.py  core/config.py     │
│  core/types.py  core/parametric.py  core/_persistence.py│
├─────────────────────────────────────────────────────────┤
│  Utilities                                              │
│  utils/polars_bridge.py  utils/split.py                 │
│  utils/exceptions.py  utils/auto_tuning.py              │
│  utils/calibration_utils.py                             │
└─────────────────────────────────────────────────────────┘
```

### Core — Abstractions (Python)

The foundational layer that all models build upon.

| Layer | Location | Pattern |
|-------|----------|---------|
| Abstract model | `uncertainty_flow/core/base.py` | `BaseUncertaintyModel` ABC — `fit()` / `predict()` → `DistributionPrediction` |
| Prediction output | `uncertainty_flow/core/distribution.py` | `DistributionPrediction` — quantile matrix + levels, interval/mean/median/sample methods |
| Configuration | `uncertainty_flow/core/config.py` | `QuantileConfig(BaseSettings)` — pydantic-settings with env prefix `UNCERTAINTY_FLOW_` |
| Type aliases | `uncertainty_flow/core/types.py` | `PolarsInput`, `TargetSpec`, `DEFAULT_QUANTILES`, enums |
| Parametric dist | `uncertainty_flow/core/parametric.py` | `ParametricDistribution` — fit normal/t/lognormal/gamma from quantiles |
| Persistence | `uncertainty_flow/core/_persistence.py` | `.uf` zip archives (pickle + metadata.json + SHA-256) |
| Prediction sets | `uncertainty_flow/core/prediction_set.py` | `PredictionSet` for conformal classification |

**Entry point (library):** `import uncertainty_flow` → `uncertainty_flow/__init__.py:1`

**Entry point (CLI):** `uv run uncertainty-flow` → `uncertainty_flow/cli.py:571` (`main()`)

### Models — Native Predictors (Python)

| Model | Location | Backend |
|-------|----------|---------|
| `QuantileForestForecaster` | `models/quantile_forest.py` | sklearn `RandomForestRegressor` with leaf distributions |
| `DeepQuantileNet` | `models/deep_quantile.py` | sklearn `MLPRegressor` with shared trunk + linear quantile heads |
| `DeepQuantileNetTorch` | `models/deep_quantile_torch.py` | PyTorch MLP (optional, requires `[ml]`) |
| `TransformerForecaster` | `models/transformer_forecaster.py` | Amazon Chronos-2 + conformal calibration (optional, requires `[ml]`) |

### Wrappers — Conformal & Ensemble (Python)

| Wrapper | Location | Method |
|---------|----------|--------|
| `ConformalRegressor` | `wrappers/conformal.py` | Split conformal around any sklearn regressor |
| `ConformalForecaster` | `wrappers/conformal_ts.py` | Time-series conformal with lag features + copula |
| `ConformalClassifier` | `wrappers/conformal_classifier.py` | APS conformal classification → `PredictionSet` |
| `AdaptiveConformalForecaster` | `wrappers/adaptive_conformal.py` | Gibbs & Candès (2021) ACI for distribution shift |
| `EnsembleBootstrapPI` | `wrappers/enbpi.py` | Xu & Xie (2021) EnbPI bootstrap ensemble |

### Higher-Level Analysis (Python)

| Module | Location | Purpose |
|--------|----------|---------|
| `CausalUncertaintyEstimator` | `causal/estimator.py` | Doubly robust / S-learner / T-learner treatment effect estimation |
| `UncertaintyExplainer` | `counterfactual/explainer.py` | Counterfactual explanations (evolutionary/gradient search) |
| `EnsembleDecomposition` | `decomposition/ensemble.py` | Refit-based epistemic/aleatoric/total uncertainty decomposition |
| `CrossModalAggregator` | `multimodal/aggregator.py` | Per-feature-group models → product/copula/independent aggregation |
| `FeatureLeverageAnalyzer` | `analysis/leverage.py` | Feature perturbation-based uncertainty driver analysis |
| `ConformalRiskControl` | `risk/control.py` | Angelopoulos et al. conformal risk control for arbitrary loss functions |
| Risk functions | `risk/risk_functions.py` | `asymmetric_loss`, `financial_var`, `inventory_cost`, `threshold_penalty` |
| Copulas | `multivariate/copula.py` | Gaussian/Clayton/Gumbel/Frank copulas + auto-selection |
| `BayesianQuantileRegressor` | `bayesian/numpyro_model.py` | NumPyro NUTS MCMC with horseshoe priors (optional, requires `[ml]`) |
| `RecalibratedModel` | `calibration/recalibration.py` | Isotonic recalibration wrapper (Kuleshov et al. 2018) |
| Dashboard | `viz/dashboard.py` | Streamlit interactive dashboard |

## Key Data Flows

**Typical prediction flow:** `PolarsInput` → `model.fit(df, target)` → trains internally (numpy/sklearn/torch) → `model.predict(df)` → `DistributionPrediction` (quantile matrix + levels) → `.interval(0.9)` / `.median()` / `.mean()` / `.sample()`

**Benchmarking flow:** `uncertainty-flow benchmark --dataset weather` → registry-backed dataset/model resolution → Hamilton validation and model DAGs → staged Bronze/Silver/Gold/Platinum materialization → checksum/signature verification → typed JSON/CSV results

**Conformal calibration flow:** Training data → split into fit/calibration → fit base model on fit set → compute nonconformity scores on calibration set → predict: point prediction ± quantile of scores

**Save/load flow:** `model.save("path.uf")` → zip(pickle + metadata.json + SHA-256) → `BaseUncertaintyModel.load("path.uf")` → validates class type + optional hash

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Core language |
| Package manager | uv | Dependency management, virtual envs |
| Build system | hatchling | PEP 517 build backend |
| Data frames | Polars >=0.20.0 | Primary data structure (input/output) |
| Numerics | NumPy >=1.24.0 | Internal array computation |
| Columnar storage | PyArrow >=23.0 | Parquet I/O, Arrow interop |
| ML | scikit-learn >=1.3.0 | Base models (RF, GBM, MLP), preprocessing |
| Statistics | SciPy >=1.11.0 | Copulas, parametric fitting, stats |
| CLI | Click >=8.0.0 | Command-line interface |
| Dataflow | Hamilton >=1.89 | Side-effect-free benchmark DAG construction |
| Validation | Pydantic >=2.0.0 + pydantic-settings >=2.13.1 | Config models, env var binding |
| Plotting | Matplotlib >=3.7.0 | Static visualizations |
| Deep learning (opt) | PyTorch >=2.0.0 | DeepQuantileNetTorch |
| Time series foundation (opt) | chronos-forecasting >=2.0 | Amazon Chronos-2 transformer |
| Bayesian (opt) | NumPyro >=0.14.0 + JAX >=0.4.0 | MCMC Bayesian quantile regression |
| Explainability (opt) | SHAP >=0.44.0 | Feature attribution for interval width |
| Datasets (opt) | datasets >=2.0.0 | HuggingFace dataset loading |
| Docs | MkDocs + mkdocs-material >=9.5.0 | Documentation site |
| API docs | mkdocstrings[python] >=0.25.0 | Auto-generated Python API reference |
| Notebooks | Quarto | .qmd tutorial notebooks with freeze cache |
| Linting | Ruff >=0.1.0 | Formatter + linter |
| Type checking | mypy >=1.10.0 | Static type analysis |
| Security | bandit >=1.7.9 + pip-audit >=2.7.0 | Vulnerability scanning |
| Testing | pytest >=9.0.3 + pytest-cov >=4.1.0 | Test runner + coverage |
| Pre-commit | pre-commit >=4.6.0 | Git hook orchestration |

## Infrastructure

- **CI:** GitHub Actions (`.github/workflows/ci.yml`) — 5 jobs: pre-commit, typecheck, test (3.11/3.12/3.13), optional-stack (torch+numpyro), docs-check, security (bandit + pip-audit)
- **Docs deploy:** GitHub Actions (`.github/workflows/docs.yml`) — builds MkDocs + Quarto notebooks, deploys to GitHub Pages at `minghao51.github.io/uncertainty_flow/`
- **No Docker** — pure Python library, no containerized deployment
- **No server** — CLI + library, Streamlit dashboard is optional local visualization

## Integrations

| Service | SDK | Purpose | Status |
|---------|-----|---------|--------|
| HuggingFace Datasets | `datasets` (opt) | Load benchmark datasets (67 chronos datasets + ts-arena) | Optional (`[ml]`) |
| Amazon Chronos-2 | `chronos-forecasting` (opt) | Foundation time series model for TransformerForecaster | Optional (`[ml]`) |
| PyTorch | `torch` (opt) | GPU-accelerated deep quantile networks | Optional (`[ml]`) |
| NumPyro/JAX | `numpyro` + `jax` (opt) | Bayesian quantile regression via MCMC | Optional (`[ml]`) |
| SHAP | `shap` (opt) | Feature attribution for prediction interval width | Optional (`[ml]`) |
| Streamlit | (dashboard.py internal) | Interactive uncertainty exploration dashboard | Optional (lazy import) |
| sklearn | `scikit-learn` | Base regressors, calibration, preprocessing | Required |
| Polars | `polars` | Primary DataFrame library for all I/O | Required |

## Environment Variables

| Variable | Context | Purpose |
|----------|---------|---------|
| `UNCERTAINTY_FLOW_DEFAULT_QUANTILES` | `core/config.py:27` | Override default quantile levels (comma-separated floats) |
| `UNCERTAINTY_FLOW_MIN_CALIBRATION_SIZE` | `core/config.py:32` | Minimum calibration set size (default: 20) |
| `UNCERTAINTY_FLOW_WARN_CALIBRATION_SIZE` | `core/config.py:38` | Warning threshold for small calibration sets (default: 50) |
| `UNCERTAINTY_FLOW_DEFAULT_CHRONOS_MODEL` | `core/config.py:44` | Default Chronos model name for TransformerForecaster (default: `chronos-bolt-small`) |
