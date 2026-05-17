# uncertainty-flow — Current State

Last updated: 2026-05-16

## What's Implemented

| Feature | Backend | Frontend |
|---------|---------|----------|
| Base uncertainty model (ABC) | Full | N/A |
| DistributionPrediction (quantiles, PIT, CRPS, sampling) | Full | N/A |
| Config (Pydantic settings, env overrides) | Full | N/A |
| Parametric distribution fitting (Normal, Student-t, LogNormal, Gamma) | Full | N/A |
| PredictionSet (classification) | Full | N/A |
| Model persistence (.uf archive with SHA-256) | Full | N/A |
| DeepQuantileNet (sklearn MLP trunk + linear heads) | Full | N/A |
| DeepQuantileNetTorch (PyTorch multi-quantile MLP) | Full (optional) | N/A |
| QuantileForestForecaster (time series) | Full | N/A |
| TransformerForecaster (Chronos-2) | Full (optional) | N/A |
| ConformalRegressor | Full | N/A |
| ConformalForecaster (time series) | Full | N/A |
| ConformalClassifier (APS procedure) | Full | N/A |
| AdaptiveConformalForecaster (Gibbs & Candes 2021) | Full | N/A |
| EnsembleBootstrapPI (ENBPI) | Full | N/A |
| BayesianQuantileRegressor (NumPyro MCMC) | Full (optional) | N/A |
| RecalibratedModel (isotonic recalibration) | Full | N/A |
| Calibration reports | Full | N/A |
| Residual analysis (uncertainty drivers) | Full | N/A |
| SHAP-based interval width attribution | Full (optional) | N/A |
| CrossModalAggregator (product, independent) | Partial | N/A |
| Copula models (multivariate dependency) | Full | N/A |
| CausalUncertaintyEstimator (DR, S-learner, T-learner) | Full | N/A |
| Counterfactual search/explainer | Full | N/A |
| EnsembleDecomposition (aleatoric/epistemic) | Full | N/A |
| FeatureLeverageAnalyzer | Full | N/A |
| ConformalRiskControl | Full | N/A |
| Risk functions (financial VaR, inventory, threshold) | Full | N/A |
| CRPS / coverage / pinball / Winkler / log-score / multivariate scores | Full | N/A |
| Split strategies (holdout, temporal, KFold, rolling-origin) | Full | N/A |
| Auto-tuning (coverage-targeted hyperparam search) | Full | N/A |
| Benchmark runner (CLI + API) | Full | N/A |
| Dataset registry (60+ datasets, local + HuggingFace) | Full | N/A |
| Streamlit dashboard | Full (optional) | N/A |
| CLI (benchmark, tune, list/download datasets) | Full | N/A |

## Stubbed / Unimplemented

All files below raise `NotImplementedError` or return `501`:

- `uncertainty_flow/multimodal/aggregator.py:157` — `aggregation='copula'` is declared as a valid option in `VALID_AGGREGATIONS` but raises `NotImplementedError` ("not implemented yet for CrossModalAggregator")
- `uncertainty_flow/models/base_quantile.py:258` — abstract method `_fit_backend` raises `NotImplementedError` (template method pattern; must be overridden by `DeepQuantileNet` and `DeepQuantileNetTorch`)
- `uncertainty_flow/models/base_quantile.py:274` — abstract method `_predict_backend` raises `NotImplementedError` (same pattern)

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| High | `except Exception: pass` silently swallows all errors when setting `random_state` on bootstrap ensemble models — masks legitimate failures (e.g., read-only attrs, type errors) | `uncertainty_flow/decomposition/ensemble.py:118` |
| Medium | 4 mypy `valid-type` errors: `callable` used as type annotation instead of `typing.Callable` in `_FamilySpec` dataclass fields | `uncertainty_flow/core/parametric.py:83-87` |
| Medium | mypy `index` error: indexing Polars `Series` with a `str` key — invalid index type | `uncertainty_flow/wrappers/adaptive_conformal.py:126` |
| Medium | mypy `arg-type` error: `RollingOriginSplit.splits()` receives `DataFrame | None` due to `self.df` being nullable | `uncertainty_flow/benchmarking/runner.py:457` |
| Medium | mypy `assignment` error: numpy array assigned where Polars `Series` expected | `uncertainty_flow/viz/_plotting.py:119` |
| Medium | `BenchmarkRunner.to_dict()` serializes model results twice — once under `"models"` key (backward-compat alias) and identically under `"results"` key, producing ~2x JSON output | `uncertainty_flow/benchmarking/runner.py:591-650` |
| Low | ruff E501 line-length violation (105 > 100 chars) | `scripts/ci_policy_checks.py:34` |
| Low | Unused import `sys` | `scripts/ci_policy_checks.py:8` |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| High | `pickle.load()` used for model deserialization — arbitrary code execution risk when loading untrusted `.uf` files. SHA-256 integrity check is optional and verifies corruption, not authenticity (no signature verification) | `uncertainty_flow/core/_persistence.py:250` |
| Low | `UNCERTAINTY_FLOW_HF_REVISION` env var accepted without validation — could reference untrusted/malicious HuggingFace dataset revisions | `uncertainty_flow/benchmarking/datasets.py:570` |

## Performance Issues

| Issue | Location |
|-------|----------|
| `FeatureLeverageAnalyzer` runs `O(n_features * n_repeats)` full model predictions without parallelization — perturbation loop is sequential | `uncertainty_flow/analysis/leverage.py:340-346` |
| `LogScore` fits a parametric distribution per sample in a Python loop — `O(n)` scipy optimizations | `uncertainty_flow/metrics/log_score.py:51-53` |
| `DistributionPrediction._forward_cdf` loops row-by-row in Python (`for i in range(n)`) — not vectorized | `uncertainty_flow/core/distribution.py:244` |
| `BaseUncertaintyModel.predict_batch` processes chunks sequentially with no concurrency or GPU batching | `uncertainty_flow/core/base.py:80-82` |
| `CrossModalAggregator.predict()` calls `model.predict()` separately per group then aggregates — groups could run in parallel | `uncertainty_flow/multimodal/aggregator.py:134-137` |
| `log_score_kde` draws samples and fits a Gaussian KDE per observation in a Python loop | `uncertainty_flow/metrics/log_score.py:125-135` |

## Maintenance Issues

| Issue | Detail |
|-------|--------|
| Scattered optional dependency guards | 6 modules use `try/except ImportError` blocks (`uncertainty_flow/__init__.py`, `uncertainty_flow/models/__init__.py`, `uncertainty_flow/bayesian/__init__.py`, `uncertainty_flow/viz/dashboard.py:72-77`, `uncertainty_flow/calibration/shap_values.py:62-68`, `uncertainty_flow/models/transformer_forecaster.py`) — no centralized optional dep manager |
| 5 source files excluded from coverage | `pyproject.toml:94-99` omits `bayesian/numpyro_model.py`, `models/deep_quantile_torch.py`, `models/transformer_forecaster.py`, `calibration/shap_values.py`, `viz/dashboard.py` — these have zero tested coverage |
| Low coverage floor (40%) | `pyproject.toml:102` — `fail_under = 40` means 60% of source can be untested without CI failure |
| 7 mypy errors in 4 files | `core/parametric.py` (4 errors), `viz/_plotting.py` (1 error), `wrappers/adaptive_conformal.py` (1 error), `benchmarking/runner.py` (1 error) — type safety gaps |
| `callable` used instead of `typing.Callable` | `uncertainty_flow/core/parametric.py:83-87` — `builtins.callable` is not valid as a type annotation per mypy |
| Duplicate JSON output in benchmark serialization | `BenchmarkRunner.to_dict()` at `uncertainty_flow/benchmarking/runner.py:591-650` emits identical model data under both `"models"` and `"results"` keys — doubles JSON payload size |
| Missing shared model registry | `MODEL_REGISTRY` in `uncertainty_flow/benchmarking/runner.py:136` is a plain dict with decorator registration — adding new benchmark models requires editing this file; no plugin/discovery mechanism |
| Hardcoded constants scattered across modules | `viz/dashboard.py:380` limits to 6 features; `viz/_plotting.py:16` hardcodes 500 max samples; `analysis/leverage.py:203` hardcodes 800-row prediction budget |
| Persistence format has no forward-compat path | `uncertainty_flow/core/_persistence.py:22` — `SUPPORTED_FORMAT_VERSIONS = {1}` only; no migration tooling |
| `DEFAULT_QUANTILES` is a dynamic proxy | `uncertainty_flow/core/types.py:25-48` — `_ConfigQuantiles` reads from global config on each access; callers that cache the value (e.g., in a list comprehension) may get stale config silently |
| No CI coverage for optional deps beyond torch | CI workflow only tests `ml` extras — `numpyro`, `streamlit`, `shap` optional paths untested in CI |
| Broad `except Exception` in CLI error handlers | `uncertainty_flow/cli.py:317,434,514,558` — catches all exceptions including `KeyboardInterrupt`, `SystemExit` in command handlers |
