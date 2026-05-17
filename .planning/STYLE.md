# uncertainty-flow — Code Style & Conventions

## File Organization

### Where Things Go

```
uncertainty_flow/                     # Main package (src-layout)
  __init__.py                         # Public API with conditional optional imports
  cli.py                              # Click CLI entry point
  core/                               # Foundational abstractions
    base.py                           #   BaseUncertaintyModel (ABC)
    distribution.py                   #   DistributionPrediction (core output object)
    types.py                          #   Type aliases, enums, constants
    config.py                         #   Pydantic QuantileConfig + global singleton
    parametric.py                     #   ParametricDistribution
    prediction_set.py                 #   PredictionSet
    _persistence.py                   #   Save/load .uf archive logic
  models/                             # Native model implementations
    base_quantile.py                  #   BaseQuantileNeuralNet (intermediate ABC)
    quantile_forest.py                #   QuantileForestForecaster
    deep_quantile.py                  #   DeepQuantileNet (scikit-learn)
    deep_quantile_torch.py            #   DeepQuantileNetTorch (torch, optional)
    transformer_forecaster.py         #   TransformerForecaster (chronos, optional)
  wrappers/                           # Conformal prediction wrappers
    conformal.py                      #   ConformalRegressor (sklearn wrapper)
    conformal_classifier.py           #   ConformalClassifier
    conformal_ts.py                   #   ConformalForecaster (time series)
    adaptive_conformal.py             #   AdaptiveConformalForecaster
    enbpi.py                          #   EnsembleBootstrapPI
  metrics/                            # Scoring/evaluation metrics
    crps.py                           #   CRPS
    pinball.py                        #   Pinball loss
    winkler.py                        #   Winkler score
    coverage.py                       #   Coverage score
    calibration.py                    #   Calibration error
    log_score.py                      #   Log score (KDE, pooled)
    point.py                          #   MAE, RMSE
    comparison.py                     #   Diebold-Mariano, skill score, MCS
    multivariate.py                   #   Energy score, variogram score
  calibration/                        # Calibration & diagnostics
    residual_analysis.py              #   Uncertainty driver computation
    recalibration.py                  #   Recalibration methods
    shap_values.py                    #   SHAP for interval width (optional)
    report.py                         #   Calibration report generation
  decomposition/                      # Uncertainty decomposition
    ensemble.py                       #   EnsembleDecomposition (aleatoric/epistemic)
  risk/                               # Risk-aware decision making
    risk_functions.py                 #   Loss functions (asymmetric, VaR, inventory)
    control.py                        #   ConformalRiskControl
  multivariate/                       # Copula-based joint modeling
    copula.py                         #   BaseCopula + families (Gaussian, Clayton, etc.)
  multimodal/                         # Multi-modal integration
    aggregator.py                     #   CrossModalAggregator
  analysis/                           # Feature analysis
    leverage.py                       #   FeatureLeverageAnalyzer
  bayesian/                           # Bayesian methods (optional)
    numpyro_model.py                  #   BayesianQuantileRegressor (optional)
  causal/                             # Causal uncertainty
    estimator.py                      #   CausalUncertaintyEstimator
  counterfactual/                     # Counterfactual explanation
    explainer.py                      #   UncertaintyExplainer
    search.py                         #   Search utilities
  utils/                              # Shared utilities
    polars_bridge.py                  #   Polars↔NumPy conversions
    split.py                          #   Train/validation split strategies
    auto_tuning.py                    #   Coverage-target hyperparameter tuning
    exceptions.py                     #   Error hierarchy (UF-E001..E004)
    calibration_utils.py              #   Calibration helpers
  viz/                                # Visualization
    _plotting.py                      #   Matplotlib plot functions
    dashboard.py                      #   Marimo-based dashboard (optional)
  benchmarking/                       # Benchmarking infrastructure
    datasets.py                       #   DatasetInfo registry + HuggingFace loader
    runner.py                         #   BenchmarkRunner
    tuning.py                         #   Auto-tuning pipeline
tests/                                # Test suite (mirrors src structure)
  conftest.py                         #   Shared fixtures + auto-marker hook
  test_*.py                           #   Top-level tests (imports, config, cli, etc.)
  core/test_*.py                      #   Tests for uncertainty_flow/core/
  models/test_*.py                    #   Tests for uncertainty_flow/models/
  metrics/test_*.py                   #   Tests for uncertainty_flow/metrics/
  wrappers/test_*.py                  #   Tests for uncertainty_flow/wrappers/
  ...                                 #   (mirrors every source subpackage)
benchmarks/                           # Standalone benchmark scripts
  run_benchmarks.py                   #   CLI entry point for batch runs
  ingest_datasets.py                  #   Dataset download/ingestion
  generate_report.py                  #   Report generation from results
  benchmark_utils.py                  #   Shared benchmark helpers
notebooks/                            # Quarto notebooks (tutorials, benchmarks)
  *_*.qmd                             #   Numbered tutorial notebooks
  _quarto.yml                         #   Quarto project config
data/                                 # Parquet datasets (gitignored .parquet)
scripts/                              # CI/repo-management scripts
docs/                                 # MkDocs documentation source
results/                              # Benchmark result CSVs/JSONs (gitignored)
```

## Naming Conventions

### Python

| Element | Convention | Example |
|---------|-----------|---------|
| Files/Dirs | `snake_case` | `base_quantile.py`, `test_ensemble.py` |
| Private files | `_leading_snake_case` | `_persistence.py`, `_plotting.py` |
| Classes | `PascalCase` | `BaseUncertaintyModel`, `EnsembleDecomposition` |
| Methods / Functions | `snake_case` | `pinball_loss()`, `materialize_lazyframe()` |
| Private methods | `_leading_underscore` | `_fit_ensemble()`, `_warn_if_far()` |
| Type aliases | `PascalCase` | `PolarsInput`, `TargetSpec`, `CopulaFamilyUnion` |
| Enums | `PascalCase(str, Enum)` with `UPPER_CASE` members | `CalibrationMethod.HOLDOUT` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_QUANTILES`, `MAX_SAMPLE_CHUNK_SIZE` |
| Exception classes | `PascalCase` ending `Error` | `ModelNotFittedError`, `InvalidDataError` |
| Warning classes | `PascalCase` ending `Warning` | `UncertaintyFlowWarning` |
| Fitted attributes | `trailing_underscore_` | `self._fitted`, `self._feature_cols_` (public fit-time attr) |
| Internal state | `_leading_underscore` | `self._rng`, `self._ensemble_models` |
| CLI commands/functions | `snake_case` | `list_datasets_cmd`, `download_all()` |
| CLi long options | `--kebab-case` | `--no-auto-tune`, `--target-coverage` |

### Quarto/YAML

| Element | Convention | Example |
|---------|-----------|---------|
| Notebook files | `NN_topic.qmd` | `01_quick_start.qmd`, `05_benchmarks.qmd` |
| YAML config | `snake_case` keys | `execute-results`, `project-cache` |

## Python Patterns

### Module Structure
- Docstring at top: `"""Brief description of module purpose."""`
- `from __future__ import annotations` for deferred evaluation
- `TYPE_CHECKING` guard for type-only imports to avoid circular deps
- `if TYPE_CHECKING: pass` used as a no-op placeholder when no type-only imports are needed yet
- Explicit `__all__` in every subpackage `__init__.py`
- Optional deps: try/except import, fallback to `None`, boolean `_available` flag
- `__init__.py` re-exports from submodules — consumers import from top-level: `from uncertainty_flow import ConformalRegressor`

### ABC-based Model Hierarchy
- `BaseUncertaintyModel(ABC)` — abstract `fit(data, target=None)`, `predict(data)`
- Concrete subclasses implement backend-specific logic
- Intermediate ABCs when sharing across models: `BaseQuantileNeuralNet(BaseUncertaintyModel)`
- `predict()` always returns `DistributionPrediction`

### Pydantic Config
- `BaseSettings` subclass with `SettingsConfigDict(env_prefix="UNCERTAINTY_FLOW_")`
- `@field_validator` classmethods for cross-field validation
- Global singleton via `get_config()` / `set_config()` / `reset_config()`

### Data Handling
- **I/O**: Polars DataFrame/LazyFrame (type alias: `PolarsInput`)
- **Internal**: NumPy arrays for performance
- **Bridge**: `materialize_lazyframe()` to collect, `to_numpy()` / `to_numpy_series()` to convert
- **Target spec**: `TargetSpec = str | list[str]`

### DistributionPrediction
- Universal prediction output consumed by all metrics and visualizations
- Internal: NumPy array (`quantile_matrix`)
- External: Polars methods (`.quantile()`, `.interval()`, `.median()`)
- Constructor validates finiteness, shape, non-emptiness
- Supports multivariate via `target_names` list

### Error Handling
- Exception hierarchy rooted at `UncertaintyFlowError(ValueError)`
- Error codes: `UF-E001` through `UF-E004`
- Warning: `UncertaintyFlowWarning(UserWarning)`
- `pytest.warns()` in tests for warning assertions

### CLI (Click)
- `@click.group()` + `@cli.command()` for subcommands
- `click.option` with both short (`-d`) and long (`--dataset`) forms
- `click.echo()` for output, `click.echo(..., err=True)` for errors
- `sys.exit(1)` on error
- `logger.exception()` before `sys.exit` for full traceback in logs
- `if __name__ == "__main__": main()` guard at bottom of `cli.py`

### Dataclasses
- `@dataclass` used for benchmark/results config (e.g., `BenchmarkConfig`, `BenchmarkResult`, `TuningResult` in `benchmarking/runner.py`)
- `from dataclasses import dataclass, field` — mutable defaults via `field(default_factory=...)`

### Metric Functions
- Standalone functions (not methods): `coverage_score(y, lower, upper)`, `pinball_loss(y, q, level)`, `winkler_score(...)`
- Return `float` for univariate, `dict[str, float]` for multivariate
- Unified dispatch via `metrics.score(pred, y_true, metric="crps")`
- `_METRIC_NAMES` set for validation

### Optional Dependency Pattern
```python
try:
    from .some_optional_import import SomeClass
    _available = True
except ImportError:
    SomeClass = None
    _available = False
```

### Soft-private trailing underscore convention for fitted attributes
Attributes set by `fit()` use a trailing underscore (scikit-learn convention):
`self._fitted`, `self._feature_cols_`, `self.intercept_`, `self.slope_`

## Testing (pytest)

- **Runner**: `uv run pytest tests/` from project root
- **Framework**: pytest 9.x with `--import-mode=importlib`
- **Coverage**: pytest-cov with `--cov=uncertainty_flow`, `--cov-fail-under=40`, branch coverage
- **File naming**: `test_<module_name>.py` (mirrors source file name)
- **Directory structure**: `tests/` mirrors `uncertainty_flow/` — `tests/core/`, `tests/metrics/`, `tests/wrappers/`, etc.
- **Test organization**: class-based — `class Test<ClassName>` with `def test_<behavior>(self)` methods
- **Descriptive test names**: `test_init_with_required_params`, `test_rejects_zero_bootstrap`
- **Shared fixtures**: `conftest.py` at `tests/` root (`time_series_data`, `univariate_time_series`)
- **Local fixtures**: defined at top of test files (inside the test class or module-level `@pytest.fixture`), used by that file's test classes
- **Mocking**: minimal; prefers real lightweight model stubs (e.g., `LinearBootstrapToyModel`) over mocks
- **Approximations**: `pytest.approx()` for float comparisons
- **Exception testing**: `pytest.raises()` with `match=` regex
- **Warning testing**: `pytest.warns()` for expected warnings
- **No async tests** — all tests are synchronous
- **Markers**: auto-applied via `pytest_collection_modifyitems` in `conftest.py`:
  - `unit` (default for `core/`, `utils/`)
  - `integration` (models, wrappers, analysis, causal, etc.)
  - `slow` (conformal_ts, counterfactual, numpyro, torch, persistence)
  - `optional` (torch, numpyro — depend on optional deps)
  - `network` (internet-required)
  - `smoke` (critical path for CI gating)
- **CI test command**: `uv run pytest tests/ --ignore=tests/models/test_deep_quantile_torch.py -q` (runs across 3.11, 3.12, 3.13)
- **Coverage exclusions**: 5 source files omitted from coverage (`deep_quantile_torch.py`, `transformer_forecaster.py`, `numpyro_model.py`, `shap_values.py`, `dashboard.py`)

## Linting & Formatting

### Python
- **Ruff**: `line-length = 100`, target `py311`, rules `E,F,I,N,W`
- **pre-commit hooks** (see `.pre-commit-config.yaml`):
  - `ruff --fix` + `ruff-format` (lints all Python except `notebooks/`)
  - `mypy uncertainty_flow/` (type checking)
  - `pip-audit --skip-editable` (dependency vulnerability scan)
  - `uv-lock` (lockfile consistency)
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `check-merge-conflict`
- **Bandit**: `bandit -r uncertainty_flow` (excludes tests/)
- **Mypy**: `python_version = 3.11`, `warn_return_any = true`, `ignore_missing_imports = true`

## Build/Dev Commands

```
uv sync                         → Install all dependencies
uv sync --extra dev             → Install dev extras (testing, linting)
uv sync --extra ml              → Install ML extras (torch, numpyro, chronos)
uv run pytest tests/            → Run full test suite
uv run pytest tests/ -m unit    → Run only unit tests
uv run pytest tests/ -m integration  → Run only integration tests
uv run mypy uncertainty_flow/   → Type-check source code
uv run ruff check .             → Lint all files
uv run ruff format . --check    → Check formatting
uv run ruff format .            → Auto-format
uv run bandit -r uncertainty_flow  → Security scan
make notebooks                  → Render all Quarto notebooks
make docs                       → Build full docs site
make docs-preview               → Serve docs locally with hot-reload
uv run uncertainty-flow <cmd>   → Run CLI benchmark tool
uv run quarto render notebooks/ → Render notebooks individually
uv run python scripts/generate_notebook_docs.py → Generate notebook doc stubs
uv run pip-audit --skip-editable → Check dependency vulnerabilities
pre-commit run --all-files      → Run all pre-commit hooks manually
```
