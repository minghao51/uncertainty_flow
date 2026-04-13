# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Helper function**: Added `materialize_lazyframe()` utility in `uncertainty_flow/utils/polars_bridge.py` for consistent LazyFrame materialization across the codebase
- **Zero-copy conversion tests**: Added `tests/test_utils.py` (160 lines) with comprehensive tests for zero-copy Polars conversion utilities

### Changed

- **Polars API usage**: Updated `df.sort()` calls to use correct positional arguments syntax (`df.sort(by, *more_by)`) instead of incorrect `by=` parameter usage. Location: `scripts/generate_report.py:59`

### Fixed

- **Critical bug in `MovingAverageBenchmark`**: Fixed incorrect z-score used for 90% prediction interval upper bound. The interval was using `z_80` (1.28) instead of `z_90` (1.645), making the interval incorrectly narrow. Location: `scripts/comprehensive_benchmark.py:246`
- **R-hat calculation edge case**: Fixed edge case handling for near-zero within-chain variance in convergence diagnostics

### Breaking Changes

#### Type System Migration (Literal → Enum)

The following type definitions have been changed from `Literal` to `str` Enum for improved type safety, IDE autocomplete, and self-documentation:

- **`CalibrationMethod`** (formerly `CalibrationMethodLiteral`)
  - Before: `Literal["holdout", "cross"]`
  - After: `CalibrationMethod` Enum with `HOLDOUT` and `CROSS` members
  - Location: `uncertainty_flow/core/types.py`

- **`CorrelationMode`** (formerly `CorrelationModeLiteral`)
  - Before: `Literal["auto", "independent"]`
  - After: `CorrelationMode` Enum with `AUTO` and `INDEPENDENT` members
  - Location: `uncertainty_flow/core/types.py`

- **`CopulaFamily`** (formerly `CopulaFamilyLiteral`)
  - Before: `Literal["gaussian", "clayton", "gumbel", "frank", "auto"] | type["BaseCopula"]`
  - After: `CopulaFamily` Enum with `GAUSSIAN`, `CLAYTON`, `GUMBEL`, `FRANK`, and `AUTO` members
  - Location: `uncertainty_flow/multivariate/copula.py`

**Migration Guide:**

1. **String values remain compatible:** All existing code passing string values (e.g., `"holdout"`, `"gaussian"`) will continue to work as Enums are `str` subclasses.

2. **Type annotations:** If you have type annotations using the old `Literal` types, update to the new Enum names:
   ```python
   # Before
   from uncertainty_flow.core.types import CalibrationMethodLiteral
   def method(m: CalibrationMethodLiteral) -> None: ...

   # After
   from uncertainty_flow.core.types import CalibrationMethod
   def method(m: CalibrationMethod) -> None: ...
   ```

3. **Backward compatibility aliases:** The old `Literal` type aliases are preserved for gradual migration:
   - `CalibrationMethodLiteral` (available in `uncertainty_flow/core/types.py`)
   - `CorrelationModeLiteral` (available in `uncertainty_flow/core/types.py`)
   - `CopulaFamilyLiteral` (available in `uncertainty_flow/multivariate/copula.py`)

4. **Usage patterns:**
   ```python
   # All of these continue to work:
   from uncertainty_flow.core.types import CalibrationMethod

   # String values (backward compatible)
   method = "holdout"

   # Enum members (recommended for new code)
   method = CalibrationMethod.HOLDOUT
   ```

## [1.0.1] - 2026-04-06

### Improved

- **Error handling**: Enhanced error handling across all prediction APIs (23 files modified) with better validation and error messages
- **Model validation**: Added improved input validation in all model implementations
- **Edge case handling**: Better handling of edge cases in distribution calculations and conformal prediction
- **Type safety**: Enhanced type annotations across the core modules

### Added

- **Package typing**: Added `uncertainty_flow/py.typed` for improved PEP 561 type checking support
- **CI/CD pipeline**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) for automated testing and quality checks
- **Test fixtures**: Enhanced test fixtures with parameterized confidence levels and sample sizes

### Changed

- **Planning documentation**: Refreshed `.planning/codebase/` documentation files (3,051 lines changed across ARCHITECTURE.md, CONCERNS.md, CONVENTIONS.md, INTEGRATIONS.md, STACK.md, STRUCTURE.md, TESTING.md)

## [1.1.0] - 2026-04-03

### Added

- **Bayesian Inference Module**: Added `uncertainty_flow/bayesian/` with NumPyro integration for Bayesian uncertainty quantification
  - `bayesian/numpyro_model.py` - NumPyro-based Bayesian models
  - `tests/bayesian/test_numpyro_model.py` (236 lines)

- **Causal Analysis Module**: Added `uncertainty_flow/causal/` for causal uncertainty estimation
  - `causal/estimator.py` - Causal uncertainty estimators
  - `tests/causal/test_estimator.py` (299 lines)

- **Multimodal Support**: Added `uncertainty_flow/multimodal/` with aggregator for handling multiple data types
  - `multimodal/aggregator.py` - Multi-modal data aggregation
  - `tests/multimodal/test_aggregator.py` (248 lines)

- **Benchmarking Framework**: Added `uncertainty_flow/benchmarking/` with standardized evaluation
  - `benchmarking/datasets.py` - Benchmark datasets
  - `benchmarking/runner.py` - Benchmark execution engine
  - `benchmarking/tuning.py` - Hyperparameter tuning utilities

- **Command-Line Interface**: Added `uncertainty_flow/cli.py` (506 lines) for programmatic access to all library features

- **Auto-Tuning Utilities**: Added `uncertainty_flow/utils/auto_tuning.py` (109 lines) for automated model optimization

- **Enhanced Distribution Module**: Expanded `core/distribution.py` with 321 additional lines of enhanced distribution methods

- **Quantile Forest Model**: Enhanced `quantile_forest.py` with additional features and improvements

### Improved

- **Documentation Structure**: Complete overhaul with comprehensive guides
  - `docs/guides/benchmarking.md` (299 lines)
  - `docs/guides/charting.md` (173 lines)
  - `docs/guides/distribution-approach.md` (83 lines)
  - Benchmark documentation with comprehensive results
  - Technical roadmap (795 lines)

- **Test Coverage**: Added extensive test suites for all new modules
  - `tests/test_package_integration.py` (97 lines)

- **Benchmark Scripts**: Added comprehensive evaluation tools
  - `scripts/comprehensive_benchmark.py` (453 lines)
  - `scripts/generate_report.py` (294 lines)

### Changed

### Breaking Changes

#### Type System Migration (Literal → Enum)

The following type definitions have been changed from `Literal` to `str` Enum for improved type safety, IDE autocomplete, and self-documentation:

- **`CalibrationMethod`** (formerly `CalibrationMethodLiteral`)
  - Before: `Literal["holdout", "cross"]`
  - After: `CalibrationMethod` Enum with `HOLDOUT` and `CROSS` members
  - Location: `uncertainty_flow/core/types.py`

- **`CorrelationMode`** (formerly `CorrelationModeLiteral`)
  - Before: `Literal["auto", "independent"]`
  - After: `CorrelationMode` Enum with `AUTO` and `INDEPENDENT` members
  - Location: `uncertainty_flow/core/types.py`

- **`CopulaFamily`** (formerly `CopulaFamilyLiteral`)
  - Before: `Literal["gaussian", "clayton", "gumbel", "frank", "auto"] | type["BaseCopula"]`
  - After: `CopulaFamily` Enum with `GAUSSIAN`, `CLAYTON`, `GUMBEL`, `FRANK`, and `AUTO` members
  - Location: `uncertainty_flow/multivariate/copula.py`

**Migration Guide:**

1. **String values remain compatible:** All existing code passing string values (e.g., `"holdout"`, `"gaussian"`) will continue to work as Enums are `str` subclasses.

2. **Type annotations:** If you have type annotations using the old `Literal` types, update to the new Enum names:
   ```python
   # Before
   from uncertainty_flow.core.types import CalibrationMethodLiteral
   def method(m: CalibrationMethodLiteral) -> None: ...

   # After
   from uncertainty_flow.core.types import CalibrationMethod
   def method(m: CalibrationMethod) -> None: ...
   ```

3. **Backward compatibility aliases:** The old `Literal` type aliases are preserved for gradual migration:
   - `CalibrationMethodLiteral` (available in `uncertainty_flow/core/types.py`)
   - `CorrelationModeLiteral` (available in `uncertainty_flow/core/types.py`)
   - `CopulaFamilyLiteral` (available in `uncertainty_flow/multivariate/copula.py`)

4. **Usage patterns:**
   ```python
   # All of these continue to work:
   from uncertainty_flow.core.types import CalibrationMethod

   # String values (backward compatible)
   method = "holdout"

   # Enum members (recommended for new code)
   method = CalibrationMethod.HOLDOUT
   ```

### Fixed

- **Critical bug in `MovingAverageBenchmark`**: Fixed incorrect z-score used for 90% prediction interval upper bound. The interval was using `z_80` (1.28) instead of `z_90` (1.645), making the interval incorrectly narrow. Location: `scripts/comprehensive_benchmark.py:246`

### Changed

- **Polars API usage**: Updated `df.sort()` calls to use correct positional arguments syntax (`df.sort(by, *more_by)`) instead of incorrect `by=` parameter usage. Location: `scripts/generate_report.py:59`

### Added

- **Helper function**: Added `materialize_lazyframe()` utility in `uncertainty_flow/utils/polars_bridge.py` for consistent LazyFrame materialization across the codebase
