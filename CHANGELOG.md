# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
