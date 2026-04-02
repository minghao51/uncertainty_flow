# Codebase Concerns

## Overview
This document outlines technical debt, bugs, security issues, and performance concerns identified in the uncertainty_flow codebase as of 2026-03-22.

**Last updated**: 2026-03-31 - Major refactoring completed (see Resolved Issues section)

## Priority Levels
- 🔴 **CRITICAL**: Must address immediately
- 🟡 **HIGH**: Should address soon
- 🔵 **MEDIUM**: Address when possible
- ⚪ **LOW**: Nice to have improvements
- ✅ **RESOLVED**: Issue has been addressed

---

## Resolved Issues

### ✅ Memory management in sampling
**Status**: RESOLVED (2026-03-31)
- Implemented chunked sampling with `MAX_SAMPLE_CHUNK_SIZE` (100K) and `MAX_TOTAL_SAMPLES` (10M) limits
- Added input validation for `n` parameter in `DistributionPrediction.sample()`
- Chunked sampling via `_sample_chunked()` method prevents memory exhaustion

### ✅ LazyFrame materialization
**Status**: RESOLVED (2026-03-31)
- Minimized `.collect()` calls in `ConformalRegressor.fit()` - collects once at start
- Optimized `polars_bridge.to_numpy()` to select and convert in single operation
- All model fit/predict methods now collect LazyFrame once at entry point

### ✅ Inefficient array operations
**Status**: RESOLVED (2026-03-31)
- Vectorized `DeepQuantileNet._predict_backend()` using matrix multiplication
- Replaced manual loop with `trunk_features @ coef_matrix + intercepts`

### ✅ Redundant sorting
**Status**: RESOLVED (2026-03-31)
- `BaseQuantileNeuralNet.predict()` now checks if sorting is needed before applying
- `_ensure_monotonicity()` uses vectorized `np.sort()` instead of row-by-row loop

### ✅ Circular dependency risk
**Status**: RESOLVED (2026-03-31)
- Added explicit documentation of circular dependency rationale in `base.py`
- Lazy import pattern is intentional and documented

### ✅ Exception swallowing
**Status**: RESOLVED (2026-03-31)
- Replaced all bare `except Exception:` with specific exception types:
  - `copula.py`: `(ValueError, OverflowError, ZeroDivisionError, np.linalg.LinAlgError)`
  - `shap_values.py`: `(ValueError, RuntimeError, np.linalg.LinAlgError)`

### ✅ Pytest version conflicts
**Status**: RESOLVED (2026-03-31)
- Standardized on `pytest>=9.0.2` in both dependencies and dev

### ✅ Input validation
**Status**: RESOLVED (2026-03-31)
- Added `random_state` validation in `BaseQuantileNeuralNet`
- Added `n` parameter validation in `DistributionPrediction.sample()`
- All public APIs now validate inputs with proper error messages

### ✅ Multivariate plotting
**Status**: RESOLVED (2026-03-31)
- Fixed `DistributionPrediction.plot()` to handle multivariate input correctly
- Properly extracts target-specific columns for multivariate cases

### ✅ Magic numbers
**Status**: RESOLVED (2026-03-31)
- Extracted constants: `MAX_SAMPLE_CHUNK_SIZE`, `MAX_TOTAL_SAMPLES`, `PLOT_MAX_SAMPLES`
- Defined at module level in `distribution.py`

### ✅ LRU cache for quantile lookups
**Status**: RESOLVED (2026-03-31)
- Added `@lru_cache(maxsize=128)` to `_find_nearest_quantile_index()`

---

## Security Issues

### 🔴 CRITICAL
**Status**: ✅ None identified

**Good practices observed**:
- No hardcoded secrets
- No exposed API keys or credentials
- No SQL injection vectors (Polars parameterized queries)

### 🟡 MEDIUM
**Input validation gaps**:
- Several modules lack comprehensive input validation
- No validation for malicious inputs in edge cases
- Missing bounds checking for numerical parameters

**Example locations**:
- `DistributionPrediction.sample()` - No validation for large `n` values
- `BaseSplit._validate_calibration_size()` - Only validates size, not data quality

**Recommendation**: Add input validation to all public APIs

### 🟡 MEDIUM
**No authentication/authorization framework**:
- If web API is planned, no auth framework in place
- No rate limiting or access control

**Recommendation**: Plan for authentication if adding web interface

---

## Performance Issues

### 🔴 CRITICAL
**Memory management in sampling**:
- `DistributionPrediction.sample()` method has no memory management
- Large `n` values can cause memory exhaustion
- No chunking or streaming for large samples

**Location**: `uncertainty_flow/core/distribution.py`

**Impact**: Can crash with large sample requests

**Recommendation**: Implement chunked sampling with memory limits

### 🔴 CRITICAL
**LazyFrame materialization**:
- Multiple `.collect()` calls in `fit()` and `predict()` methods
- Repeated materialization of same data
- No caching of collected DataFrames

**Locations**:
- `ConformalRegressor.fit()` - Multiple collects
- `DeepQuantileNet.predict()` - Repeated materialization

**Impact**: Poor performance with large datasets

**Recommendation**: Cache collected DataFrames, minimize collects

### 🟡 HIGH
**Inefficient array operations**:
- Manual matrix multiplication loops in `DistributionPrediction._extract_trunk_features()`
- Should use vectorized NumPy operations

**Location**: `uncertainty_flow/core/distribution.py`

**Impact**: Slower than necessary for large datasets

**Recommendation**: Replace with NumPy vectorized operations

### 🟡 HIGH
**Redundant sorting**:
- `DeepQuantileNet.predict()` sorts quantile_matrix twice
- Unnecessary computational overhead

**Location**: `uncertainty_flow/models/deep_quantile.py`

**Impact**: 2x sorting overhead

**Recommendation**: Remove redundant sort operation

### 🔵 MEDIUM
**No caching for quantile lookups**:
- Repeated quantile level lookups not cached
- Same quantiles computed multiple times

**Recommendation**: Implement LRU cache for quantile computations

---

## Technical Debt

### 🟡 HIGH
**Circular dependency risk**:
- `BaseUncertaintyModel.calibration_report()` has comment: "# Import here to avoid circular dependency"
- Lazy import of calibration module

**Location**: `uncertainty_flow/core/base.py:74`

**Impact**: Code smell, potential for actual circular dependencies

**Recommendation**: Refactor to eliminate circular dependency

### 🟡 HIGH
**Mixed architecture patterns**:
- sklearn backend models vs torch models without unified interface
- Inconsistent patterns across backends

**Locations**:
- `DeepQuantileNet` (sklearn)
- `DeepQuantileNetTorch` (torch)

**Impact**: Inconsistent user experience, harder to maintain

**Recommendation**: Define clear interface contract, enforce across backends

### 🟡 HIGH
**Hardcoded default quantiles**:
- `DEFAULT_QUANTILES` defined in multiple files
- No centralized configuration

**Locations**:
- `uncertainty_flow/core/types.py`
- Various model files

**Impact**: Inconsistency risk, hard to change defaults

**Recommendation**: Single source of truth for defaults

### 🟡 HIGH
**Inconsistent error handling**:
- Different patterns across modules
- Some raise exceptions, others return None
- No standard error types

**Impact**: Unpredictable error behavior

**Recommendation**: Define error handling standard, implement consistently

### 🔵 MEDIUM
**Documentation gaps**:
- Missing type hints for some function parameters
- Incomplete docstrings for private methods
- No examples for some complex functions

**Recommendation**: Complete type hints and docstrings

### 🔵 MEDIUM
**Test coverage gaps**:
- Limited edge case testing for multivariate scenarios
- No tests for wrappers (`conformal.py`, `conformal_ts.py`)
- No tests for calibration module

**Recommendation**: Increase test coverage to >80% across all modules

### ⚪ LOW
**Code duplication**:
- Similar validation logic repeated across multiple classes
- Duplicated DataFrame manipulation patterns

**Recommendation**: Extract common utilities

---

## Bugs & Issues

### 🟡 HIGH
**Index handling inconsistency**:
- `DistributionPrediction.plot()` assumes single target
- Doesn't handle multivariate input correctly

**Location**: `uncertainty_flow/core/distribution.py`

**Impact**: Plotting fails or produces incorrect output for multivariate

**Recommendation**: Fix index handling for multivariate case

### 🟡 HIGH
**Exception swallowing**:
- Bare `except Exception:` block in `ConformalRegressor` (line 161)
- Catches and suppresses all exceptions

**Location**: `uncertainty_flow/wrappers/conformal.py:161`

**Impact**: Hides real errors, makes debugging impossible

**Recommendation**: Catch specific exceptions only

### 🟡 HIGH
**Type inconsistency**:
- Methods return different types (Series vs DataFrame) without clear documentation
- Unclear what return type to expect

**Impact**: Confusing API, potential type errors

**Recommendation**: Document and standardize return types

### 🔵 MEDIUM
**Magic numbers**:
- Hardcoded values like `500` samples for plotting without explanation
- No constants defined for magic numbers

**Example**: `DistributionPrediction.plot()` uses 500 samples

**Impact**: Code is less maintainable

**Recommendation**: Extract named constants with documentation

### 🔵 MEDIUM
**Import organization**:
- Some files import after function definitions instead of at top
- Inconsistent import ordering

**Impact**: Code style inconsistency

**Recommendation**: Enforce consistent import ordering via ruff

### 🔵 MEDIUM
**Missing validation**:
- `random_state` not consistently validated across models
- Some models accept any value, others require specific types

**Impact**: Inconsistent behavior, potential runtime errors

**Recommendation**: Standardize random_state validation

---

## Architecture Concerns

### 🔴 CRITICAL
**Tight coupling**:
- Models import metrics directly
- Metrics depend on DistributionPrediction structure
- Hard to change one without affecting other

**Impact**: Difficult to evolve independently

**Recommendation**: Introduce interfaces/protocols for decoupling

### 🔴 CRITICAL
**Missing abstraction for I/O formats**:
- No abstraction layer for Polars, NumPy, Pandas
- Direct conversions scattered throughout
- Hard to support additional formats

**Impact**: Difficult to add new data format support

**Recommendation**: Create abstract data format interface

### 🔴 CRITICAL
**Single responsibility violation**:
- `DistributionPrediction` handles storage, computation, AND visualization
- Too many responsibilities in one class

**Location**: `uncertainty_flow/core/distribution.py`

**Impact**: Hard to test, maintain, and extend

**Recommendation**: Split into separate classes for storage, computation, visualization

### 🟡 HIGH
**Version dependency conflicts**:
- `pyproject.toml` shows conflicting pytest versions:
  - `dependencies`: `pytest>=9.0.2`
  - `dev`: `pytest>=7.4.0`

**Location**: `pyproject.toml:12,18`

**Impact**: Unresolved dependency version

**Recommendation**: Resolve to single pytest version requirement

### 🟡 HIGH
**Package structure unclear**:
- Unclear separation between core functionality and extensions
- Hard to identify what's essential vs. optional

**Impact**: Confusing for contributors

**Recommendation**: Document package structure, clarify core vs. extensions

### 🟡 HIGH
**No centralized configuration**:
- Configuration scattered across modules
- No global settings mechanism

**Impact**: Hard to configure behavior globally

**Recommendation**: Implement configuration system

---

## Recommended Actions

### Priority 1 (Critical) - Address Immediately

1. **Fix memory management** in sampling methods
   - Implement chunked sampling
   - Add memory limits

2. **Address circular dependency** in calibration report
   - Refactor import structure
   - Eliminate lazy imports

3. **Implement proper input validation** across all modules
   - Validate all public API inputs
   - Add comprehensive error messages

4. **Resolve pytest version conflicts**
   - Standardize on single pytest version
   - Update pyproject.toml

### Priority 2 (High) - Address Soon

1. **Standardize error handling patterns**
   - Define error type hierarchy
   - Document error handling strategy
   - Fix exception swallowing in ConformalRegressor

2. **Remove duplicate sorting operations**
   - Audit all sorting operations
   - Remove redundant sorts

3. **Fix index handling in multivariate plotting**
   - Support multivariate plotting correctly
   - Add tests for multivariate case

4. **Consolidate quantile level configuration**
   - Single source of truth
   - Centralized configuration

5. **Minimize LazyFrame materialization**
   - Cache collected DataFrames
   - Reduce redundant .collect() calls

### Priority 3 (Medium) - Address When Possible

1. **Add comprehensive type hints**
   - Complete missing type annotations
   - Run type checker (mypy)

2. **Implement caching for repeated computations**
   - Quantile level lookups
   - Expensive computations

3. **Improve documentation for return types**
   - Document all return types
   - Add examples

4. **Increase test coverage**
   - Add tests for wrappers
   - Add tests for calibration
   - Edge case coverage

5. **Replace inefficient array operations**
   - Vectorize with NumPy
   - Profile and optimize hot paths

---

## Files Requiring Immediate Attention

### 🔴 Critical
1. `uncertainty_flow/core/distribution.py` - Performance and memory issues
2. `uncertainty_flow/models/deep_quantile.py` - Architecture coupling
3. `uncertainty_flow/wrappers/conformal.py` - Exception handling (line 161)
4. `pyproject.toml` - Dependency conflicts

### 🟡 High Priority
1. `uncertainty_flow/core/base.py` - Circular dependency
2. `uncertainty_flow/models/deep_quantile_torch.py` - Inconsistent patterns
3. `uncertainty_flow/core/types.py` - Default configuration

---

## Monitoring Areas

### Performance Metrics
- **Memory usage** during sampling operations
- **Execution time** for large datasets
- **LazyFrame materialization** count
- **Cache hit rates** (when caching implemented)

### Code Quality Metrics
- **Test coverage** per module
- **Type coverage** (percentage of code with type hints)
- **Code duplication** percentage
- **Cyclomatic complexity** per function

### Technical Debt Metrics
- **Number of TODO/FIXME comments**
- **Number of open issues**
- **Code churn** in frequently modified files
- **Bug fix frequency** per module

---

## Debt Reduction Strategy

### Short-term (1-2 weeks)
1. Fix critical bugs and security issues
2. Resolve dependency conflicts
3. Add input validation to public APIs

### Medium-term (1-2 months)
1. Refactor circular dependencies
2. Standardize error handling
3. Improve test coverage to >80%

### Long-term (3-6 months)
1. Architectural refactoring for better separation of concerns
2. Performance optimization
3. Comprehensive documentation

---

## Related Documentation
- **ARCHITECTURE.md**: System architecture and design patterns
- **CONVENTIONS.md**: Code style and development practices
- **TESTING.md**: Testing strategy and coverage goals
