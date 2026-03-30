# CHANGELOG.md

All notable changes to `uncertainty_flow` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project scaffold and package structure
- `BaseUncertaintyModel` abstract base class
- `DistributionPrediction` core output object with `.quantile()`, `.interval()`, `.mean()`, `.plot()`
- `ConformalRegressor` — tabular conformal wrapper for any sklearn estimator
- `ConformalForecaster` — temporal-aware conformal wrapper with multivariate support
- `QuantileForestForecaster` — quantile regression forest with leaf distribution storage
- `DeepQuantileNet` — multi-quantile MLP with shared trunk (sklearn backend)
- Polars I/O layer with LazyFrame support (`utils/polars_bridge.py`)
- Holdout and cross-conformal calibration split strategies
- Residual correlation analysis for uncertainty driver detection (`uncertainty_drivers_`)
- Gaussian copula for multivariate joint intervals
- Calibration report (Polars DataFrame) with pinball loss, Winkler score, coverage
- Warning system with `UF-W` / `UF-E` codes
- Standalone metrics: `pinball_loss`, `winkler_score`, `coverage_score`
- `uv`-based dev setup
- Full test suite with pytest

---

## Release Notes Template

```
## [X.Y.Z] — YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality (non-breaking)

### Deprecated
- Features that will be removed in a future version

### Removed
- Features removed in this release

### Fixed
- Bug fixes

### Security
- Security-relevant changes
```
