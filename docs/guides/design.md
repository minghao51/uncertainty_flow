# Design Notes

This file keeps the durable design decisions that are still useful when reading or extending the codebase. It is intentionally narrower than the architecture doc and should avoid repeating package inventories or historical implementation snapshots.

## Core Principles

1. Distribution-first output. Public prediction APIs return `DistributionPrediction`, not raw point forecasts.
2. Honest guarantees. Docs must distinguish clearly between mathematically calibrated intervals and empirical uncertainty estimates.
3. Polars at the boundary, NumPy in the core. DataFrames are ergonomic for callers; arrays keep the implementation interoperable with scientific Python tooling.
4. Shared downstream shape. Different model families should feel interchangeable once they reach prediction output.

## Decision Log

### `D-001` Distribution-first API

`DistributionPrediction` is the primary output contract. That keeps model families interchangeable and makes interval, quantile, sampling, and plotting workflows consistent.

### `D-002` Polars I/O, NumPy internals

The package accepts Polars DataFrames and LazyFrames, then centralizes conversion inside `utils/polars_bridge.py`. This prevents ad hoc conversion logic from leaking across the codebase.

### `D-003` Calibration strategy follows data shape

Tabular workflows default to random holdout calibration. Forecasting workflows use temporal holdout. This avoids leaking future information while keeping calibration behavior predictable.

### `D-004` Dependence is modeled separately from marginals

Multivariate uncertainty is handled by combining per-target marginals with a copula layer. That keeps target-level prediction code independent from joint dependence modeling and allows multiple copula families to coexist.

### `D-005` Diagnostics are first-class

Calibration reports and uncertainty-driver analysis are part of the public workflow, not hidden implementation details. The package should help users inspect interval quality, not just generate intervals.

## What Does Not Belong Here

Move content elsewhere when it becomes one of these:

- package structure or file inventory: [../architecture/overview.md](../architecture/overview.md)
- model-by-model guarantee comparisons: [./models.md](./models.md)
- future priorities or speculative work: [../project/roadmap.md](../project/roadmap.md)
- historical implementation snapshots: [../archive/README.md](../archive/README.md)
