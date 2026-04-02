# Roadmap

This file should describe future work, not already-shipped functionality.

## Current Focus

- tighten and stabilize the public API surface ahead of wider release
- improve docs so the maintained guides, architecture notes, and API spec stay aligned
- expand verification around newer optional components such as benchmarking, transformers, and torch-backed models

## Near-Term Priorities

### Release hardening

- improve test coverage for optional dependency paths
- verify CLI and benchmarking flows against current docs
- reduce drift between `README.md`, `docs/`, and public exports

### Documentation quality

- keep one canonical explanation per topic instead of parallel guides
- continue moving historical implementation snapshots into `docs/archive/`
- align model capability docs with optional imports and runtime behavior

### Model and evaluation depth

- strengthen benchmarking coverage across more datasets and failure modes
- improve guidance for choosing between empirical and calibrated workflows
- refine multivariate evaluation and calibration diagnostics

## Longer-Term Ideas

| Idea | Notes |
|---|---|
| Classification uncertainty | Prediction sets and conformal classification flows |
| Interactive calibration dashboard | Richer visual diagnostics for interval quality |
| Ensemble uncertainty workflows | Model disagreement as an additional uncertainty signal |
| Bayesian or posterior-based models | Higher-cost, richer uncertainty estimation |
| Streaming or async inference | Useful for serving contexts, but not a current priority |

## Completed Work

Features that have already landed should stay in the changelog or archive, not in this roadmap. See [./changelog.md](./changelog.md) and [../archive/README.md](../archive/README.md) for historical context.
