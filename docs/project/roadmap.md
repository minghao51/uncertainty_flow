# Roadmap

This file should describe future work, not already-shipped functionality.

For the full prioritized technical roadmap with implementation details, see [./technical_roadmap.md](./technical_roadmap.md).

## Current Focus

- tighten and stabilize the public API surface ahead of wider release
- improve docs so the maintained guides, architecture notes, and API spec stay aligned
- expand verification around newer optional components such as benchmarking, transformers, and torch-backed models

## Near-Term Priorities (Phase 1)

- **Quantile-based CRPS** — replace Gaussian approximation with exact quantile CRPS
- **Remove dead scipy import** in `distribution.py:sample()`
- **Unified metric API** — single `score()` entry point for all metrics
- **`.summary()` on `DistributionPrediction`** — one-call overview
- **Multivariate plot facets** — fix silent first-target-only plotting

See [./technical_roadmap.md#phase-1--fix-whats-broken-v020](./technical_roadmap.md) for details.

## Mid-Term Priorities (Phase 2)

- PIT histograms and continuous calibration curves
- Isotonic recalibration wrapper (close the calibration loop)
- Sequential conformal updates (ACI) for non-stationary data
- Rolling-origin and sliding-window time series evaluation
- Non-crossing quantile training

See [./technical_roadmap.md#phase-2--core-statistical-rigor-v030](./technical_roadmap.md) for details.

## Longer-Term Ideas (Phases 3–4)

| Idea | Phase | Notes |
|---|---|---|
| Parametric distribution fitting | 3 | Normal, Student-t, LogNormal from quantiles |
| Log-score metric | 3 | Strictly proper scoring rule alongside CRPS |
| Energy score for multivariate | 3 | Proper multivariate scoring rule |
| Extended copula support (>2 targets) | 3 | Vine copulas or pairwise chaining |
| SHAP and leverage analysis in model API | 3 | Wire standalone modules into `BaseUncertaintyModel` |
| EnbPI for time series conformal | 4 | Bootstrap + sequential conformal |
| Conformal classification / prediction sets | 4 | APS prediction sets |
| Batch and streaming prediction API | 4 | Production serving support |
| Model comparison suite | 4 | Skill scores, Diebold-Mariano tests |
| Risk function integration | 4 | Wire `risk_functions.py` into `ConformalRiskControl` |
| Test coverage to 80%+ | 4 | From current 65% floor |

## Completed Work

Features that have already landed should stay in the changelog or archive, not in this roadmap. See [./changelog.md](./changelog.md) and [../archive/README.md](../archive/README.md) for historical context.
