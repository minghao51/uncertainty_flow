# Phase 4 Runtime Handoff: Leverage Analysis Performance

**Date:** 2026-04-11  
**Status:** Handoff  
**Owner:** Next implementation thread

---

## Summary

Phase 4 functionality is now materially in place:

- refit-based `EnsembleDecomposition`
- target-specific multivariate leverage analysis
- matching scoped tests

The main residual risk is runtime. `FeatureLeverageAnalyzer` is still fairly expensive, especially in the multivariate path where each feature is perturbed repeatedly and each perturbation triggers full-model prediction across all rows.

This handoff is for performance follow-up only. Behavior should be preserved unless a change is explicitly called out and tested.

---

## Current State

The expensive path is concentrated in:

- `uncertainty_flow/analysis/leverage.py`
- `tests/analysis/test_leverage.py`

Current behavior:

- univariate `analyze()` computes baseline interval widths, then perturbs each feature several times
- multivariate `analyze_multivariate()` repeats that work per target
- perturbation count is already soft-capped internally to keep tests from becoming unusable

The implementation is now correct enough for the scoped Phase 4 behavior, but it is not yet efficient enough to be a comfortable default for larger evaluation frames.

---

## Problem Statement

We want leverage analysis to remain:

1. simple
2. idiomatic
3. target-specific in multivariate mode
4. cheap enough for practical local analysis

Without:

- regressing current output columns
- introducing joint/copula leverage in this pass
- adding heavy abstraction or framework complexity

---

## Likely Runtime Drivers

1. Repeated full-frame prediction for every `(target, feature, perturbation)` combination.
2. Multivariate mode scales roughly with `n_targets * n_features * n_perturbations`.
3. Perturbation currently rebuilds full prediction outputs even when only width deltas are needed.
4. Test fixtures use moderate row counts, so slowdowns are visible even before real-world usage.

---

## Recommended Work Plan

### 1. Add lightweight benchmarking around leverage analysis

Add a small non-committed benchmark or test-adjacent timing helper that measures:

- univariate `analyze()` on a medium frame
- multivariate `analyze_multivariate()` on a comparable frame
- relative cost as `n_features`, `n_targets`, and `n_perturbations` increase

Goal: identify whether row count, perturbation count, or repeated target passes dominates.

### 2. Reduce duplicate prediction work

Focus on the simplest safe wins first:

- reuse baseline predictions and widths aggressively
- avoid rebuilding or recomputing intermediate structures more than needed
- centralize “extract interval widths for target” logic if any duplicate branches remain

Do not change public output shape.

### 3. Make perturbation cost proportional to value

Replace the current hard internal cap with a more explicit policy if helpful:

- keep `n_perturbations` user-facing
- derive an effective perturbation count from both the requested value and frame size
- document the policy clearly in the docstring if behavior becomes approximate

Preferred outcome: predictable runtime without surprising API behavior.

### 4. Consider row subsampling for leverage estimation

If runtime still dominates after low-risk cleanup, add an optional estimation parameter such as:

```python
max_samples_for_leverage: int | None = None
```

Behavior:

- if unset, preserve full-frame behavior
- if set and input is larger, analyze a deterministic subsample
- apply the same policy in univariate and multivariate paths

Only add this if profiling shows it materially helps.

### 5. Keep multivariate scope narrow

Do not add:

- joint leverage
- copula impact metrics
- new output columns

This thread should only improve runtime for the existing per-target output contract.

---

## Testing Requirements

Keep existing functional tests passing, especially:

- `tests/analysis/test_leverage.py`
- the multivariate target-specific tests added during Phase 4 closure

Add follow-up tests only if they protect a concrete performance-related behavior, for example:

- deterministic subsampling if a sampling option is introduced
- identical output schema before and after optimization
- same target coverage in `analyze_multivariate()`

Avoid brittle wall-clock assertions in CI.

---

## Acceptance Criteria

The handoff task is complete when:

1. leverage analysis is materially faster in local profiling, especially for multivariate cases
2. output schema and recommendations remain unchanged
3. multivariate target-specific tests still pass
4. any approximation or sampling rule is documented in code/docstrings
5. no joint/copula leverage work is introduced

---

## Suggested Validation Commands

```bash
uv run pytest tests/analysis/test_leverage.py -q
uv run pytest tests/analysis/test_leverage.py::TestFeatureLeverageAnalyzerMultivariate -q
```

If benchmark helpers are added, include one short command in the final thread summary showing how to reproduce the before/after comparison.

---

## Notes For Next Thread

- The recent Phase 4 closure already fixed correctness bugs in multivariate leverage and decomposition.
- A separate bug in `QuantileForestForecaster` multi-target feature selection was also fixed; do not revert that logic.
- Prefer small, mechanical improvements over another conceptual redesign.
