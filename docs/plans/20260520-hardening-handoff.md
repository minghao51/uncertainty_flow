# Codebase Hardening Handoff

**Date:** 2026-05-20
**Status:** Ready for implementation
**Owner:** Next implementation thread
**Scope:** 7 surgical fixes from audit of unstaged changes across 8 files

---

## Summary

The current unstaged changes introduce exception narrowing, copula removal, legacy alias opt-out, and silent-swallow fixes. The audit identified 7 follow-up items. Since this project is not deployed or used externally, all deprecation/compat overhead is stripped — we remove dead paths outright.

**Principle:** No backward compatibility. No deprecation warnings. Remove the old, keep the clean.

---

## Task List

### Task 1: Revert Bandit CI Change

**Problem:** `.github/workflows/ci.yml:95` removed `B301,B403` from bandit skips. This **will fail CI** because `_persistence.py:8` (B403: `import pickle`) and `_persistence.py:250` (B301: `pickle.load`) are still flagged.

**Action:**
- Revert line to: `uv run bandit -r uncertainty_flow -s B101,B110,B301,B403`
- Alternatively, add `# nosec B403` at `_persistence.py:8` and `# nosec B301` at `_persistence.py:250`, then remove B301/B403 from both CI and `pyproject.toml`. Either approach works; the `# nosec` approach is cleaner long-term.

**Files:**
- `.github/workflows/ci.yml` (line 95)
- Optional: `uncertainty_flow/core/_persistence.py` (lines 8, 250)

**Verification:** `uv run bandit -r uncertainty_flow -s B101,B110` should exit 0 if using `# nosec`, or just keep the skips.

---

### Task 2: Remove `models` Alias Entirely

**Problem:** `to_dict()` emits identical data under both `"models"` and `"results"` keys. The current diff added an opt-out parameter, but since there are no external consumers, we should just remove the alias.

**Action:**
- Remove `include_legacy_models_alias` parameter from `to_dict()`
- Remove `"models"` key from the returned dict in all branches
- Remove `"deprecated_aliases"` from metadata
- Remove `include_legacy_models_alias` from `save_json()` if it was threaded through
- Update tests: remove `test_to_dict_without_legacy_models_alias`, update `test_to_dict_with_multiple_models` to assert `"models"` is absent and `"results"` exists
- In the early-return branch (`_run_result is None`), return `{"metadata": {}, "results": []}` only

**Files:**
- `uncertainty_flow/benchmarking/runner.py` (lines 561-610)
- `tests/benchmarking/test_runner.py` (lines 134-152 and the existing to_dict test)

**Verification:** `uv run pytest tests/benchmarking/test_runner.py -x -q`

---

### Task 3: Fix `to_dict()` Early-Return Consistency

**Problem:** When `_run_result is None`, the method returns `{"metadata": {}, "results": [], "models": []}`. After Task 2 removes the alias, this should return `{"metadata": {}, "results": []}`.

**Action:**
- This is handled by Task 2. Just ensure the None branch returns the clean payload.

**Files:** Same as Task 2.

---

### Task 4: Thread `save_json` Through (Optional After Task 2)

**Problem:** `save_json()` calls `self.to_dict()` with no arguments. After Task 2, there's no parameter to thread — this is resolved by the removal.

**Action:** No action needed after Task 2. `save_json` just calls `self.to_dict()` which now returns clean output.

---

### Task 5: Centralize `RECOVERABLE_EXCEPTIONS`

**Problem:** Three files define overlapping but inconsistent exception tuples independently:
- `cli.py:25` — has `click.ClickException`, `ImportError` but not `TypeError`, `KeyError`
- `runner.py:34` — has `ImportError` but not `TypeError`, `KeyError`
- `datasets.py:12` — has `TypeError`, `KeyError` but not `ImportError`

**Action:**

1. Add to `uncertainty_flow/utils/exceptions.py`:
```python
RECOVERABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    UncertaintyFlowError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    ImportError,
)
```

2. Update `datasets.py`:
```python
from ..utils.exceptions import RECOVERABLE_EXCEPTIONS
# Remove local RECOVERABLE_DATASET_EXCEPTIONS definition
# Use RECOVERABLE_EXCEPTIONS directly in except clause
```

3. Update `runner.py`:
```python
from ..utils.exceptions import RECOVERABLE_EXCEPTIONS
# Remove local RECOVERABLE_BENCHMARK_EXCEPTIONS definition
# Use RECOVERABLE_EXCEPTIONS directly in except clause
```

4. Update `cli.py`:
```python
from ..utils.exceptions import RECOVERABLE_EXCEPTIONS
RECOVERABLE_CLI_EXCEPTIONS = RECOVERABLE_EXCEPTIONS + (click.ClickException,)
# Remove local RECOVERABLE_CLI_EXCEPTIONS definition, replace with this
```

**Files:**
- `uncertainty_flow/utils/exceptions.py` (add constant)
- `uncertainty_flow/benchmarking/datasets.py` (line 12)
- `uncertainty_flow/benchmarking/runner.py` (lines 33-38)
- `uncertainty_flow/cli.py` (lines 23-28)

**Verification:** `uv run ruff check` + `uv run mypy` + `uv run pytest tests/ -x -q`

---

### Task 6: Fix `_aggregate()` Defensive Dispatch

**Problem:** `aggregator.py:155-159` — after removing the copula branch, the method has no `else`/fallback. If `aggregation` were somehow invalid (despite init validation), it silently falls through to `_aggregate_independent`.

**Action:**
```python
def _aggregate(self, group_preds: dict[str, DistributionPrediction]) -> np.ndarray:
    if self.aggregation == "product":
        return self._aggregate_product(group_preds)
    if self.aggregation == "independent":
        return self._aggregate_independent(group_preds)
    raise ConfigurationError(f"Unknown aggregation: {self.aggregation}")
```

**Files:**
- `uncertainty_flow/multimodal/aggregator.py` (lines 155-159)

**Verification:** `uv run pytest tests/multimodal/test_aggregator.py -x -q`

---

### Task 7: Update `STATE.md`

**Problem:** `.planning/STATE.md` lists several bugs that are now resolved by the current unstaged changes + the tasks above.

**Action — mark resolved:**
- Line 57: `except Exception: pass` in ensemble.py — **RESOLVED** (logger.debug)
- Line 49: copula stub — **RESOLVED** (removed from valid aggregations)
- Line 62-63: duplicate JSON in to_dict — **RESOLVED** (alias removed)
- Line 99: broad except Exception in CLI — **RESOLVED** (narrowed)

**Action — update:**
- Line 63: ruff E501 in `scripts/ci_policy_checks.py:34` — check if still present
- Line 64: unused import `sys` in `scripts/ci_policy_checks.py:8` — check if still present
- Security section line 70: pickle risk — update with `# nosec` status if Task 1 applied that fix
- Performance issues: unchanged (out of scope)

**Files:**
- `.planning/STATE.md`

---

## Execution Order

```
Task 1  ──►  Task 5  ──►  Task 6
Task 2  ──┘               │
                          ▼
                       Task 7
```

- Tasks 1 and 2 are independent, do them first.
- Task 5 depends on Task 1 being stable (touches same exception pattern).
- Task 6 is independent but quick, can run in parallel with Task 5.
- Task 7 is last — update docs after all code changes land.

---

## Out of Scope

These are noted but explicitly not in this pass:

| Item | Reason |
|------|--------|
| Pickle → safetensors migration | Large refactor, separate effort |
| Performance issues in STATE.md | Unrelated to this audit |
| mypy errors in parametric.py, plotting.py, adaptive_conformal.py | Pre-existing, not introduced by current changes |
| Low coverage floor (40%) | Strategic decision, not a bug fix |
| Shared model registry / plugin discovery | Feature work |
| `DEFAULT_QUANTILES` dynamic proxy staleness | Subtle, needs separate design discussion |

---

## Verification Checklist

After all tasks are complete, run:

```bash
uv run ruff check uncertainty_flow/ tests/
uv run ruff format uncertainty_flow/ tests/ --check
uv run mypy uncertainty_flow/
uv run bandit -r uncertainty_flow -s B101,B110
uv run pytest tests/ -x -q
```

All must pass clean.
