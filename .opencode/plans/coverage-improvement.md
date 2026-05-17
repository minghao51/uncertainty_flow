# Coverage Improvement Plan

## Current State
- **Baseline coverage**: 46% (from `tests/core/`, `tests/metrics/`, `tests/utils/`)
- **`--cov-fail-under`**: 40%
- **Modules below 50%**: counterfactual, decomposition, risk, multimodal, multivariate (without its tests), wrappers (conformal_classifier, enbpi, adaptive_conformal), viz

---

## Phase 1: Quick Wins — Edge Case Gaps

### 1.1 `core/_persistence.py` — Add to `tests/core/test_persistence.py`

Add new test classes:

```python
class TestPersistenceHelpers:
    def test_safe_version_known_package(self):
        v = _safe_version("numpy")
        assert v is not None
        assert len(v) > 0

    def test_safe_version_unknown_package(self):
        assert _safe_version("non-existent-package-xyz") is None

    def test_library_version_returns_string(self):
        v = _library_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_compute_archive_sha256(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello")
        h = compute_archive_sha256(p)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_target_names_from_targets_list(self):
        model = types.SimpleNamespace(targets=["a", "b"])
        assert _target_names(model) == ["a", "b"]

    def test_target_names_from_target_col(self):
        model = types.SimpleNamespace(targets=None, _target_col_="price")
        assert _target_names(model) == ["price"]

    def test_target_names_returns_none(self):
        model = types.SimpleNamespace()
        assert _target_names(model) is None

    def test_quantile_levels_from_attribute(self):
        model = types.SimpleNamespace(quantile_levels=[0.1, 0.5, 0.9])
        assert _quantile_levels(model) == [0.1, 0.5, 0.9]

    def test_quantile_levels_default_quantiles(self):
        model = types.SimpleNamespace(_quantiles_=True)
        levels = _quantile_levels(model)
        assert levels is not None
        assert all(0 < q < 1 for q in levels)

    def test_warn_version_mismatches_no_warning(self, recwarn):
        _warn_version_mismatches({"dependencies": {"numpy": _safe_version("numpy")}})
        assert len(recwarn) == 0

    def test_warn_version_mismatches_with_warning(self, recwarn):
        _warn_version_mismatches({"dependencies": {"numpy": "0.0.1"}})
        assert len(recwarn) >= 1
```

Add import: `import types` at top.

### 1.2 `core/parametric.py` — Add to `tests/core/test_parametric.py`

Add test class:

```python
class TestParametricEdgeCases:
    def test_optimization_fallback_returns_initial(self):
        """When optimization fails, initial params should be returned."""
        qv = np.array([-1.0, 0.0, 1.0])
        ql = np.array([0.1, 0.5, 0.9])
        init = _fit_family("normal", qv, ql)
        refined = _refine_params("normal", init, qv * 1e100, ql)
        assert isinstance(refined, dict)
        assert "loc" in refined

    def test_auto_fallback_when_all_families_fail(self):
        """When all families fail to fit, auto should fall back to normal."""
        qv = np.array([0.0, 0.0, 0.0])
        ql = np.array([0.1, 0.5, 0.9])
        dist = fit_parametric(qv, ql, family="auto")
        assert dist.family == "normal"

    def test_ks_distance_overflow_returns_inf(self):
        qv = np.array([1e308, 1e308, 1e308])
        ql = np.array([0.1, 0.5, 0.9])
        ks = _ks_distance("normal", {"loc": 1e308, "scale": 1e308}, qv, ql)
        assert ks == np.inf
```

### 1.3 `metrics/comparison.py` — Add to `tests/metrics/test_comparison.py`

Add test classes:

```python
class TestExtractErrorsPinball:
    def test_pinball_metric_via_skill_score(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = skill_score(pred_a, pred_b, y_arr, metric="pinball")
        assert result.shape[0] == 1
        assert result["skill_score"][0] is not None

    def test_pinball_multivariate(self):
        levels = [0.1, 0.5, 0.9]
        y_true = np.array([[0.0, 10.0], [1.0, 11.0]])
        q_t1 = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
        q_t2 = np.array([[9.0, 10.0, 11.0], [10.0, 11.0, 12.0]])
        pred_a = DistributionPrediction(
            np.column_stack([q_t1, q_t2]), levels, target_names=["t1", "t2"]
        )
        pred_b = DistributionPrediction(
            np.column_stack([q_t1 + 2, q_t2 - 2]), levels, target_names=["t1", "t2"]
        )
        result = skill_score(pred_a, pred_b, y_true, metric="pinball")
        assert result["skill_score"][0] is not None


class TestSkillScoreEdgeCases:
    def test_zero_baseline_score(self, predictions):
        pred_a, _, y_arr = predictions
        result = skill_score(pred_a, pred_a, y_arr, metric="crps")
        assert result["skill_score"][0] == 0.0


class TestDieboldMarianoEdgeCases:
    def test_reject_branch(self, predictions):
        pred_a, pred_b, y_arr = predictions
        err_a = np.abs(y_arr - pred_a.median().to_numpy().ravel())
        err_b = np.abs(y_arr - pred_b.median().to_numpy().ravel()) * 1000
        result = diebold_mariano_test(err_a, err_b)
        assert result["result"][0] == "reject"
        assert result["better_model"][0] == "A"


class TestModelConfidenceSetEdgeCases:
    def test_three_models_with_elimination(self, predictions):
        pred_a, pred_b, y_arr = predictions
        pred_c = pred_b
        result = model_confidence_set({"A": pred_a, "B": pred_b, "C": pred_c}, y_arr, metric="mae")
        assert result.shape[0] == 3
```

### 1.4 `analysis/leverage.py` — Add to `tests/analysis/test_leverage.py`

Add test class + fixtures:

```python
class TestRecommendationBranches:
    def test_accept_uncertainty(self):
        rec = _generate_recommendation(10.0, 1.0, 0.1)
        assert rec == "accept_uncertainty"

    def test_collect_more_data(self):
        rec = _generate_recommendation(1.0, 5.0, 0.1)
        assert rec == "collect_more_data"

    def test_high_leverage(self):
        rec = _generate_recommendation(2.0, 3.0, 0.8)
        assert rec == "high_leverage"

    def test_low_leverage(self):
        rec = _generate_recommendation(2.0, 3.0, 0.1)
        assert rec == "low_leverage"

    def test_recommendation_threshold_boundary(self):
        rec = _generate_recommendation(2.0, 3.0, 0.5)
        assert rec == "low_leverage"


class TestFormatRecommendation:
    def test_all_keys_present(self):
        for key in ["accept_uncertainty", "collect_more_data", "high_leverage", "low_leverage"]:
            result = _format_recommendation(key)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_key(self):
        assert _format_recommendation("unknown") == "Unknown"


class TestInternalHelpers:
    def test_point_matrix_multi_target(self):
        n, nq = 10, 3
        q = np.column_stack([
            np.linspace(-2, 2, n),
            np.linspace(-1, 3, n),
            np.linspace(0, 4, n),
            np.linspace(1, 5, n),
            np.linspace(2, 6, n),
            np.linspace(3, 7, n),
        ])
        pred = DistributionPrediction(q, [0.1, 0.5, 0.9], target_names=["a", "b"])
        mat = _point_matrix(pred)
        assert mat.shape == (n, 2)

    def test_interval_width_matrix_multi_target(self):
        n, nq = 10, 3
        q = np.column_stack([
            np.linspace(-2, 2, n),
            np.linspace(0, 4, n),
            np.linspace(2, 6, n),
            np.linspace(-1, 3, n),
            np.linspace(1, 5, n),
            np.linspace(3, 7, n),
        ])
        pred = DistributionPrediction(q, [0.1, 0.5, 0.9], target_names=["a", "b"])
        mat = _interval_width_matrix(pred, 0.8)
        assert mat.shape == (n, 2)

    def test_rank_correlation_single_target(self):
        mat = np.random.randn(10, 1)
        corr = _rank_correlation_matrix(mat)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 1.0

    def test_mean_upper_triangle_single(self):
        assert _mean_upper_triangle_abs(np.eye(2)) == 0.0

    def test_mean_upper_triangle_single_row(self):
        assert _mean_upper_triangle_abs(np.array([[1.0]])) == 0.0

    def test_analyze_empty_data_returns_schema(self, sample_forecaster):
        """analyze() with all constant features should return empty DataFrame with schema."""
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        data = pl.DataFrame({"x1": [5.0] * 10, "x2": [3.0] * 10})
        result = analyzer.analyze(data)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0


class TestDecompositionFallback:
    def test_single_bin_fallback(self, sample_forecaster):
        analyzer = FeatureLeverageAnalyzer(sample_forecaster)
        feat = np.ones(20)
        widths = np.random.randn(20)**2
        alea, epi = analyzer._compute_decomposition(feat, widths)
        assert alea == 0.0
        assert epi == 0.0
```

Add imports:
```python
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.analysis.leverage import (
    _generate_recommendation, _format_recommendation,
    _point_matrix, _interval_width_matrix,
    _rank_correlation_matrix, _mean_upper_triangle_abs,
)
```

---

## Phase 2: Enhance Existing Test Files

### 2.1 `decomposition/ensemble.py` — Enhance `tests/decomposition/test_ensemble.py`

Add property-based tests:
```python
@given(...)
def test_decomposition_total_equals_sum(multi_model_prediction):
    dec = EnsembleDecomposition(models).decompose(data)
    assert abs(dec["aleatoric"] + dec["epistemic"] - dec["total"]) < 1e-10

def test_fit_ensemble_error_path():
    """Silent except for random_state should not break."""
    ...

def test_interval_width_matrix_multi_target():
    ...
```

### 2.2 `risk/control.py` + `risk/risk_functions.py` — Enhance

Add tests for:
- Multi-target prediction branch in `_prediction_mean` (lines 22-23)
- Multi-target in `_interval_half_width` (lines 33-35)
- `_risk_metric_fn("mean")` branch (line 207)
- `_estimate_risk` unset state (line 219)
- Shape validation (line 134)
- All risk functions in `risk_functions.py`

### 2.3 `utils/split.py` — Enhance

Add edge case tests:
- Holdout validation splits
- KFold with non-default n_splits
- Rolling-origin with small data
- Sliding-window boundary cases

### 2.4 `multimodal/aggregator.py` — Fix/enhance

The existing test times out — likely too much data. Add minimal-data tests for:
- `product` aggregation mode fit/predict
- `independent` aggregation mode fit/predict

---

## Phase 3: Wrapper Coverage

### 3.1 `conformal_classifier.py`
- Test `ConformalClassifier.fit()` with synthetic binary classification
- Test `predict()` and `predict_set()` output shapes

### 3.2 `enbpi.py`
- Test `EnsembleBootstrapPI.fit()` and `predict()` with small synthetic data
- Verify coverage near nominal level

### 3.3 `adaptive_conformal.py`
- Test `AdaptiveConformalForecaster` update loop
- Verify moving window / error coverage adapts

### 3.4 `conformal.py` + `conformal_ts.py`
- Enhance with edge-case branches, single-row predict, multi-target predict

---

## Phase 4: Counterfactual Module

### 4.1 `search.py`
- Test `EvolutionarySearch.search()` with simple synthetic model
- Test `GradientSearch.search()` finite-difference path
- Test `SearchResult.to_polars()`

### 4.2 `explainer.py`
- Test `explain_batch()` with small data
- Test `compare_features()` returns valid DataFrame
- Test `summary()` returns config dict

---

## Coverage Target Progression

| After | Threshold |
|-------|-----------|
| Phase 1 | 45% |
| Phase 2 | 50% |
| Phase 3 | 60% |
| Phase 4 | 65%+ |
