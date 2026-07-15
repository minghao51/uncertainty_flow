import numpy as np
import polars as pl
import pytest

import uncertainty_flow.benchmarking.tuning as tuning_module
from uncertainty_flow.benchmarking.tuning import (
    SEARCH_SPACE,
    TuningConfig,
    TuningResult,
    _score_result,
    auto_tune_model,
)
from uncertainty_flow.utils.exceptions import ConfigurationError


def _sample_df(n: int = 100, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "date": np.arange(n),
            "price": 10 + np.arange(n) * 0.5 + rng.standard_normal(n) * 0.5,
        }
    )


class TestTuningConfig:
    def test_defaults(self):
        cfg = TuningConfig()
        assert cfg.target_coverage == 0.9
        assert cfg.n_samples == 500
        assert cfg.timeout == 120
        assert cfg.hybrid_validation is False

    def test_custom(self):
        cfg = TuningConfig(target_coverage=0.8, timeout=60, hybrid_validation=True)
        assert cfg.target_coverage == 0.8
        assert cfg.timeout == 60
        assert cfg.hybrid_validation is True

    @pytest.mark.parametrize(
        "kwargs",
        (
            {"target_coverage": 1.0},
            {"n_samples": 2},
            {"timeout": 0},
        ),
    )
    def test_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            TuningConfig(**kwargs)


class TestTuningResult:
    def test_fields(self):
        r = TuningResult(
            model_name="qf",
            best_params={"n_estimators": 30},
            best_score=1.5,
            coverage_90=0.9,
            sharpness_90=1.2,
            winkler_90=2.5,
            train_time_sec=0.3,
            trials=3,
        )
        assert r.model_name == "qf"
        assert r.best_params == {"n_estimators": 30}
        assert r.trials == 3
        assert r.validation_strategy == "unknown"

    def test_validation_metadata_defaults(self):
        r = TuningResult(
            model_name="x",
            best_params={},
            best_score=0,
            coverage_90=0,
            sharpness_90=0,
            winkler_90=0,
            train_time_sec=0,
            trials=0,
        )
        assert r.validation_split_type == "unknown"
        assert r.validation_n_splits == 1


class TestScoreResult:
    def test_perfect_coverage_low_score(self):
        score_good = _score_result(0.9, 1.0, 2.0, target_coverage=0.9)
        score_bad = _score_result(0.5, 5.0, 10.0, target_coverage=0.9)
        assert score_good < score_bad

    def test_over_coverage_penalized(self):
        score_over = _score_result(1.0, 1.0, 2.0, target_coverage=0.9)
        score_on = _score_result(0.9, 1.0, 2.0, target_coverage=0.9)
        assert score_over > score_on

    def test_large_coverage_error_magnified(self):
        score_near = _score_result(0.85, 1.0, 2.0, target_coverage=0.9)
        score_far = _score_result(0.5, 1.0, 2.0, target_coverage=0.9)
        assert score_far > score_near


class TestSearchSpace:
    def test_known_models(self):
        assert "quantile-forest" in SEARCH_SPACE
        assert "conformal-regressor" in SEARCH_SPACE
        assert "conformal-forecaster" in SEARCH_SPACE

    def test_quantile_forest_keys(self):
        assert "n_estimators" in SEARCH_SPACE["quantile-forest"]
        assert "horizon" in SEARCH_SPACE["quantile-forest"]


class TestAutoTuneModel:
    def test_unknown_model_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown model"):
            auto_tune_model(
                model_name="nonexistent",
                df=_sample_df(),
                target="price",
                horizon=3,
            )

    def test_sample_limit_is_applied_before_validation(self, monkeypatch):
        observed_rows = []
        original_selector = tuning_module.select_validation_plan

        def capture_rows(df, **kwargs):
            observed_rows.append(len(df))
            return original_selector(df, **kwargs)

        monkeypatch.setattr(tuning_module, "select_validation_plan", capture_rows)
        monkeypatch.setitem(
            tuning_module.SEARCH_SPACE,
            "quantile-forest",
            {"n_estimators": [5], "horizon": [2]},
        )

        auto_tune_model(
            model_name="quantile-forest",
            df=_sample_df(150),
            target="price",
            horizon=2,
            config=TuningConfig(n_samples=100),
        )

        assert observed_rows == [100]

    def test_timeout_before_first_trial_is_explicit(self, monkeypatch):
        ticks = iter((0.0, 2.0))
        monkeypatch.setattr(tuning_module.time, "monotonic", lambda: next(ticks, 2.0))

        with pytest.raises(TimeoutError, match="before completing a trial"):
            auto_tune_model(
                model_name="quantile-forest",
                df=_sample_df(),
                target="price",
                horizon=2,
                config=TuningConfig(timeout=1),
            )
