"""Tests for model comparison suite."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.metrics import (
    diebold_mariano_test,
    model_confidence_set,
    skill_score,
)
from uncertainty_flow.wrappers import ConformalRegressor


@pytest.fixture
def predictions():
    np.random.seed(42)
    n = 150
    df = pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "y": np.random.randn(n),
        }
    )

    model_a = ConformalRegressor(
        base_model=GradientBoostingRegressor(random_state=42, n_estimators=10),
        auto_tune=False,
        random_state=42,
    )
    model_b = ConformalRegressor(
        base_model=GradientBoostingRegressor(random_state=99, n_estimators=10),
        auto_tune=False,
        random_state=99,
    )
    model_a.fit(df, target="y")
    model_b.fit(df, target="y")

    pred_a = model_a.predict(df)
    pred_b = model_b.predict(df)
    y_arr = df["y"].to_numpy()
    return pred_a, pred_b, y_arr


class TestSkillScore:
    def test_crps_skill_score(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = skill_score(pred_a, pred_b, y_arr, metric="crps")
        assert result.shape[0] == 1
        assert "skill_score" in result.columns

    def test_mae_skill_score(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = skill_score(pred_a, pred_b, y_arr, metric="mae")
        assert result["skill_score"][0] is not None

    def test_identical_models(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = skill_score(pred_a, pred_a, y_arr, metric="crps")
        assert abs(result["skill_score"][0]) < 1e-10

    def test_multivariate_mae_uses_target_aligned_truth(self):
        levels = [0.1, 0.5, 0.9]
        y_true = np.array([[0.0, 10.0], [1.0, 11.0]])

        # Perfect medians for model A.
        q_a_t1 = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
        q_a_t2 = np.array([[9.0, 10.0, 11.0], [10.0, 11.0, 12.0]])
        pred_a = DistributionPrediction(
            np.column_stack([q_a_t1, q_a_t2]),
            levels,
            target_names=["t1", "t2"],
        )

        # Shifted medians for model B.
        q_b_t1 = q_a_t1 + 2.0
        q_b_t2 = q_a_t2 - 2.0
        pred_b = DistributionPrediction(
            np.column_stack([q_b_t1, q_b_t2]),
            levels,
            target_names=["t1", "t2"],
        )

        res = skill_score(pred_a, pred_b, y_true, metric="mae")
        assert res["skill_score"][0] > 0.9


class TestDieboldMariano:
    def test_basic_dm_test(self, predictions):
        pred_a, pred_b, y_arr = predictions
        median_a = pred_a.median().to_numpy().ravel()
        median_b = pred_b.median().to_numpy().ravel()
        err_a = np.abs(y_arr - median_a)
        err_b = np.abs(y_arr - median_b)
        result = diebold_mariano_test(err_a, err_b)
        assert "dm_statistic" in result.columns
        assert "p_value" in result.columns

    def test_two_sided(self, predictions):
        pred_a, _, y_arr = predictions
        median_a = pred_a.median().to_numpy().ravel()
        err_a = np.abs(y_arr - median_a)
        result = diebold_mariano_test(err_a, err_a, one_sided=False)
        assert result["p_value"][0] > 0.9

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            diebold_mariano_test(np.array([1, 2]), np.array([1]))


class TestModelConfidenceSet:
    def test_two_models(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = model_confidence_set({"A": pred_a, "B": pred_b}, y_arr, metric="mae")
        assert result.shape[0] == 2
        assert "in_set" in result.columns

    def test_single_model(self, predictions):
        pred_a, _, y_arr = predictions
        result = model_confidence_set({"A": pred_a}, y_arr, metric="crps")
        assert result.shape[0] == 1
        assert result["in_set"][0]

    def test_three_models(self, predictions):
        pred_a, pred_b, y_arr = predictions
        result = model_confidence_set({"A": pred_a, "B": pred_b, "C": pred_a}, y_arr, metric="mae")
        assert result.shape[0] == 3

    def test_multivariate_model_ranking(self):
        levels = [0.1, 0.5, 0.9]
        y_true = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])

        q_good_t1 = np.array([[-1, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=float)
        q_good_t2 = np.array([[9, 10, 11], [10, 11, 12], [11, 12, 13]], dtype=float)
        good = DistributionPrediction(
            np.column_stack([q_good_t1, q_good_t2]),
            levels,
            target_names=["t1", "t2"],
        )

        bad = DistributionPrediction(
            np.column_stack([q_good_t1 + 5, q_good_t2 - 5]),
            levels,
            target_names=["t1", "t2"],
        )

        mcs = model_confidence_set({"good": good, "bad": bad}, y_true, metric="mae")
        scores = {row["model"]: row["score"] for row in mcs.iter_rows(named=True)}
        assert scores["good"] < scores["bad"]


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
            np.column_stack([q_t1, q_t2]),
            levels,
            target_names=["t1", "t2"],
        )
        pred_b = DistributionPrediction(
            np.column_stack([q_t1 + 2, q_t2 - 2]),
            levels,
            target_names=["t1", "t2"],
        )
        result = skill_score(pred_a, pred_b, y_true, metric="pinball")
        assert result["skill_score"][0] is not None


class TestSkillScoreEdgeCases:
    def test_zero_baseline_score_handled(self, predictions):
        pred_a, _, y_arr = predictions
        result = skill_score(pred_a, pred_a, y_arr, metric="crps")
        assert result["skill_score"][0] == 0.0


class TestDieboldMarianoEdgeCases:
    def test_reject_branch(self, predictions):
        pred_a, pred_b, y_arr = predictions
        err_a = np.abs(y_arr - pred_a.median().to_numpy().ravel())
        err_b = np.abs(y_arr - pred_b.median().to_numpy().ravel()) * 1000
        result = diebold_mariano_test(err_a, err_b, one_sided=False)
        assert result["result"][0] == "reject"
        assert result["better_model"][0] == "A"

    def test_one_sided_p_value_rejects_when_a_better(self, predictions):
        pred_a, _, y_arr = predictions
        err_a = np.abs(y_arr - pred_a.median().to_numpy().ravel())
        rng = np.random.default_rng(42)
        err_b = err_a + rng.uniform(5, 15, size=len(err_a))
        result = diebold_mariano_test(err_a, err_b, one_sided=True)
        assert result["p_value"][0] < 0.05
        assert result["result"][0] == "reject"
        assert result["better_model"][0] == "A"

    def test_one_sided_p_value_large_when_a_worse(self, predictions):
        pred_a, _, y_arr = predictions
        err_a = np.abs(y_arr - pred_a.median().to_numpy().ravel())
        rng = np.random.default_rng(42)
        err_b = np.maximum(err_a - rng.uniform(0.01, 1.0, size=len(err_a)), 0.0)
        result = diebold_mariano_test(err_a, err_b, one_sided=True)
        assert result["p_value"][0] > 0.5


class TestModelConfidenceSetEdgeCases:
    def test_three_models_with_elimination(self, predictions):
        pred_a, pred_b, y_arr = predictions
        pred_c = pred_b
        result = model_confidence_set({"A": pred_a, "B": pred_b, "C": pred_c}, y_arr, metric="mae")
        assert result.shape[0] == 3
