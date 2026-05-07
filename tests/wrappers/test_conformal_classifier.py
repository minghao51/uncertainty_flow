"""Tests for ConformalClassifier."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from uncertainty_flow.utils.exceptions import ModelNotFittedError
from uncertainty_flow.wrappers import ConformalClassifier


@pytest.fixture
def binary_df():
    np.random.seed(42)
    n = 200
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "label": ["a"] * 100 + ["b"] * 100,
        }
    )


@pytest.fixture
def multiclass_df():
    np.random.seed(42)
    n = 300
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "label": ["a"] * 100 + ["b"] * 100 + ["c"] * 100,
        }
    )


class TestConformalClassifier:
    def test_fit_predict_binary(self, binary_df):
        model = ConformalClassifier(
            base_model=RandomForestClassifier(random_state=42, n_estimators=10),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        assert pred._n_samples == 200
        assert pred._n_classes == 2
        assert pred.size > 0

    def test_fit_predict_multiclass(self, multiclass_df):
        model = ConformalClassifier(
            base_model=RandomForestClassifier(random_state=42, n_estimators=10),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(multiclass_df, target="label")
        pred = model.predict(multiclass_df)
        assert pred._n_classes == 3
        size_by_sample = pred.size_by_sample()
        assert all(s >= 1 for s in size_by_sample)

    def test_prediction_set_access(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.95,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        s0 = pred.set(0)
        assert isinstance(s0, list)
        assert all(isinstance(c, str) for c in s0)
        all_sets = pred.set()
        assert len(all_sets) == 200

    def test_summary(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        summary = pred.summary()
        assert summary["coverage_target"][0] == 0.9
        assert summary["avg_set_size"][0] > 0

    def test_probabilities(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        probs = pred.probabilities()
        assert probs.shape == (200, 2)

    def test_coverage_property(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.85,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        assert pred.coverage == 0.85

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="coverage_target"):
            ConformalClassifier(LogisticRegression(), coverage_target=0.0)
        with pytest.raises(ValueError, match="calibration_size"):
            ConformalClassifier(LogisticRegression(), calibration_size=1.5)

    def test_not_fitted_error(self):
        model = ConformalClassifier(LogisticRegression())
        with pytest.raises(ModelNotFittedError):
            model.predict(None)

    def test_repr(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred = model.predict(binary_df)
        r = repr(pred)
        assert "PredictionSet" in r
        assert "n_samples=" in r

    def test_predict_batch(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        batches = list(model.predict_batch(binary_df, batch_size=50))
        assert len(batches) == 4
        total = sum(b._n_samples for b in batches)
        assert total == 200

    def test_predict_batch_single_batch(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        batches = list(model.predict_batch(binary_df, batch_size=500))
        assert len(batches) == 1
        assert batches[0]._n_samples == 200

    def test_save_load_roundtrip(self, binary_df, tmp_path):
        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42),
            coverage_target=0.9,
            random_state=42,
        )
        model.fit(binary_df, target="label")
        pred_before = model.predict(binary_df[:5])

        path = tmp_path / "model.uf"
        model.save(path)
        loaded = ConformalClassifier.load(path)
        pred_after = loaded.predict(binary_df[:5])

        assert pred_before._n_samples == pred_after._n_samples
        assert pred_before._n_classes == pred_after._n_classes
        np.testing.assert_allclose(
            pred_before.probabilities().to_numpy(),
            pred_after.probabilities().to_numpy(),
            atol=1e-6,
        )

    def test_fit_requires_target(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(),
            coverage_target=0.9,
        )
        with pytest.raises(Exception, match="target is required"):
            model.fit(binary_df)

    def test_fit_missing_target_column(self, binary_df):
        model = ConformalClassifier(
            base_model=LogisticRegression(),
            coverage_target=0.9,
        )
        with pytest.raises(ValueError, match="not found"):
            model.fit(binary_df, target="nonexistent")

    def test_fit_no_features(self):
        df = pl.DataFrame({"label": ["a", "b", "a", "b"]})
        model = ConformalClassifier(
            base_model=LogisticRegression(),
            coverage_target=0.9,
        )
        with pytest.raises(ValueError, match="No feature columns"):
            model.fit(df, target="label")

    def test_build_prediction_sets_threshold_zero(self):
        probs = np.array([[0.7, 0.3]])
        sets = ConformalClassifier._build_prediction_sets(probs, 0.0, ["a", "b"])
        assert len(sets) == 1
        assert len(sets[0]) == 1
        assert sets[0][0] == "a"

    def test_aps_threshold_uses_miscoverage_alpha(self):
        model = ConformalClassifier(
            base_model=LogisticRegression(),
            coverage_target=0.9,
            random_state=42,
        )
        model._model = type("M", (), {"classes_": np.array(["a", "b"])})()

        calib_probs = np.array(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]],
            dtype=float,
        )
        y_calib = np.array(["a", "a", "a", "a", "a"])

        threshold = model._compute_aps_threshold(
            calib_probs,
            y_calib,
            miscoverage_alpha=0.1,
        )
        assert threshold >= 0.8

    def test_empirical_coverage_near_target_on_easy_data(self):
        rng = check_random_state(42)
        n = 1500
        x1 = rng.randn(n)
        x2 = rng.randn(n)
        logits = 2.0 * x1 - 1.5 * x2
        probs_a = 1.0 / (1.0 + np.exp(-logits))
        labels = np.where(rng.rand(n) < probs_a, "a", "b")
        df = pl.DataFrame({"x1": x1, "x2": x2, "label": labels})

        model = ConformalClassifier(
            base_model=LogisticRegression(random_state=42, max_iter=500),
            coverage_target=0.9,
            calibration_size=0.3,
            random_state=42,
        )
        model.fit(df, target="label")
        pred = model.predict(df)
        sets = pred.set()
        truth = df["label"].to_numpy()
        covered = np.mean([truth[i] in sets[i] for i in range(n)])
        assert covered >= 0.85
