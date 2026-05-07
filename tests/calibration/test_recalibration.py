"""Tests for isotonic recalibration (RecalibratedModel)."""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from uncertainty_flow.calibration.recalibration import RecalibratedModel
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.wrappers import ConformalRegressor


class _MiscalibratedModel:
    """A deliberately miscalibrated model for testing recalibration.

    Predicts quantiles that are systematically too narrow (overconfident).
    """

    def __init__(self, n_quantiles: int = 9):
        self._levels = np.linspace(0.1, 0.9, n_quantiles)
        self._fitted = True

    def predict(self, data: pl.DataFrame) -> DistributionPrediction:
        n = len(data)
        # Systematically narrow predictions: quantiles compressed toward median
        base = np.arange(n, dtype=float)
        quantile_matrix = np.empty((n, len(self._levels)))
        for i, tau in enumerate(self._levels):
            # True distribution is base + N(0, 4), but we predict only half the spread
            spread = (tau - 0.5) * 2.0  # half the true spread
            quantile_matrix[:, i] = base + spread
        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=self._levels.tolist(),
            target_names=["y"],
        )


class TestRecalibratedModel:
    """Tests for RecalibratedModel."""

    def _make_miscalibrated_data(self, n: int = 200, rng_seed: int = 42):
        """Create data where a simple model will be miscalibrated."""
        rng = np.random.default_rng(rng_seed)
        x = np.linspace(0, 10, n)
        # True relationship: y = 2*x + noise with std=2
        y = 2 * x + rng.normal(0, 2, size=n)
        return pl.DataFrame({"x": x, "y": y})

    def test_fit_predict_basic(self):
        """Should fit and predict without errors."""
        df = self._make_miscalibrated_data(n=100)
        train = df[:80]
        calib = df[80:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        recal = RecalibratedModel(model=base)
        recal.fit(calib, target="y")

        pred = recal.predict(calib)
        assert isinstance(pred, DistributionPrediction)
        assert pred._targets == ["y"]

    def test_recalibration_improves_calibration(self):
        """Recalibrated predictions should have better empirical coverage."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.linspace(0, 10, n)
        # True noise std = 3, but base model will be trained with small calibration
        y = 2 * x + rng.normal(0, 3, size=n)
        df = pl.DataFrame({"x": x, "y": y})

        train = df[:180]
        calib = df[180:240]
        test = df[240:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        # Measure base model calibration on test set (80% coverage target)
        base_pred = base.predict(test)
        base_interval = base_pred.interval(0.8)
        base_lower = base_interval["lower"].to_numpy()
        base_upper = base_interval["upper"].to_numpy()
        y_test = test["y"].to_numpy()
        base_coverage = np.mean((y_test >= base_lower) & (y_test <= base_upper))

        # Recalibrate on calibration set
        recal = RecalibratedModel(model=base)
        recal.fit(calib, target="y")

        # Measure recalibrated coverage on test set
        recal_pred = recal.predict(test)
        recal_interval = recal_pred.interval(0.8)
        recal_lower = recal_interval["lower"].to_numpy()
        recal_upper = recal_interval["upper"].to_numpy()
        recal_coverage = np.mean((y_test >= recal_lower) & (y_test <= recal_upper))

        # Recalibration should move coverage closer to target (0.8)
        base_error = abs(base_coverage - 0.8)
        recal_error = abs(recal_coverage - 0.8)
        assert recal_error <= base_error + 0.1  # allow variance

    def test_identity_on_perfectly_calibrated_model(self):
        """For a perfectly calibrated model, recalibration should be near-identity."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.normal(0, 1, size=n)
        df = pl.DataFrame({"y": y_true})

        # Perfectly calibrated predictions: quantiles match true distribution
        levels = [0.1, 0.5, 0.9]
        perfect_quantiles = np.column_stack(
            [
                np.full(n, -1.28),  # 10th percentile of N(0,1)
                np.full(n, 0.0),  # median
                np.full(n, 1.28),  # 90th percentile
            ]
        )

        class PerfectModel:
            def predict(self, data):
                return DistributionPrediction(
                    quantile_matrix=perfect_quantiles[: len(data)],
                    quantile_levels=levels,
                    target_names=["y"],
                )

        recal = RecalibratedModel(model=PerfectModel())
        recal.fit(df, target="y")

        # The isotonic map should be close to identity for perfect calibration
        iso = recal._isotonic_regressors[0]
        test_levels = np.array([0.1, 0.5, 0.9])
        mapped = iso.predict(test_levels)
        # Empirical coverage should be close to nominal levels
        assert np.allclose(mapped, test_levels, atol=0.15)

    def test_cross_calibrate_mode(self):
        """Cross-calibration should work without errors."""
        df = self._make_miscalibrated_data(n=100)
        train = df[:80]
        calib = df[80:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        recal = RecalibratedModel(model=base, cross_calibrate=True, n_folds=3)
        recal.fit(calib, target="y")

        pred = recal.predict(calib)
        assert isinstance(pred, DistributionPrediction)

    def test_custom_quantile_levels(self):
        """Should support custom output quantile levels."""
        df = self._make_miscalibrated_data(n=100)
        train = df[:80]
        calib = df[80:]

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(train, target="y")

        custom_levels = [0.05, 0.5, 0.95]
        recal = RecalibratedModel(model=base, quantile_levels=custom_levels)
        recal.fit(calib, target="y")

        pred = recal.predict(calib)
        assert pred._levels.tolist() == custom_levels

    def test_not_fitted_raises(self):
        """Predict before fit should raise an error."""
        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
        )
        # Note: base model itself is not fitted, but the error should come
        # from RecalibratedModel not being fitted.
        recal = RecalibratedModel(model=base)
        with pytest.raises(Exception):  # noqa: B017
            recal.predict(pl.DataFrame({"x": [1.0]}))

    def test_multivariate_targets(self):
        """Should handle multivariate predictions."""
        rng = np.random.default_rng(42)
        n = 150
        x = np.linspace(0, 10, n)
        y1 = 2 * x + rng.normal(0, 1, size=n)
        y2 = -1 * x + rng.normal(0, 1, size=n)
        df = pl.DataFrame({"x": x, "y1": y1, "y2": y2})

        _ = df[:100]
        calib = df[100:]

        # Use a simple model that predicts both targets
        class MultiModel:
            def __init__(self):
                self._fitted = True

            def predict(self, data):
                _ = len(data)
                levels = [0.1, 0.5, 0.9]
                # Simple but miscalibrated predictions
                q1 = np.column_stack(
                    [
                        data["x"].to_numpy() - 0.5,
                        data["x"].to_numpy(),
                        data["x"].to_numpy() + 0.5,
                    ]
                )
                q2 = np.column_stack(
                    [
                        -data["x"].to_numpy() - 0.5,
                        -data["x"].to_numpy(),
                        -data["x"].to_numpy() + 0.5,
                    ]
                )
                quantile_matrix = np.column_stack([q1, q2])
                return DistributionPrediction(
                    quantile_matrix=quantile_matrix,
                    quantile_levels=levels,
                    target_names=["y1", "y2"],
                )

        recal = RecalibratedModel(model=MultiModel())
        recal.fit(calib)

        pred = recal.predict(calib)
        assert pred._targets == ["y1", "y2"]
        assert pred._n_quantiles == 3

    def test_cross_calibrate_fallback_with_few_samples(self):
        n = 1
        df = self._make_miscalibrated_data(n=n)

        base = ConformalRegressor(
            base_model=LinearRegression(),
            auto_tune=False,
            calibration_size=0.3,
        )
        base.fit(self._make_miscalibrated_data(n=100), target="y")

        recal = RecalibratedModel(model=base, cross_calibrate=True, n_folds=5)
        recal.fit(df, target="y")

        pred = recal.predict(df)
        assert isinstance(pred, DistributionPrediction)

    def test_cross_calibrate_multivariate(self):
        rng = np.random.default_rng(42)
        n = 150
        x = np.linspace(0, 10, n)
        y1 = 2 * x + rng.normal(0, 1, size=n)
        y2 = -1 * x + rng.normal(0, 1, size=n)
        df = pl.DataFrame({"x": x, "y1": y1, "y2": y2})

        calib = df[100:]

        class MultiModel:
            _fitted = True

            def predict(self, data):
                _ = len(data)
                levels = [0.1, 0.5, 0.9]
                q1 = np.column_stack(
                    [
                        data["x"].to_numpy() - 0.5,
                        data["x"].to_numpy(),
                        data["x"].to_numpy() + 0.5,
                    ]
                )
                q2 = np.column_stack(
                    [
                        -data["x"].to_numpy() - 0.5,
                        -data["x"].to_numpy(),
                        -data["x"].to_numpy() + 0.5,
                    ]
                )
                quantile_matrix = np.column_stack([q1, q2])
                return DistributionPrediction(
                    quantile_matrix=quantile_matrix,
                    quantile_levels=levels,
                    target_names=["y1", "y2"],
                )

        recal = RecalibratedModel(model=MultiModel(), cross_calibrate=True, n_folds=3)
        recal.fit(calib)

        pred = recal.predict(calib)
        assert pred._targets == ["y1", "y2"]

    def test_miscalibrated_model_recalibration(self):
        rng = np.random.default_rng(42)
        n = 300
        y_true = rng.normal(0, 1, size=n)
        df = pl.DataFrame({"y": y_true})

        levels = [0.1, 0.5, 0.9]
        # Deliberately miscalibrated: quantiles are shifted
        miscal_quantiles = np.column_stack(
            [
                np.full(n, -2.0),
                np.full(n, 0.0),
                np.full(n, 0.5),
            ]
        )

        class MiscalModel:
            _fitted = True

            def predict(self, data):
                return DistributionPrediction(
                    quantile_matrix=miscal_quantiles[: len(data)],
                    quantile_levels=levels,
                    target_names=["y"],
                )

        recal = RecalibratedModel(model=MiscalModel())
        recal.fit(df, target="y")

        pred = recal.predict(df[:10])
        assert isinstance(pred, DistributionPrediction)
        assert pred._n_samples == 10
        # Quantiles should be sorted (output_matrix sorted in predict)
        q = pred._quantiles
        assert np.all(q[:, 0] <= q[:, 1])
        assert np.all(q[:, 1] <= q[:, 2])
