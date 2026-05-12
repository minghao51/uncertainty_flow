"""Full workflow integration tests."""

import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.metrics import score
from uncertainty_flow.models import QuantileForestForecaster
from uncertainty_flow.wrappers import ConformalRegressor


@pytest.fixture
def sample_data():
    """Generate sample tabular data for testing."""
    n = 200
    return pl.DataFrame(
        {
            "feature1": list(range(n)),
            "feature2": [x * 2 + 0.5 for x in range(n)],
            "feature3": [x**0.5 for x in range(n)],
            "target": [x * 1.5 + 10 + (x % 5) for x in range(n)],
        }
    )


@pytest.fixture
def time_series_data():
    """Generate sample time series data for testing."""
    n = 100
    return pl.DataFrame(
        {
            "date": list(range(n)),
            "price": [100 + x + x % 10 for x in range(n)],
            "volume": [1000 + x * 2 for x in range(n)],
        }
    )


class TestConformalRegressorWorkflow:
    """Test ConformalRegressor full fit/predict/save/load workflow."""

    def test_fit_predict(self, sample_data):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(sample_data, target="target")
        pred = model.predict(sample_data)

        assert pred is not None
        assert hasattr(pred, "interval")
        interval = pred.interval(0.9)
        assert "lower" in interval.columns
        assert "upper" in interval.columns
        assert len(interval) == len(sample_data)

    def test_calibration_report(self, sample_data):
        train, test = sample_data.head(150), sample_data.tail(50)
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(train, target="target")
        report = model.calibration_report(test, target="target")

        assert report is not None
        assert len(report) >= 1
        assert "quantile" in report.columns
        assert "achieved_coverage" in report.columns

    def test_save_load_roundtrip(self, sample_data, tmp_path):
        train, test = sample_data.head(150), sample_data.tail(50)

        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(train, target="target")
        original_pred = model.predict(test)

        path = tmp_path / "conformal_model.uf"
        model.save(path)
        assert path.exists()

        loaded = ConformalRegressor.load(path)
        loaded_pred = loaded.predict(test)

        orig_interval = original_pred.interval(0.9)
        loaded_interval = loaded_pred.interval(0.9)

        assert orig_interval["lower"].equals(loaded_interval["lower"])
        assert orig_interval["upper"].equals(loaded_interval["upper"])


class TestQuantileForestForecasterWorkflow:
    """Test QuantileForestForecaster full fit/predict workflow."""

    def test_fit_predict(self, time_series_data):
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=5,
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)

        assert pred is not None
        interval = pred.interval(0.9)
        assert "price_lower" in interval.columns
        assert "price_upper" in interval.columns
        assert "volume_lower" in interval.columns
        assert "volume_upper" in interval.columns

    def test_multivariate_interval(self, time_series_data):
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=5,
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)

        mean = pred.median()
        assert isinstance(mean, pl.DataFrame)
        assert "price" in mean.columns
        assert "volume" in mean.columns

    def test_sample(self, time_series_data):
        model = QuantileForestForecaster(
            targets=["price"],
            horizon=3,
            n_estimators=5,
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)
        samples = pred.sample(n=5, random_state=42)

        assert "sample_id" in samples.columns
        assert "price" in samples.columns
        assert len(samples) == 5 * len(time_series_data)


class TestDistributionPredictionInterface:
    """Test DistributionPrediction interface consistency."""

    def test_quantile_method_univariate(self, sample_data):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(sample_data, target="target")
        pred = model.predict(sample_data)

        q = pred.quantile([0.1, 0.5, 0.9])
        assert "q_0.100" in q.columns
        assert "q_0.500" in q.columns
        assert "q_0.900" in q.columns

    def test_quantile_method_multivariate(self, time_series_data):
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=5,
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)

        q = pred.quantile([0.1, 0.5])
        assert "price_q_0.100" in q.columns
        assert "volume_q_0.100" in q.columns

    def test_uncertainty_decomposition(self, sample_data):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(sample_data, target="target")
        pred = model.predict(sample_data)

        decomp = pred.uncertainty_decomposition(confidence=0.9)
        assert "aleatoric" in decomp
        assert "epistemic" in decomp
        assert "total" in decomp


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end scenarios simulating real usage."""

    def test_tabular_uncertainty_quantification(self, sample_data):
        """Simulate a complete tabular uncertainty quantification workflow."""
        train, test = sample_data.head(150), sample_data.tail(50)

        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
            calibration_size=0.2,
            auto_tune=False,
            random_state=42,
        )
        model.fit(train, target="target")

        pred = model.predict(test)

        report = model.calibration_report(test, target="target")

        pred_quantiles = pred.quantile([0.05, 0.5, 0.95])
        pred_intervals = pred.interval(confidence=0.9)

        assert report.shape[0] >= 1
        assert pred_quantiles.shape[0] == len(test)
        assert pred_intervals.shape[0] == len(test)

    def test_time_series_multivariate_forecasting(self, time_series_data):
        """Simulate a complete time series forecasting workflow."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=5,
            n_estimators=10,
            copula_family="gaussian",
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)

        pred = model.predict(time_series_data)

        samples = pred.sample(n=10, random_state=42)

        decomp = pred.uncertainty_decomposition(confidence=0.9)

        assert samples.shape[0] == 10 * len(time_series_data)
        assert decomp["total"] >= 0


@pytest.mark.integration
class TestScoreAPIIntegration:
    """End-to-end tests for the unified score() API."""

    def test_univariate_score_pipeline(self, sample_data):
        """fit → predict → score(pred, y, metric) for all univariate metrics."""
        train, test = sample_data.head(150), sample_data.tail(50)
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(train, target="target")
        pred = model.predict(test)

        y_true = test["target"].to_numpy()

        crps = score(pred, y_true, "crps")
        assert isinstance(crps, float)
        assert crps >= 0

        coverage = score(pred, y_true, "coverage")
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

        mae = score(pred, y_true, "mae")
        assert isinstance(mae, float)
        assert mae >= 0

        pinball = score(pred, y_true, "pinball")
        assert isinstance(pinball, float)
        assert pinball >= 0

    def test_multivariate_score_pipeline(self, time_series_data):
        """fit → predict → score(pred, y, metric) for multivariate metrics."""
        model = QuantileForestForecaster(
            targets=["price", "volume"],
            horizon=3,
            n_estimators=5,
            auto_tune=False,
            random_state=42,
        )
        model.fit(time_series_data)
        pred = model.predict(time_series_data)

        y_true = time_series_data.select(["price", "volume"]).to_numpy()

        crps = score(pred, y_true, "crps")
        assert isinstance(crps, dict)
        assert "price" in crps and "volume" in crps

        coverage = score(pred, y_true, "coverage")
        assert isinstance(coverage, dict)
        assert "price" in coverage and "volume" in coverage

    def test_crps_consistency_with_direct_call(self, sample_data):
        """score(pred, y, 'crps') matches pred.crps(y)."""
        train, test = sample_data.head(150), sample_data.tail(50)
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(n_estimators=5, random_state=42),
            auto_tune=False,
            random_state=42,
        )
        model.fit(train, target="target")
        pred = model.predict(test)
        y_true = test["target"].to_numpy()

        via_score = score(pred, y_true, "crps")
        via_direct = pred.crps(y_true)
        assert via_score == pytest.approx(via_direct)
