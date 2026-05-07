"""Tests for predict_batch API."""

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.wrappers import ConformalRegressor, EnsembleBootstrapPI


@pytest.fixture
def df():
    np.random.seed(42)
    n = 100
    return pl.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "y": np.random.randn(n),
        }
    )


class TestPredictBatch:
    def test_conformal_batch(self, df):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(random_state=42, n_estimators=10),
            auto_tune=False,
            random_state=42,
        )
        model.fit(df, target="y")
        chunks = list(model.predict_batch(df, batch_size=30))
        assert len(chunks) == 4
        total = sum(c._n_samples for c in chunks)
        assert total == 100

    def test_enbpi_batch(self, df):
        model = EnsembleBootstrapPI(
            base_model=GradientBoostingRegressor(random_state=42),
            n_estimators=5,
            random_state=42,
        )
        model.fit(df, target="y")
        chunks = list(model.predict_batch(df, batch_size=40))
        assert len(chunks) == 3
        total = sum(c._n_samples for c in chunks)
        assert total == 100

    def test_single_batch(self, df):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(random_state=42, n_estimators=10),
            auto_tune=False,
            random_state=42,
        )
        model.fit(df, target="y")
        chunks = list(model.predict_batch(df, batch_size=1000))
        assert len(chunks) == 1

    def test_small_batch_last_chunk(self, df):
        model = ConformalRegressor(
            base_model=GradientBoostingRegressor(random_state=42, n_estimators=10),
            auto_tune=False,
            random_state=42,
        )
        model.fit(df, target="y")
        chunks = list(model.predict_batch(df, batch_size=33))
        assert len(chunks) == 4
        assert chunks[-1]._n_samples == 1
