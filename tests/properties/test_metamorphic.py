import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.ensemble import GradientBoostingRegressor

from uncertainty_flow.wrappers import ConformalRegressor


def _make_tabular_df(n: int, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * x1 - 1.0 * x2 + rng.standard_normal(n) * 0.5
    return pl.DataFrame({"x1": x1, "x2": x2, "y": y})


def _fit_predict(df_train, df_test):
    model = ConformalRegressor(
        base_model=GradientBoostingRegressor(n_estimators=10, random_state=42),
        calibration_size=0.3,
        auto_tune=False,
        random_state=42,
    )
    model.fit(df_train, target="y")
    return model.predict(df_test)


@given(shift=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
@settings(deadline=None, max_examples=15)
def test_translation_invariance(shift):
    df = _make_tabular_df(200, seed=42)
    df_shifted = df.with_columns((pl.col("y") + shift).alias("y"))

    pred_a = _fit_predict(df.head(150), df.tail(30))
    pred_b = _fit_predict(df_shifted.head(150), df_shifted.tail(30))

    med_a = pred_a.median().to_numpy()
    med_b = pred_b.median().to_numpy()
    np.testing.assert_allclose(med_b, med_a + shift, atol=0.5)


@given(scale=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False))
@settings(deadline=None, max_examples=15)
def test_scale_equivariance(scale):
    df = _make_tabular_df(200, seed=42)
    df_scaled = df.with_columns((pl.col("y") * scale).alias("y"))

    pred_a = _fit_predict(df.head(150), df.tail(30))
    pred_b = _fit_predict(df_scaled.head(150), df_scaled.tail(30))

    int_a = pred_a.interval(0.9)
    int_b = pred_b.interval(0.9)
    width_a = float((int_a["upper"] - int_a["lower"]).mean())
    width_b = float((int_b["upper"] - int_b["lower"]).mean())
    assert width_b > 0
    ratio = width_b / width_a if width_a > 0 else 1.0
    assert abs(ratio - scale) < scale * 0.5


@given(shift=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
@settings(deadline=None, max_examples=10)
def test_translation_preserves_interval_width(shift):
    df = _make_tabular_df(200, seed=42)
    df_shifted = df.with_columns((pl.col("y") + shift).alias("y"))

    pred_a = _fit_predict(df.head(150), df.tail(30))
    pred_b = _fit_predict(df_shifted.head(150), df_shifted.tail(30))

    int_a = pred_a.interval(0.9)
    int_b = pred_b.interval(0.9)
    width_a = float((int_a["upper"] - int_a["lower"]).mean())
    width_b = float((int_b["upper"] - int_b["lower"]).mean())
    np.testing.assert_allclose(width_b, width_a, rtol=0.05)
