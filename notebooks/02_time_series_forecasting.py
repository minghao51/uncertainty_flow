# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars>=0.20",
#     "scikit-learn>=1.3",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Time Series Forecasting with Uncertainty

    Forecast temperature using the **weather dataset** (36K+ hourly measurements) with conformal prediction bands.

    This notebook compares two approaches:
    - **ConformalForecaster** — wrap any sklearn model, guaranteed coverage
    - **QuantileForestForecaster** — quantile regression forest, empirical coverage
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor

    from uncertainty_flow.metrics import coverage_score, winkler_score
    from uncertainty_flow.models import QuantileForestForecaster
    from uncertainty_flow.wrappers import ConformalForecaster

    return (
        ConformalForecaster,
        GradientBoostingRegressor,
        QuantileForestForecaster,
        coverage_score,
        pl,
        winkler_score,
    )


@app.cell
def _(pl):
    weather = pl.read_parquet("../data/weather.parquet")
    weather_clean = weather.drop_nulls()
    f"Rows: {weather_clean.height:,} | Columns: {weather_clean.width}"
    return (weather_clean,)


@app.cell
def _(mo):
    target_col = mo.ui.dropdown(
        options=["T (degC)", "rh (%)", "VPdef (mbar)"],
        value="T (degC)",
        label="Target variable",
    )
    horizon = mo.ui.slider(start=1, stop=24, step=1, value=6, label="Forecast horizon (hours)")
    mo.vstack([target_col, horizon])
    return horizon, target_col


@app.cell
def _(target_col, weather_clean):
    df = weather_clean.select([target_col.value])
    n = df.height
    split_idx = int(n * 0.85)
    train_ts = df[:split_idx]
    test_ts = df[split_idx:]
    f"Train: {train_ts.height:,} | Test: {test_ts.height:,}"
    return split_idx, test_ts, train_ts


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Train ConformalForecaster")
    run_btn
    return (run_btn,)


@app.cell
def _(
    ConformalForecaster,
    GradientBoostingRegressor,
    horizon,
    run_btn,
    target_col,
    test_ts,
    train_ts,
):
    if run_btn.value:
        ts_model = ConformalForecaster(
            base_model=GradientBoostingRegressor(random_state=42),
            targets=target_col.value,
            horizon=horizon.value,
            lags=[1, 2, 3, 6, 12, 24],
            copula_family="independent",
            auto_tune=False,
            random_state=42,
        )
        ts_model.fit(train_ts)
        ts_pred = ts_model.predict(test_ts)
    else:
        ts_model = None
        ts_pred = None
    f"Model fitted: {ts_model is not None}"
    return (ts_pred,)


@app.cell
def _(ts_pred):
    _interval_out = "Click 'Train ConformalForecaster' above"
    if ts_pred is not None:
        interval = ts_pred.interval(confidence=0.9)
        _interval_out = interval.head(10)
    _interval_out
    return


@app.cell
def _(horizon, split_idx, target_col, ts_pred, weather_clean):
    _plot_out = "No predictions yet"
    if ts_pred is not None:
        actuals_ts = weather_clean[target_col.value][
            split_idx + horizon.value : split_idx + horizon.value + ts_pred._n_samples
        ]
        _plot_out = ts_pred.plot(
            actuals=actuals_ts[:200],
            confidence_bands=[0.5, 0.8, 0.9],
            title=f"Conformal Forecast — {target_col.value} (h={horizon.value})",
        )
    _plot_out
    return


@app.cell
def _(mo):
    run_btn2 = mo.ui.run_button(label="Train QuantileForestForecaster")
    run_btn2
    return (run_btn2,)


@app.cell
def _(
    QuantileForestForecaster,
    horizon,
    run_btn2,
    target_col,
    test_ts,
    train_ts,
):
    if run_btn2.value:
        qf_model = QuantileForestForecaster(
            targets=target_col.value,
            horizon=horizon.value,
            n_estimators=100,
            auto_tune=False,
            random_state=42,
        )
        qf_model.fit(train_ts)
        qf_pred = qf_model.predict(test_ts)
    else:
        qf_model = None
        qf_pred = None
    f"QF Model fitted: {qf_model is not None}"
    return (qf_pred,)


@app.cell
def _(
    coverage_score,
    horizon,
    pl,
    qf_pred,
    split_idx,
    target_col,
    ts_pred,
    weather_clean,
    winkler_score,
):
    results = []
    for label, pred in [("ConformalForecaster", ts_pred), ("QuantileForest", qf_pred)]:
        if pred is None:
            continue
        actuals_slice = weather_clean[target_col.value][
            split_idx + horizon.value : split_idx + horizon.value + pred._n_samples
        ]
        interval_df = pred.interval(confidence=0.9)
        cov = coverage_score(actuals_slice, interval_df["lower"], interval_df["upper"])
        wink = winkler_score(
            actuals_slice, interval_df["lower"], interval_df["upper"], confidence=0.9
        )
        results.append({"model": label, "coverage": round(cov, 3), "winkler": round(wink, 2)})

    comparison = "Train models above to compare"
    if results:
        comparison = pl.DataFrame(results)
    comparison
    return


if __name__ == "__main__":
    app.run()
