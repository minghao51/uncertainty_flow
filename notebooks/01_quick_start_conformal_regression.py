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
    # Quick Start: Conformal Regression

    Wrap any scikit-learn model with **statistically rigorous coverage guarantees** using `ConformalRegressor`.

    This notebook demonstrates the core workflow:
    1. Load tabular data (concrete compressive strength)
    2. Wrap a GradientBoostingRegressor with conformal prediction
    3. Extract intervals, quantiles, and samples from predictions
    4. Evaluate calibration
    """)
    return


@app.cell
def _():
    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    from uncertainty_flow.metrics import coverage_score, winkler_score
    from uncertainty_flow.utils.split import select_validation_plan
    from uncertainty_flow.wrappers import ConformalRegressor

    return (
        ConformalRegressor,
        GradientBoostingRegressor,
        Ridge,
        coverage_score,
        pl,
        select_validation_plan,
        winkler_score,
    )


@app.cell
def _(pl):
    df = pl.read_parquet("../data/concrete.parquet")
    df.head(10)
    return (df,)


@app.cell
def _(mo):
    target_col = mo.ui.dropdown(
        options=["strength"],
        value="strength",
        label="Target column",
    )
    target_col
    return (target_col,)


@app.cell
def _(df, select_validation_plan, target_col):
    plan = select_validation_plan(df, task_type="tabular", random_state=42)
    train_df, test_df = plan.outer_split
    actuals = test_df[target_col.value]
    f"Strategy: {plan.metadata.strategy_name} | Train: {len(train_df)} | Test: {len(test_df)}"
    return actuals, plan, test_df, train_df


@app.cell
def _(mo):
    base_model_choice = mo.ui.dropdown(
        options=["GradientBoosting", "Ridge"],
        value="GradientBoosting",
        label="Base model",
    )
    coverage_slider = mo.ui.slider(
        start=0.5,
        stop=0.99,
        step=0.01,
        value=0.9,
        label="Coverage target",
    )
    mo.vstack([base_model_choice, coverage_slider])
    return base_model_choice, coverage_slider


@app.cell
def _(
    ConformalRegressor,
    GradientBoostingRegressor,
    Ridge,
    base_model_choice,
    coverage_slider,
    target_col,
    train_df,
):
    base_cls = GradientBoostingRegressor if base_model_choice.value == "GradientBoosting" else Ridge
    base = base_cls(random_state=42)
    model = ConformalRegressor(
        base_model=base,
        coverage_target=coverage_slider.value,
        auto_tune=True,
        random_state=42,
    )
    model.fit(train_df, target=target_col.value)
    model
    return (model,)


@app.cell
def _(model, test_df):
    pred = model.predict(test_df)
    pred
    return (pred,)


@app.cell
def _(coverage_slider, pred):
    interval_df = pred.interval(confidence=coverage_slider.value)
    interval_df.head(10)
    return


@app.cell
def _(pred):
    quantiles_df = pred.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles_df.head(10)
    return


@app.cell
def _(pred):
    mean_series = pred.mean()
    mean_series.head(10)
    return


@app.cell
def _(actuals, coverage_score, coverage_slider, pred, winkler_score):
    interval = pred.interval(confidence=coverage_slider.value)
    coverage = coverage_score(actuals, interval["lower"], interval["upper"])
    winkler = winkler_score(
        actuals, interval["lower"], interval["upper"], confidence=coverage_slider.value
    )
    f"Coverage: {coverage:.3f} | Winkler score: {winkler:.2f}"
    return


@app.cell
def _(actuals, coverage_slider, pred, target_col):
    fig = pred.plot(
        actuals=actuals,
        confidence_bands=[0.5, 0.8, coverage_slider.value],
        title=f"Conformal Prediction — {target_col.value}",
    )
    return


@app.cell
def _(model, target_col, test_df):
    calibration = model.calibration_report(test_df, test_df[target_col.value])
    calibration
    return


@app.cell
def _(model):
    drivers = model.uncertainty_drivers_
    drivers
    return


if __name__ == "__main__":
    app.run()
