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
    # Split Strategies: Sensible Defaults for Validation

    The `select_validation_plan()` function auto-selects the right validation strategy based on your task type and data size. No more deciding between `train_test_split` and `TimeSeriesSplit` — just tell it what kind of data you have.

    This notebook demonstrates:
    1. **One-call API** — one function, correct strategy
    2. **Task-type awareness** — temporal vs random, and the cost of getting it wrong
    3. **Small-data auto-CV** — KFold kicks in automatically when data is scarce
    4. **Hybrid mode** — outer holdout + inner CV for robust tuning
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.ensemble import GradientBoostingRegressor

    from uncertainty_flow.metrics import coverage_score, winkler_score
    from uncertainty_flow.utils.split import (
        select_validation_plan,
        RandomHoldoutSplit,
        TemporalHoldoutSplit,
    )
    from uncertainty_flow.utils import ValidationSplitPlan
    from uncertainty_flow.wrappers import ConformalForecaster

    DATA = Path(__file__).parent.parent / "data"

    return (
        DATA,
        ConformalForecaster,
        GradientBoostingRegressor,
        RandomHoldoutSplit,
        TemporalHoldoutSplit,
        ValidationSplitPlan,
        coverage_score,
        np,
        pl,
        plt,
        select_validation_plan,
        winkler_score,
    )


@app.cell
def _(mo):
    mo.md("## 1. The One-Call API")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Instead of manually slicing data and guessing the right split strategy, just call:

    ```python
    plan = select_validation_plan(data, task_type="tabular", random_state=42)
    train, val = plan.outer_split
    ```

    The function inspects your **task type** and **data size**, then selects the correct strategy. Let's see it in action.
    """)
    return


@app.cell
def _(DATA, pl, select_validation_plan):
    df_tabular = pl.read_parquet(DATA / "concrete.parquet")
    plan_tabular = select_validation_plan(df_tabular, task_type="tabular", random_state=42)
    plan_tabular.metadata
    return df_tabular, plan_tabular


@app.cell
def _(DATA, pl, select_validation_plan):
    df_ts = pl.read_parquet(DATA / "weather.parquet").drop_nulls()
    plan_ts = select_validation_plan(df_ts, task_type="time_series", random_state=42)
    plan_ts.metadata
    return df_ts, plan_ts


@app.cell
def _(mo):
    mo.md(r"""
    **Key observations:**
    - Tabular data → `random_holdout` (exchangeability assumption holds)
    - Time series → `temporal_holdout` (preserves temporal ordering, prevents leakage)
    - No manual `df[:split]` slicing needed
    """)
    return


@app.cell
def _(mo):
    mo.md("## 2. Task-Type Awareness: The Cost of Wrong Splits")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Using a **random split on time series data** creates **data leakage**: future observations leak into the training set, making the model look better than it really is.

    Below we compare two split strategies on weather time series:
    - ✅ **Temporal holdout** (correct) — trains on past, evaluates on future
    - ❌ **Random holdout** (wrong) — shuffles data, leaking future into training

    We then train a `ConformalForecaster` with each split and compare metrics.
    """)
    return


@app.cell
def _(
    ConformalForecaster,
    GradientBoostingRegressor,
    RandomHoldoutSplit,
    TemporalHoldoutSplit,
    coverage_score,
    df_ts,
    np,
    pl,
    winkler_score,
):
    target = "T (degC)"
    horizon = 6

    df_small = df_ts.select(target).head(2000)

    temporal = TemporalHoldoutSplit()
    temp_train, temp_val = temporal.split(df_small, 0.2)
    temp_model = ConformalForecaster(
        base_model=GradientBoostingRegressor(random_state=42),
        targets=target,
        horizon=horizon,
        lags=3,
        calibration_size=0.2,
        auto_tune=False,
        random_state=42,
    )
    temp_model.fit(temp_train)
    temp_pred = temp_model.predict(temp_val)
    temp_interval = temp_pred.interval(0.9)
    temp_y_true = temp_val[target].to_numpy()[-len(temp_interval) :]
    temp_lower = temp_interval["lower"].to_numpy()
    temp_upper = temp_interval["upper"].to_numpy()
    temp_coverage = coverage_score(temp_y_true, temp_lower, temp_upper)
    temp_winkler = winkler_score(temp_y_true, temp_lower, temp_upper, confidence=0.9)

    random = RandomHoldoutSplit(random_state=42)
    rand_train, rand_val = random.split(df_small, 0.2)
    rand_model = ConformalForecaster(
        base_model=GradientBoostingRegressor(random_state=42),
        targets=target,
        horizon=horizon,
        lags=3,
        calibration_size=0.2,
        auto_tune=False,
        random_state=42,
    )
    rand_model.fit(rand_train)
    rand_pred = rand_model.predict(rand_val)
    rand_interval = rand_pred.interval(0.9)
    rand_y_true = rand_val[target].to_numpy()[-len(rand_interval) :]
    rand_lower = rand_interval["lower"].to_numpy()
    rand_upper = rand_interval["upper"].to_numpy()
    rand_coverage = coverage_score(rand_y_true, rand_lower, rand_upper)
    rand_winkler = winkler_score(rand_y_true, rand_lower, rand_upper, confidence=0.9)
    return (
        df_small,
        rand_coverage,
        rand_lower,
        rand_model,
        rand_pred,
        rand_train,
        rand_upper,
        rand_val,
        rand_winkler,
        rand_y_true,
        random,
        target,
        temp_coverage,
        temp_interval,
        temp_lower,
        temp_model,
        temp_pred,
        temp_train,
        temp_upper,
        temp_val,
        temp_winkler,
        temp_y_true,
        temporal,
    )


@app.cell
def _(mo, pl, rand_coverage, rand_winkler, temp_coverage, temp_winkler):
    metrics_df = pl.DataFrame({
        "Strategy": ["Temporal Holdout (correct)", "Random Holdout (leakage)"],
        "Coverage (90% target)": [f"{temp_coverage:.3f}", f"{rand_coverage:.3f}"],
        "Winkler Score": [f"{temp_winkler:.3f}", f"{rand_winkler:.3f}"],
    })
    mo.md(f"""
    #### Metrics Comparison
    **What happened?** The random split lets future data leak into training, so the model appears better calibrated (coverage closer to 90%). But when deployed on truly out-of-time data, the temporal model tells the truth — the random split's "good" coverage is an illusion.
    """)
    metrics_df
    return (metrics_df,)


@app.cell
def _(mo):
    mo.md("## 3. Small-Data Auto-CV")
    return


@app.cell
def _(mo):
    mo.md(r"""
    When your dataset has **fewer than 250 rows**, `select_validation_plan` automatically switches from a single holdout to **KFold cross-validation**. This gives more stable tuning at the cost of some bias.

    Let's see it in action — subset the concrete data to just 120 rows.
    """)
    return


@app.cell
def _(df_tabular, select_validation_plan):
    df_small_tab = df_tabular.head(120)
    plan_small = select_validation_plan(df_small_tab, task_type="tabular", random_state=42)
    plan_small.metadata
    return df_small_tab, plan_small


@app.cell
def _(mo):
    mo.md(r"""
    Notice: `strategy_name` changed from `random_holdout` to `kfold_cv`, and `n_splits` > 1. The plan now includes `inner_splits` — multiple train/validation folds.

    Let's compare tuning stability between single holdout and CV for a small dataset.
    """)
    return


@app.cell
def _(
    GradientBoostingRegressor,
    RandomHoldoutSplit,
    df_small_tab,
    np,
    select_validation_plan,
    target,
):
    _target = "strength"

    single_holdout_scores = []
    cv_scores = []

    for seed in range(20):
        plan = select_validation_plan(df_small_tab, task_type="tabular", random_state=seed)
        if plan.inner_splits:
            fold_scores = []
            for split_train, split_val in plan.inner_splits:
                model = GradientBoostingRegressor(n_estimators=30, random_state=seed)
                model.fit(
                    split_train.select([c for c in split_train.columns if c != _target]).to_numpy(),
                    split_train[_target].to_numpy(),
                )
                preds = model.predict(
                    split_val.select([c for c in split_val.columns if c != _target]).to_numpy()
                )
                err = float(np.mean(np.abs(preds - split_val[_target].to_numpy())))
                fold_scores.append(err)
            cv_scores.append(float(np.mean(fold_scores)))

        splitter = RandomHoldoutSplit(random_state=seed)
        s_train, s_val = splitter.split(df_small_tab, 0.2)
        model = GradientBoostingRegressor(n_estimators=30, random_state=seed)
        model.fit(
            s_train.select([c for c in s_train.columns if c != _target]).to_numpy(),
            s_train[_target].to_numpy(),
        )
        preds = model.predict(
            s_val.select([c for c in s_val.columns if c != _target]).to_numpy()
        )
        score = float(np.mean(np.abs(preds - s_val[_target].to_numpy())))
        single_holdout_scores.append(score)

    return (
        cv_scores,
        fold_scores,
        model,
        plan,
        preds,
        score,
        seed,
        single_holdout_scores,
        splitter,
    )


@app.cell
def _(cv_scores, np, plt, single_holdout_scores):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([single_holdout_scores, cv_scores], tick_labels=["Single Holdout", "KFold CV (auto)"])
    ax.set_ylabel("MAE")
    ax.set_title("Tuning Score Variance: Holdout vs CV on Small Dataset")
    ax.grid(axis="y", alpha=0.3)
    fig
    return (ax, fig)


@app.cell
def _(cv_scores, np, single_holdout_scores):
    _holdout_var = float(np.var(single_holdout_scores))
    _cv_var = float(np.var(cv_scores))
    _reduction = (1 - _cv_var / _holdout_var) * 100
    f"Holdout variance: {_holdout_var:.6f}  |  CV variance: {_cv_var:.6f}  |  Variance reduction: {_reduction:.0f}%"
    return


@app.cell
def _(mo):
    mo.md("## 4. Hybrid Mode: Outer Holdout + Inner CV")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Sometimes you want **both**: a clean outer holdout for final evaluation, plus inner CV folds for robust hyperparameter tuning.

    Enable this with `hybrid_mode=True`:
    ```python
    plan = select_validation_plan(data, task_type="tabular",
                                   hybrid_mode=True, random_state=42)
    ```
    - `plan.outer_split` → held-out test set
    - `plan.inner_splits` → CV folds from the training portion
    """)
    return


@app.cell
def _(df_tabular, select_validation_plan):
    plan_hybrid = select_validation_plan(
        df_tabular, task_type="tabular", hybrid_mode=True, random_state=42
    )
    plan_hybrid.metadata
    return (plan_hybrid,)


@app.cell
def _(plan_hybrid):
    f"Inner CV folds: {len(plan_hybrid.inner_splits)}  |  Outer train rows: {len(plan_hybrid.outer_split[0])}  |  Outer test rows: {len(plan_hybrid.outer_split[1])}"
    return


@app.cell
def _(mo):
    mo.md("## 5. Hybrid Mode for Time Series")
    return


@app.cell
def _(df_ts, select_validation_plan):
    plan_hybrid_ts = select_validation_plan(
        df_ts.head(2000), task_type="time_series", hybrid_mode=True, random_state=42
    )
    plan_hybrid_ts.metadata
    return (plan_hybrid_ts,)


@app.cell
def _(mo):
    mo.md(r"""
    For time series, hybrid mode means:
    - **Temporal outer split** (last 20% held out) → preserves temporal ordering
    - **Random inner CV folds** on the training portion → enables robust hyperparameter tuning

    This is the best of both worlds: no data leakage in the final evaluation, plus stable tuning across multiple train/validation splits.
    """)
    return


@app.cell
def _(mo):
    mo.md("## 6. Summary: When to Use What")
    return


@app.cell
def _(mo, pl):
    summary_df = pl.DataFrame({
        "Task Type": ["Tabular (large)", "Tabular (small, <250 rows)", "Time Series"],
        "Auto Strategy": ["Random Holdout", "KFold CV", "Temporal Holdout"],
        "Why": ["Exchangeability holds; single split is fast", "Single split too noisy; CV reduces variance", "Preserves ordering; prevents leakage"],
        "Hybrid": ["Adds CV for tuning stability", "CV + outer holdout for final eval", "Temporal outer + OOS inner CV"],
    })
    mo.md("**Bottom line:** Use `select_validation_plan(data, task_type=...)` as your default split strategy. It handles the edge cases so you don't have to think about them.")
    summary_df
    return (summary_df,)


if __name__ == "__main__":
    app.run()
