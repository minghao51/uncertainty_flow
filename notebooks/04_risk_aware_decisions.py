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

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Risk-Aware Decision Making

Standard conformal prediction controls **coverage probability** — but real decisions often care about **cost**.

This notebook shows how to use `ConformalRiskControl` with custom risk functions:

- **Inventory management** — stockouts cost 10x more than excess inventory
- **Asymmetric loss** — underpredictions penalized more than overpredictions
- **Threshold penalty** — large errors are catastrophic
""")
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor

    from uncertainty_flow.risk import (
        ConformalRiskControl,
        asymmetric_loss,
        inventory_cost,
        threshold_penalty,
    )
    from uncertainty_flow.wrappers import ConformalRegressor

    return (
        ConformalRegressor,
        ConformalRiskControl,
        GradientBoostingRegressor,
        asymmetric_loss,
        inventory_cost,
        np,
        pl,
        threshold_penalty,
    )


@app.cell
def _(pl):
    df = pl.read_parquet("../data/energy_efficiency.parquet")
    f"Shape: {df.shape}"
    df.head(5)
    return (df,)


@app.cell
def _(df, pl):
    cols = df.columns
    target_col = cols[-1]
    n = df.height
    split = int(n * 0.7)
    split2 = int(n * 0.85)
    train_df = df[:split]
    calib_df = df[split:split2]
    test_df = df[split2:]
    f"Target: {target_col} | Train: {train_df.height} | Calib: {calib_df.height} | Test: {test_df.height}"
    return calib_df, n, split, split2, target_col, test_df, train_df


@app.cell
def _(mo):
    scenario = mo.ui.dropdown(
        options={
            "Inventory Cost (stockout=10x)": "inventory",
            "Asymmetric Loss (underpredict=3x)": "asymmetric",
            "Threshold Penalty (large errors)": "threshold",
        },
        value="inventory",
        label="Risk scenario",
    )
    target_risk = mo.ui.slider(
        start=0.01, stop=0.5, step=0.01, value=0.1, label="Target risk level"
    )
    mo.vstack([scenario, target_risk])
    return scenario, target_risk


@app.cell
def _(mo, scenario):
    description = {
        "inventory": "**Inventory scenario**: Stockouts cost $10/unit, holding costs $1/unit. We control expected inventory cost per prediction.",
        "asymmetric": "**Asymmetric scenario**: Underpredictions cost 3x more than overpredictions. Useful when missing high values is expensive.",
        "threshold": "**Threshold scenario**: Errors above 5 units are penalized at 10x rate. Models the cost of catastrophic mispredictions.",
    }
    mo.md(description.get(scenario.value, ""))
    return (description,)


@app.cell
def _(
    ConformalRegressor,
    ConformalRiskControl,
    GradientBoostingRegressor,
    asymmetric_loss,
    calib_df,
    inventory_cost,
    np,
    pl,
    scenario,
    target_col,
    target_risk,
    test_df,
    train_df,
):
    base_model = ConformalRegressor(
        base_model=GradientBoostingRegressor(random_state=42),
        auto_tune=False,
        random_state=42,
    )
    base_model.fit(train_df, target=target_col)

    risk_fns = {
        "inventory": inventory_cost(holding_cost=1.0, stockout_cost=10.0),
        "asymmetric": asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=3.0),
        "threshold": threshold_penalty(threshold=5.0, penalty_above=10.0, penalty_below=1.0),
    }

    risk_model = ConformalRiskControl(
        base_model=base_model,
        risk_function=risk_fns[scenario.value],
        target_risk=target_risk.value,
        random_state=42,
    )
    risk_model.fit(calib_df, target=target_col)

    risk_pred = risk_model.predict(test_df.drop(target_col))
    summary = risk_model.summary()
    threshold_val = risk_model.risk_threshold()
    f"Risk threshold: {threshold_val:.4f} | Target risk: {summary['target_risk']}"
    return base_model, risk_fns, risk_model, risk_pred, summary, threshold_val


@app.cell
def _(risk_pred):
    risk_pred.head(10)
    return


@app.cell
def _(np, pl, risk_pred):
    total = risk_pred.height
    flagged = risk_pred.filter(pl.col("exceeds_threshold")).height
    mean_risk = risk_pred["risk"].mean()
    max_risk = risk_pred["risk"].max()

    stats_df = pl.DataFrame(
        {
            "metric": [
                "Total predictions",
                "Flagged (exceeds threshold)",
                "Flag rate",
                "Mean risk",
                "Max risk",
            ],
            "value": [
                str(total),
                str(flagged),
                f"{flagged / total:.1%}",
                f"{mean_risk:.4f}",
                f"{max_risk:.4f}",
            ],
        }
    )
    stats_df
    return flagged, max_risk, mean_risk, stats_df, total


@app.cell
def _(np, pl, risk_pred):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    risks = risk_pred["risk"].to_numpy()
    axes[0].hist(risks, bins=40, alpha=0.7, edgecolor="black")
    axes[0].axvline(
        np.mean(risks), color="red", linestyle="--", label=f"Mean: {np.mean(risks):.3f}"
    )
    axes[0].set_title("Risk Distribution")
    axes[0].set_xlabel("Estimated Risk")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    preds = risk_pred["prediction"].to_numpy()
    flagged_mask = risk_pred["exceeds_threshold"].to_numpy()
    axes[1].scatter(
        range(len(preds)), preds, c=flagged_mask.astype(int), cmap="coolwarm", alpha=0.5, s=10
    )
    axes[1].set_title("Predictions (red = exceeds risk threshold)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Prediction")

    plt.tight_layout()
    fig
    return


@app.cell
def _(base_model, pl, target_col, test_df):
    base_pred = base_model.predict(test_df)
    base_interval = base_pred.interval(confidence=0.9)
    comparison = pl.concat(
        [
            pl.DataFrame({"source": "interval_lower", "value": base_interval["lower"][:20]}),
            pl.DataFrame({"source": "interval_upper", "value": base_interval["upper"][:20]}),
        ]
    )
    comparison
    return base_interval, base_pred, comparison


if __name__ == "__main__":
    app.run()
