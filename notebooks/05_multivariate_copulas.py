# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars>=0.20",
#     "scikit-learn>=1.3",
#     "scipy>=1.11",
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
    # Multivariate Copulas & Cross-Modal Aggregation

    When forecasting **multiple targets simultaneously**, their joint distribution matters.

    This notebook explores:

    1. **Copula families** — model different types of inter-target dependence
    2. **Auto-selection** — let BIC choose the best copula from data
    3. **Joint sampling** — generate correlated samples from copula-aware predictions
    4. **Cross-modal aggregation** — combine predictions from different feature groups

    We use the **synthetic multivariate** dataset with 3 target columns (y1, y2, y3).
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl

    from uncertainty_flow.multivariate import GaussianCopula
    from uncertainty_flow.multivariate.copula import (
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
        auto_select_copula,
    )
    from uncertainty_flow.utils.split import select_validation_plan

    return (
        ClaytonCopula,
        FrankCopula,
        GaussianCopula,
        GumbelCopula,
        auto_select_copula,
        np,
        pl,
        select_validation_plan,
    )


@app.cell
def _(pl):
    df = pl.read_parquet("../data/synthetic_multivariate.parquet")
    f"Shape: {df.shape}"
    df.head(5)
    return (df,)


@app.cell
def _(df, select_validation_plan):
    targets = ["y1", "y2", "y3"]
    features = [c for c in df.columns if c not in targets]
    plan = select_validation_plan(df, task_type="tabular", holdout_fraction=0.2, random_state=42)
    train, test = plan.outer_split
    f"Plan: {plan.metadata.strategy_name} | Features: {features} | Targets: {targets} | Train: {len(train)} | Test: {len(test)}"
    return plan, targets, test, train


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Copula Family Comparison
    """)
    return


@app.cell
def _(
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    pl,
    targets,
    train,
):
    residuals = train.select(targets).to_numpy()

    families = {
        "Gaussian (linear)": GaussianCopula(),
        "Clayton (lower tail)": ClaytonCopula(),
        "Gumbel (upper tail)": GumbelCopula(),
        "Frank (symmetric)": FrankCopula(),
    }

    fitted = {}
    for name, copula in families.items():
        try:
            copula.fit(residuals[:, :2])
            fitted[name] = copula
        except Exception:
            fitted[name] = None

    results = []
    for name, copula in fitted.items():
        if copula is not None:
            ll = copula.log_likelihood(residuals[:, :2])
            results.append(
                {"family": name, "theta": round(copula.theta_, 4), "log_likelihood": round(ll, 2)}
            )
        else:
            results.append({"family": name, "theta": None, "log_likelihood": None})

    pl.DataFrame(results)
    return (residuals,)


@app.cell
def _(auto_select_copula, residuals):
    best_family = auto_select_copula(residuals[:, :2])
    f"BIC-selected copula family: **{best_family}**"
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Joint Sampling from Copulas
    """)
    return


@app.cell
def _(GaussianCopula, np, residuals):
    import matplotlib.pyplot as _plt

    copula_gauss = GaussianCopula()
    copula_gauss.fit(residuals[:, :2])

    n_q = 11
    quantile_levels = np.linspace(0.05, 0.95, n_q)
    marginals = np.zeros((1, 2, n_q))
    for t in range(2):
        for qi, q in enumerate(quantile_levels):
            marginals[0, t, qi] = np.quantile(residuals[:, t], q)

    joint_samples = copula_gauss.sample(
        marginals,
        n_samples=1000,
        quantile_levels=quantile_levels,
        random_state=42,
    )

    _fig, _ax = _plt.subplots(figsize=(8, 6))
    if joint_samples.ndim == 3:
        s1 = joint_samples[0, :, 0]
        s2 = joint_samples[0, :, 1]
    else:
        s1 = joint_samples[:, 0]
        s2 = joint_samples[:, 1]
    _ax.scatter(s1, s2, alpha=0.3, s=5)
    _ax.set_xlabel("Target y1")
    _ax.set_ylabel("Target y2")
    _ax.set_title(f"Gaussian Copula Joint Samples (n={len(s1)})")
    _plt.tight_layout()
    _fig
    return


@app.cell
def _(GaussianCopula, residuals):
    import matplotlib.pyplot as _plt2

    copula_full = GaussianCopula()
    copula_full.fit(residuals)
    corr = copula_full.correlation_matrix_

    _fig2, _ax2 = _plt2.subplots(figsize=(6, 5))
    _im = _ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    _ax2.set_xticks(range(3))
    _ax2.set_yticks(range(3))
    _ax2.set_xticklabels(["y1", "y2", "y3"])
    _ax2.set_yticklabels(["y1", "y2", "y3"])
    _ax2.set_title("Inter-Target Correlation Matrix")
    _fig2.colorbar(_im, ax=_ax2)
    _plt2.tight_layout()
    _fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. QuantileForestForecaster with Copula-Aware Joint Forecasts
    """)
    return


@app.cell
def _(mo):
    run_qf = mo.ui.run_button(label="Train Multivariate Forecaster")
    run_qf
    return (run_qf,)


@app.cell
def _(run_qf, targets, test, train):
    if run_qf.value:
        from uncertainty_flow.models import QuantileForestForecaster

        mv_model = QuantileForestForecaster(
            targets=targets,
            horizon=1,
            n_estimators=100,
            copula_family="auto",
            auto_tune=False,
            random_state=42,
        )
        mv_model.fit(train)
        mv_pred = mv_model.predict(test)
        mv_interval = mv_pred.interval(confidence=0.9)
        mv_samples = mv_pred.sample(1000, random_state=42)
    else:
        mv_model = None
        mv_pred = None
        mv_interval = None
        mv_samples = None
    f"Multivariate model fitted: {mv_model is not None}"
    return mv_interval, mv_pred


@app.cell
def _(mv_interval):
    _interval_out = "Click 'Train Multivariate Forecaster' above"
    if mv_interval is not None:
        _interval_out = mv_interval.head(10)
    _interval_out
    return


@app.cell
def _(mv_pred):
    _decomp_out = "No predictions yet"
    if mv_pred is not None:
        decomp = mv_pred.uncertainty_decomposition(confidence=0.9)
        _decomp_out = f"Aleatoric: {decomp['aleatoric']:.4f} | Epistemic: {decomp['epistemic']:.4f} | Total: {decomp['total']:.4f}"
    _decomp_out
    return


if __name__ == "__main__":
    app.run()
