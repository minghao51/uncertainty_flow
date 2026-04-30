# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars>=0.20",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Methods & Benchmarks Overview

    A guide to every model in `uncertainty_flow`, when to use it, and how they compare on real benchmarks.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    raw = pl.read_csv("../docs/benchmarks/comparison_table.csv", try_parse_dates=False)
    f"Loaded {raw.height} benchmark rows across {raw['dataset'].n_unique()} datasets and {raw['model'].n_unique()} models"
    return (raw,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Methodology Reference

    Each model returns a `DistributionPrediction` object. Here is what each one does and when to reach for it.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ConformalRegressor

    - **Family:** Tabular wrapper
    - **Coverage:** ✅ Mathematical guarantee (exchangeability assumption) — Post-sort non-crossing
    - **Use when:** You already have a scikit-learn regressor and want calibrated intervals on tabular data.
    - **Method:** Splits data into train + calibration, computes residual quantiles on calibration, adds them to test predictions.
    - **Multivariate:** No
    - **Dependencies:** `scikit-learn` only
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### ConformalForecaster

    - **Family:** Forecasting wrapper
    - **Coverage:** ✅ Mathematical guarantee (with temporal correction) — Post-sort non-crossing
    - **Use when:** Forecasting time series where interval calibration matters more than raw speed.
    - **Method:** Constructs lag features, splits temporally, computes residual quantiles from the end of training.
    - **Multivariate:** Yes (copula-aware joint sampling)
    - **Dependencies:** `scikit-learn` only
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### QuantileForestForecaster

    - **Family:** Native tree model
    - **Coverage:** ⚠️ Empirical only — By-construction non-crossing
    - **Use when:** You need a fast, interpretable quantile forecaster without deep learning.
    - **Method:** Random Forest stores full leaf distributions, enabling true quantile computation (not split-conformal).
    - **Multivariate:** Yes (copula-aware joint sampling)
    - **Dependencies:** `scikit-learn` only
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### DeepQuantileNet / DeepQuantileNetTorch

    - **Family:** Native neural model
    - **Coverage:** ⚠️ Empirical only — Post-sort non-crossing
    - **Use when:** Nonlinear signals that tree models underfit. Torch backend for GPU/accelerator support.
    - **Method:** Multi-quantile MLP with shared trunk (sklearn) or PyTorch backend.
    - **Multivariate:** Yes
    - **Dependencies:** `scikit-learn` (+ `torch` for Torch variant)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### BayesianQuantileRegressor

    - **Family:** Bayesian (optional numpyro)
    - **Coverage:** Posterior-based — Post-sort non-crossing
    - **Use when:** You need full posterior distributions with credible intervals (not discrete quantiles).
    - **Warning:** Default settings over-regularize (horseshoe prior). Requires hyperparameter tuning.
    - **Multivariate:** No
    - **Dependencies:** `numpyro` + `jax` (optional)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Additional Modules

    | Module | Purpose | Coverage |
    |--------|---------|----------|
    | **CausalUncertaintyEstimator** | Treatment effect (CATE/ATE) estimation with conformal CIs | Conformal on CATE |
    | **CrossModalAggregator** | Combine predictions from separate feature groups | Inherited from base models |
    | **ConformalRiskControl** | Control expected loss instead of coverage | Risk-bounded |
    | **EnsembleDecomposition** | Bootstrap-based aleatoric/epistemic decomposition | Refit-based |
    | **FeatureLeverageAnalyzer** | Score features by impact on interval width | Correlation-based |
    | **UncertaintyExplainer** | Find feature changes that reduce uncertainty | Counterfactual search |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Overall Rankings

    Models ranked by average Winkler score @ 90% confidence across all datasets.
    """)
    return


@app.cell
def _(mo, pl, raw):
    _overall = (
        raw.group_by("model")
        .agg(
            [
                pl.col("winkler_90").mean().alias("avg_winkler"),
                pl.col("coverage_90").mean().alias("avg_coverage"),
                pl.col("mae").mean().alias("avg_mae"),
                pl.col("calibration_error").mean().alias("avg_cal_error"),
                pl.col("timing_mean").mean().alias("avg_time_sec"),
                pl.col("dataset").count().alias("datasets"),
            ]
        )
        .sort("avg_winkler")
    )
    mo.ui.table(_overall)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## By Dataset

    Select a dataset below to see per-dataset model comparisons.
    """)
    return


@app.cell
def _(mo, raw):
    _datasets = sorted(raw["dataset"].unique().to_list())
    dataset_selector = mo.ui.dropdown(options=_datasets, value="weather", label="Dataset")
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(dataset_selector, mo, pl, raw):
    _df = raw.filter(pl.col("dataset") == dataset_selector.value).sort("winkler_90")
    mo.ui.table(_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive Comparison

    Pick a dataset and two metrics to compare models across.
    """)
    return


@app.cell
def _(mo, raw):
    _models = sorted(raw["model"].unique().to_list())
    _metrics_list = [
        "winkler_90",
        "coverage_90",
        "sharpness_90",
        "crps",
        "mae",
        "calibration_error",
        "timing_mean",
    ]
    _ds = sorted(raw["dataset"].unique().to_list())

    x_metric = mo.ui.dropdown(options=_metrics_list, value="winkler_90", label="X-axis metric")
    y_metric = mo.ui.dropdown(options=_metrics_list, value="coverage_90", label="Y-axis metric")
    scatter_ds = mo.ui.dropdown(options=_ds, value="weather", label="Dataset")
    mo.vstack([scatter_ds, x_metric, y_metric])
    return scatter_ds, x_metric, y_metric


@app.cell
def _(pl, raw, scatter_ds, x_metric, y_metric):
    import matplotlib.pyplot as _plt

    _df = raw.filter(pl.col("dataset") == scatter_ds.value)
    _x_vals = _df[x_metric.value].to_numpy()
    _y_vals = _df[y_metric.value].to_numpy()
    _labels = _df["model"].to_list()

    _fig, _ax = _plt.subplots(figsize=(10, 7))
    _ax.scatter(_x_vals, _y_vals, s=80, alpha=0.8)

    for i, label in enumerate(_labels):
        _ax.annotate(
            label,
            (_x_vals[i], _y_vals[i]),
            fontsize=8,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    _ax.set_xlabel(x_metric.value.replace("_", " ").title())
    _ax.set_ylabel(y_metric.value.replace("_", " ").title())
    _ax.set_title(f"Model Comparison — {scatter_ds.value.title()}")
    _ax.grid(True, alpha=0.3)
    _plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Key Findings

    ### Finding 1: deep-quantile-torch Dominates on Low-Dimensional Data

    Best Winkler on weather (0.0149) and exchange_rate (0.1351). Captures nonlinear relationships tree-based methods miss on smaller feature spaces. Struggles on 320-feature electricity (793.21 Winkler).

    ### Finding 2: deep-quantile Has the Best Calibration

    Near-perfect calibration error on weather (0.001) and exchange_rate (0.003). 90.1% coverage on weather is essentially exact. Wider intervals than torch version.

    ### Finding 3: quantile-forest Remains Best for High-Dimensional Data

    On electricity (320 features), Winkler 321.37 — less than half the next best. 90.7% coverage, 0.007 calibration error.

    ### Finding 4: BayesianQuantileRegressor Fails on Default Settings

    0% coverage on weather/electricity, 0.2% on exchange_rate. Horseshoe prior over-regularizes. Requires significant tuning.

    ### Finding 5: conformal-regressor and gradient-boosting Are Identical

    Both wrap `GradientBoostingRegressor` in `ConformalRegressor` with the same defaults → byte-identical.

    ### Finding 6: Simple Baselines Are Sanity Checks Only

    `naive-forecast` and `moving-average` consistently rank last. Near-zero computation time is their only advantage.

    ### Finding 7: Speed vs Quality Tradeoff

    | Speed Tier | Models | Avg Winkler | Avg Time |
    |-----------|--------|------------|---------|
    | Ultra-fast (<5ms) | naive, moving-average | 750.71 | 0.002s |
    | Fast (<50ms) | linear, ridge | 179.08 | 0.015s |
    | Medium (<200ms) | random-forest, conformal-\* | 140.05 | 0.21s |
    | Slower (>1s) | quantile-forest, deep-quantile-\* | 217.27 | 1.42s |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Recommendations

    | Scenario | Model | Why |
    |----------|-------|-----|
    | **Low-dimensional, best accuracy** | `deep-quantile-torch` | Best Winkler on weather & exchange_rate |
    | **High-dimensional, best accuracy** | `quantile-forest` | Dominates on electricity (320 features) |
    | **Best calibration** | `deep-quantile` | Near-exact 90% coverage across datasets |
    | **Fast production** | `conformal-forecaster` | Best quality in medium speed tier |
    | **Guaranteed coverage** | `conformal-regressor` or `conformal-forecaster` | Mathematical coverage guarantee |
    | **Asymmetric costs** | `ConformalRiskControl` | Control expected loss, not coverage |
    | **Full posterior** | `BayesianQuantileRegressor` | Needs careful tuning first |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Source

    Benchmark results from `docs/benchmarks/README.md` (April 26, 2026). 1,000 obs per dataset, horizon=3, auto-tuning disabled. See `docs/benchmarks/comparison_table.csv` for full raw data.
    """)
    return


if __name__ == "__main__":
    app.run()
