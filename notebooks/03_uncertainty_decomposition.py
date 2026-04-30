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
    # Uncertainty Decomposition: Aleatoric vs. Epistemic

    Not all uncertainty is equal. This notebook decomposes prediction uncertainty into:

    - **Aleatoric uncertainty** — irreducible noise in the data (e.g., measurement error)
    - **Epistemic uncertainty** — reducible uncertainty from limited data or model capacity

    We use the `EnsembleDecomposition` class with bootstrap refitting to quantify each component
    on the **synthetic heteroscedastic** dataset (where noise varies with input features).
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor

    from uncertainty_flow.decomposition import EnsembleDecomposition
    from uncertainty_flow.wrappers import ConformalRegressor

    return (
        ConformalRegressor,
        EnsembleDecomposition,
        GradientBoostingRegressor,
        pl,
    )


@app.cell
def _(pl):
    df = pl.read_parquet("../data/synthetic_heteroscedastic.parquet")
    f"Shape: {df.shape}"
    df.head(10)
    return (df,)


@app.cell
def _(mo):
    n_bootstrap = mo.ui.slider(start=3, stop=30, step=1, value=10, label="Bootstrap iterations")
    confidence = mo.ui.slider(start=0.5, stop=0.99, step=0.01, value=0.9, label="Confidence level")
    sample_size = mo.ui.slider(
        start=100, stop=2000, step=100, value=500, label="Evaluation sample size"
    )
    mo.vstack([n_bootstrap, confidence, sample_size])
    return confidence, n_bootstrap, sample_size


@app.cell
def _(mo):
    run_decomp = mo.ui.run_button(label="Run Decomposition")
    run_decomp
    return (run_decomp,)


@app.cell
def _(
    ConformalRegressor,
    EnsembleDecomposition,
    GradientBoostingRegressor,
    confidence,
    df,
    n_bootstrap,
    run_decomp,
    sample_size,
):
    if run_decomp.value:
        n = df.height
        split_idx = int(n * 0.7)
        train_df = df[:split_idx]
        eval_df = df[split_idx:]
        eval_sample = eval_df.sample(n=min(sample_size.value, eval_df.height), seed=42)

        def model_factory():
            return ConformalRegressor(
                base_model=GradientBoostingRegressor(random_state=42),
                auto_tune=False,
                random_state=42,
            )

        decomp = EnsembleDecomposition(
            model_factory=model_factory,
            train_data=train_df,
            target="y",
            confidence=confidence.value,
            n_bootstrap=n_bootstrap.value,
            random_state=42,
        )

        overall = decomp.decompose(eval_sample)
        by_sample = decomp.decompose_by_sample(eval_sample)
    else:
        overall = None
        by_sample = None
        eval_sample = None
    f"Decomposition complete: {overall is not None}"
    return by_sample, overall


@app.cell
def _(mo, overall):
    _summary = "Click 'Run Decomposition' above"
    if overall is not None:
        aleatoric_pct = overall["aleatoric"] / overall["total"] * 100
        epistemic_pct = overall["epistemic"] / overall["total"] * 100
        _summary = mo.md(f"""
    | Component | Value |
    |---|---|
    | **Aleatoric** (data noise) | {overall["aleatoric"]:.4f} |
    | **Epistemic** (model uncertainty) | {overall["epistemic"]:.4f} |
    | **Total** | {overall["total"]:.4f} |

    **Interpretation**: Aleatoric = {aleatoric_pct:.1f}% of total uncertainty.
    Reducible with more data: {epistemic_pct:.1f}%.
    """)
    _summary
    return


@app.cell
def _(by_sample, pl):
    _stats = "No decomposition results yet"
    if by_sample is not None:
        summary_stats = by_sample.select(
            [
                pl.col("aleatoric").mean().alias("mean_aleatoric"),
                pl.col("epistemic").mean().alias("mean_epistemic"),
                pl.col("total").mean().alias("mean_total"),
                pl.col("aleatoric").std().alias("std_aleatoric"),
                pl.col("epistemic").std().alias("std_epistemic"),
            ]
        )
        _stats = summary_stats
    _stats
    return


@app.cell
def _(by_sample):
    _plot_result = "No results to plot"
    if by_sample is not None:
        import matplotlib.pyplot as plt

        _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(_axes, ["aleatoric", "epistemic", "total"]):
            ax.hist(by_sample[col].to_numpy(), bins=30, alpha=0.7, edgecolor="black")
            ax.set_title(f"{col.title()} Distribution")
            ax.set_xlabel("Uncertainty")
            ax.set_ylabel("Count")
        plt.tight_layout()
        _plot_result = _fig
    _plot_result
    return


@app.cell
def _(by_sample):
    _sorted = "No results yet"
    if by_sample is not None:
        _sorted = by_sample.sort("total", descending=True).head(10)
    _sorted
    return


if __name__ == "__main__":
    app.run()
