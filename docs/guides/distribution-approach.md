# Distribution-First Concepts

This guide is the narrative overview of how `uncertainty_flow` is meant to be used. It complements the architecture and API docs by focusing on concepts and actual usage patterns.

## The Core Idea

`uncertainty_flow` is built around one main choice: `predict()` should return a distribution-oriented object, not just a point estimate.

That means downstream code starts from:

```python
pred = model.predict(df_test)
```

and then chooses the view it needs:

```python
pred.quantile([0.1, 0.5, 0.9])
pred.interval(0.9)
pred.mean()
pred.sample(100)
pred.plot()
```

This is the most important usage idea in the project. Different model families can vary a lot internally, but they converge to the same downstream contract.

## Two Main Workflows

### Tabular regression

For unordered feature tables, the usual path is `ConformalRegressor`. It lets you wrap an existing scikit-learn style regressor and add calibrated uncertainty without changing the rest of your modeling workflow too much.

### Forecasting

For ordered time series, the common choices are `ConformalForecaster`, `QuantileForestForecaster`, and optional higher-capacity models such as `TransformerForecaster`. These workflows care about temporal splits and, in the multivariate case, joint uncertainty across targets.

## Why `DistributionPrediction` Matters

Without a shared output type, every model family would need its own interval helpers, quantile access patterns, and plotting conventions. `DistributionPrediction` keeps that complexity out of user code.

In practice, it means:

- model switching is cheaper
- evaluation code can be reused across model families
- downstream consumers can ask for intervals, samples, or quantiles the same way every time

## Intervals, Samples, and Diagnostics

Most real usage falls into three buckets:

- interval extraction for decisions or reporting
- sampling for simulation or scenario analysis, including copula-aware joint draws for multivariate predictions
- diagnostics to understand whether uncertainty quality is trustworthy

That last part matters. Generating an interval is easy; understanding whether it is calibrated is the hard part. The docs around calibration and model guarantees are meant to be read together with this guide.

## Multivariate Uncertainty

When predicting several targets together, independent intervals are often misleading. If targets move together, joint uncertainty needs a dependence model.

`uncertainty_flow` handles that by pairing:

- per-target marginal predictions
- a copula layer for dependence structure

This is why multivariate support is more than just returning several independent lower and upper bounds.

## Persistence In The Workflow

Because `predict()` returns a stable `DistributionPrediction` surface across model families, model persistence is also uniform:

```python
model.save("models/example.uf")
loaded = type(model).load("models/example.uf")
pred = loaded.predict(df_test)
```

That lets you persist a fitted calibrated or copula-backed model without changing downstream consumer code.

## A Typical Mental Model

You can think about the library in four layers:

1. Pick a model family that matches the problem shape and guarantee needs.
2. Fit on Polars inputs while the library handles array conversion internally.
3. Predict into `DistributionPrediction`.
4. Inspect calibration and uncertainty diagnostics before trusting the result.

## Where To Go Next

- For package structure and internals, read [../architecture/overview.md](../architecture/overview.md).
- For plotting and visual output examples, read [./charting.md](./charting.md).
- For model-by-model tradeoffs, read [./models.md](./models.md).
- For calibration interpretation, read [./calibration.md](./calibration.md).
- For exact method surfaces, read [../api/spec.md](../api/spec.md).
