# Models Guide

This document is the current guide to model selection and guarantee tradeoffs.

## Guarantee Definitions

| Term | Meaning |
|---|---|
| Mathematical coverage | Interval coverage is backed by a conformal-style guarantee under stated assumptions. |
| Empirical coverage | Coverage must be validated on held-out data and can drift under shift. |
| By construction | Quantile ordering comes from the model form itself. |
| Post-sort | Quantile ordering is repaired after inference rather than enforced by the underlying model. |

## Model Matrix

| Model | Family | Coverage | Non-crossing | Multivariate | Best Fit |
|---|---|---|---|---|---|
| `ConformalRegressor` | Tabular wrapper | Mathematical | Post-sort | No | Add calibrated intervals to an existing regressor |
| `ConformalForecaster` | Forecasting wrapper | Mathematical with temporal assumptions | Post-sort | Yes | Forecasting with calibrated intervals |
| `QuantileForestForecaster` | Native tree model | Empirical | By construction | Yes | Fast, interpretable forecasting |
| `DeepQuantileNet` | Native neural model | Empirical | Post-sort | Yes | Nonlinear patterns with larger datasets |
| `DeepQuantileNetTorch` | Optional neural model | Empirical | Training-time support available | Yes | GPU-backed training and deeper experimentation |
| `TransformerForecaster` | Optional foundation-model wrapper | Empirical or calibrated depending on workflow | Model-dependent | Yes | Pretrained forecasting workflows |

## Shared Output Contract

Every model returns `DistributionPrediction`, so downstream usage stays consistent:

```python
pred = model.predict(df_test)
pred.quantile([0.1, 0.5, 0.9])
pred.interval(0.9)
pred.mean()
pred.sample(100)
```

That shared surface was previously described in a separate guide and is intentionally consolidated here.

## Model Notes

### `ConformalRegressor`

Use this when you already have a scikit-learn compatible regressor and want calibrated intervals on tabular data. The guarantee depends on exchangeability and a sufficiently large calibration set.

### `ConformalForecaster`

Use this when forecasting data is ordered and interval calibration matters more than raw speed. It supports multi-target forecasting through the multivariate layer.

### `QuantileForestForecaster`

Use this when you want a solid non-deep-learning baseline with fast quantile retrieval and sensible behavior on moderate-sized datasets.

### `DeepQuantileNet` and `DeepQuantileNetTorch`

Use these when the signal is nonlinear enough that tree models underfit. The torch-backed path is the better fit when you want a more flexible training loop or accelerator support.

### `TransformerForecaster`

Use this for pretrained time-series workflows when the optional dependency stack is available and the project favors foundation-model-style forecasting.

## Multivariate Dependence

Multi-target models rely on a copula layer rather than treating targets as independent. Depending on the workflow, this may use Gaussian, Clayton, Gumbel, or Frank families, with auto-selection in supported paths.

## Choosing Quickly

- Need a mathematical interval guarantee for tabular regression: `ConformalRegressor`
- Need a mathematical interval guarantee for forecasting: `ConformalForecaster`
- Need fast empirical quantiles without deep learning: `QuantileForestForecaster`
- Need higher-capacity nonlinear modeling: `DeepQuantileNet` or `DeepQuantileNetTorch`
- Need optional pretrained forecasting workflows: `TransformerForecaster`
