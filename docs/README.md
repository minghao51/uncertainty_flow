# Uncertainty Flow

> Probabilistic forecasting and uncertainty quantification — as easy as `fit` / `predict`.

---

## Quick Links

- **[Guides](guides/distribution-approach.md)** — Learn the distribution-first workflow
- **[API Reference](api/core.md)** — Auto-generated docs for every module
- **[Architecture](architecture/overview.md)** — Package structure and data flow
- **[Benchmarks](benchmarks/README.md)** — Benchmark results and reproduction

## Overview

`uncertainty_flow` is built **distribution-first**: every model returns a `DistributionPrediction` object, not just a number. Uncertainty is not an afterthought.

### Key Features

| Feature | Description |
|---|---|
| **Distribution-first API** | `model.predict()` returns quantiles, intervals, mean, samples, and plots |
| **Polars-native I/O** | Pass Polars DataFrames or LazyFrames directly |
| **Conformal wrappers** | Wrap any scikit-learn model with statistically rigorous coverage guarantees |
| **Multivariate support** | Marginal CDFs with copula-backed joint sampling |
| **Model persistence** | Save/load fitted models with `.save()` / `.load()` |
| **Calibration reports** | Polars DataFrame reports — paste-ready for model cards |
| **Risk functions** | Financial VaR, inventory cost, asymmetric loss, and conformal risk control |

### Install

```bash
pip install uncertainty-flow
```

### Quick Start

```python
from uncertainty_flow import (
    QuantileForestForecaster,
    coverage_score,
    pinball_loss,
)

model = QuantileForestForecaster(quantiles=[0.1, 0.5, 0.9])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred.interval(alpha=0.8))
print("Coverage:", coverage_score(y_test, pred))
print("Pinball loss:", pinball_loss(y_test, pred))
```
