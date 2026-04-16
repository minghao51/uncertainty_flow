# uncertainty_flow

> Probabilistic forecasting and uncertainty quantification — as easy as `fit` / `predict`.

[![PyPI version](https://badge.fury.io/py/uncertainty-flow.svg)](https://badge.fury.io/py/uncertainty-flow)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why uncertainty_flow?

Most forecasting libraries are optimised for point predictions. Real-world decisions require understanding *uncertainty* — but existing tools either lack robust support or scatter it across incompatible APIs.

`uncertainty_flow` is built **distribution-first**: every model returns a `DistributionPrediction` object, not just a number. Uncertainty is not an afterthought.

---

## Key Features

| Feature | Description |
|---|---|
| **Distribution-first API** | `model.predict()` returns a `DistributionPrediction` object with `.quantile()`, `.interval()`, `.mean()`, `.sample()`, and `.plot()` |
| **Polars-native I/O** | Pass Polars DataFrames or LazyFrames directly — including lazy evaluation support |
| **Conformal wrappers** | Wrap any scikit-learn model with statistically rigorous coverage guarantees |
| **Multivariate support** | Marginal CDFs per target, with copula-backed joint sampling for multivariate forecasts |
| **Model persistence** | Save and load fitted models with `.save()` / `.load()` using `.uf` archives |
| **Calibration reports** | `.calibration_report()` returns a Polars DataFrame — paste-ready for model cards |
| **Uncertainty driver detection** | Automatic residual correlation analysis surfaces which features drive interval width |
| **Time series ready** | Univariate and multivariate forecasting from day one |
| **Honest guarantees** | Every model is clearly documented: coverage-guaranteed vs. best-effort |

---

## Installation

```bash
pip install uncertainty-flow
# or, recommended:
uv add uncertainty-flow
```

---

## Quickstart

### Tabular — Wrap any scikit-learn model

```python
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor
from uncertainty_flow.wrappers import ConformalRegressor

# Load data as Polars DataFrame
df = pl.read_csv("data.csv")
X_train, X_test = df[:800], df[800:]

# Wrap any sklearn estimator
base = GradientBoostingRegressor()
model = ConformalRegressor(base_model=base)
model.fit(X_train, target="price")

# Get a full distribution prediction
pred = model.predict(X_test)

# Extract what you need
pred.interval(confidence=0.9)       # (lower: Series, upper: Series)
pred.quantile([0.05, 0.5, 0.95])    # Polars DataFrame of quantiles
pred.mean()                         # median (point estimate)
pred.plot()                         # fan chart + calibration overlay
```

### Time Series — Multivariate forecasting

```python
from uncertainty_flow.models import QuantileForestForecaster

model = QuantileForestForecaster(
    targets=["price", "volume"],
    horizon=14,
    copula_family="auto",        # learns a supported copula from data
)
model.fit(ts_train)

pred = model.predict(ts_test)
pred.interval(confidence=0.9)    # marginal intervals across both targets
pred.sample(100, random_state=42)  # copula-aware joint samples
```

### Persistence

```python
model = QuantileForestForecaster(targets="price", horizon=7, auto_tune=False)
model.fit(ts_train)
model.save("models/example.uf")

loaded = QuantileForestForecaster.load("models/example.uf")
pred = loaded.predict(ts_test)
```

### Calibration Report

```python
report = model.calibration_report(X_test, y_test)
# Returns a Polars DataFrame:
# ┌────────────┬──────────────────┬───────────────────┬──────────┬───────────────┐
# │ quantile   │ requested_coverage│ achieved_coverage │ sharpness│ winkler_score │
# ╞════════════╪══════════════════╪═══════════════════╪══════════╪═══════════════╡
# │ 0.80       │ 0.80             │ 0.83              │ 12.4     │ 18.2          │
# │ 0.90       │ 0.90             │ 0.88              │ 17.1     │ 22.7          │
# │ 0.95       │ 0.95             │ 0.91              │ 21.3     │ 28.4          │
# └────────────┴──────────────────┴───────────────────┴──────────┴───────────────┘
```

---

## Coverage Guarantees

Not all models are equal. See [Models Guide](./docs/guides/models.md) for the full breakdown.

| Model | Coverage Guarantee | Non-Crossing |
|---|---|---|
| `ConformalRegressor` | ✅ Guaranteed (exchangeability assumption) | ✅ Post-sort |
| `QuantileForestForecaster` | ⚠️ Empirical only | ✅ Post-sort |
| `DeepQuantileNet` | ⚠️ Empirical only | ✅ Post-sort |
| `ConformalForecaster` | ✅ Guaranteed (with temporal correction) | ✅ Post-sort |

---

## Roadmap

See [Roadmap](./docs/project/roadmap.md) for remaining planned features such as faster SHAP explainers and broader multivariate scalability.

---

## Documentation

| Doc | What it covers |
|---|---|
| [Architecture Overview](./docs/architecture/overview.md) | Package structure, data flow, module families |
| [Models Guide](./docs/guides/models.md) | Model selection, guarantee tradeoffs, choosing quickly |
| [Calibration Guide](./docs/guides/calibration.md) | Interpreting calibration reports, miscalibration response |
| [Charting Guide](./docs/guides/charting.md) | Plotting, intervals, samples, visual inspection |
| [Benchmarking Guide](./docs/guides/benchmarking.md) | CLI benchmarks, datasets, auto-tuning |
| [API Reference](./docs/api/spec.md) | Full API specification for all public classes |
| [Contributing](./docs/project/contributing.md) | Dev setup, testing conventions, adding new models |
| [Changelog](./docs/project/changelog.md) | Release history |

---

## Contributing

See [Contributing Guide](./docs/project/contributing.md) for dev setup, testing conventions, and how to add a new model.

---

## License

MIT
