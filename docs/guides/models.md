# MODELS.md — Model Catalogue & Guarantee Matrix

This document is the **source of truth** for what each model in `uncertainty_flow` guarantees. Before using a model in a production system, read its entry here.

---

## Guarantee Definitions

| Term | Definition |
|---|---|
| **Coverage Guarantee** | A *mathematical* guarantee that the prediction interval will contain the true value with at least the requested probability (given the stated assumptions). |
| **Empirical Coverage** | No mathematical guarantee. Coverage is measured empirically on held-out data and may degrade under distribution shift. |
| **Non-Crossing (training)** | Quantile ordering is enforced *during* model training, so Q95 ≥ Q90 ≥ ... ≥ Q05 always holds structurally. |
| **Non-Crossing (post-sort)** | Quantile ordering is enforced *after* inference by sorting outputs. Corrects the output but does not improve the underlying model. |
| **Temporal Correction** | Extra adjustment applied for time series conformal prediction, accounting for the fact that time series data is not exchangeable. |

---

## Model Matrix

| Model | Type | Coverage Guarantee | Assumptions | Non-Crossing | Multivariate | Best For |
|---|---|---|---|---|---|---|
| `ConformalRegressor` | Tabular wrapper | ✅ Guaranteed | Exchangeability of calibration residuals | ✅ Post-sort | ❌ | Wrapping any sklearn model for tabular data |
| `ConformalForecaster` | Time series wrapper | ✅ Guaranteed (with temporal correction) | Approximate exchangeability (temporal split) | ✅ Post-sort | ✅ (Gaussian copula) | Multi-step forecasting with coverage guarantees |
| `QuantileForestForecaster` | Native model | ⚠️ Empirical only | IID data; sufficient tree depth | ✅ By construction | ✅ (Gaussian copula) | Fast inference, interpretable, no DL required |
| `DeepQuantileNet` | Native model | ⚠️ Empirical only | Sufficient training data; stable distribution | ✅ Post-sort | ✅ (Gaussian copula) | Complex nonlinear patterns, larger datasets |

---

## Detailed Model Entries

---

### `ConformalRegressor`

**What it does:** Wraps any scikit-learn–compatible regressor. Uses split conformal prediction on a held-out calibration set to construct intervals with guaranteed marginal coverage.

**Coverage guarantee:** Yes — marginal coverage at the requested confidence level is guaranteed under the *exchangeability assumption* (i.e., calibration and test samples are drawn from the same distribution). This is a distribution-free guarantee: no assumptions about the shape of the data distribution are required.

**When the guarantee breaks:**
- Distribution shift between train/calibration and test data
- Calibration set is too small (< 50 samples: warning; < 20 samples: error)
- Adversarial or structured test data that violates exchangeability

**Non-crossing:** Post-sort at inference. If crossing is detected in > 5% of predictions, a `UF-W002` warning is emitted.

**Multivariate:** Not supported. Use `ConformalForecaster` for multiple targets.

**Example:**
```python
from uncertainty_flow.wrappers import ConformalRegressor
from sklearn.ensemble import GradientBoostingRegressor

model = ConformalRegressor(base_model=GradientBoostingRegressor())
model.fit(df_train, target="price")
pred = model.predict(df_test)
pred.interval(0.9)
```

---

### `ConformalForecaster`

**What it does:** Temporal-aware conformal wrapper for multi-step time series forecasting. Applies a temporal coverage correction (based on Barber et al. 2023) to account for the non-exchangeability inherent in time-ordered data. Supports multivariate targets via a Gaussian copula.

**Coverage guarantee:** Yes — with the temporal correction, coverage guarantees hold approximately for stationary time series under mild temporal dependence. The correction inflates interval width slightly relative to tabular conformal to compensate for temporal autocorrelation.

**When the guarantee breaks:**
- Strongly non-stationary series (structural breaks, regime changes)
- Very long forecast horizons where temporal dependence becomes severe
- Calibration set is the wrong portion of the series (never use random split for time series — `uncertainty_flow` enforces temporal split automatically)

**Non-crossing:** Post-sort.

**Multivariate:** Yes. When `len(targets) > 1` and `target_correlation='auto'`, a Gaussian copula is fit on the training residuals to model inter-target dependence. Set `target_correlation='independent'` to disable.

**Copula caveat:** The Gaussian copula captures linear correlation between targets. If the true dependence structure is highly nonlinear (e.g., tail dependence), the joint intervals will be approximate. Documented in the calibration report.

**Example:**
```python
from uncertainty_flow.wrappers import ConformalForecaster
from sklearn.ensemble import RandomForestRegressor

model = ConformalForecaster(
    base_model=RandomForestRegressor(),
    targets=["demand", "price"],
    horizon=7,
    target_correlation="auto",
)
model.fit(df_train)
pred = model.predict(df_test)
pred.interval(0.9)  # joint intervals for demand and price
```

---

### `QuantileForestForecaster`

**What it does:** A Quantile Regression Forest that stores the full empirical distribution of training samples at each leaf. Quantile retrieval is fast — no retraining required for different quantile levels.

**Coverage guarantee:** Empirical only. Coverage is measured on held-out data. No mathematical guarantee is provided.

**Strengths:**
- Fast inference (leaf lookup, no iterative procedure)
- Naturally non-crossing (quantiles are derived from the same leaf distribution)
- Interpretable (forest feature importances)
- Works well with moderate-sized tabular and time series datasets

**Non-crossing:** By construction — all quantiles are derived from the same empirical leaf distribution, so Q95 ≥ Q90 ≥ ... ≥ Q05 always holds without post-sort.

**Multivariate:** Supported via Gaussian copula (same as `ConformalForecaster`).

**Limitations:**
- Can underestimate uncertainty near the extremes of the feature space (extrapolation problem common to tree-based models)
- Calibration degrades under distribution shift
- `min_samples_leaf` must be set with care: too small → overfitted leaf distributions; too large → over-smoothed

**Example:**
```python
from uncertainty_flow.models import QuantileForestForecaster

model = QuantileForestForecaster(
    targets="price",
    horizon=14,
    n_estimators=300,
    min_samples_leaf=10,
)
model.fit(df_train)
pred = model.predict(df_test)
pred.quantile([0.05, 0.5, 0.95])
```

---

### `DeepQuantileNet`

**What it does:** A neural network (MLP architecture) with a shared trunk and multiple quantile output heads. Trained simultaneously on all quantile levels using the pinball loss. Designed for larger datasets with complex nonlinear patterns.

**Coverage guarantee:** Empirical only.

**Strengths:**
- Learns complex nonlinear relationships
- Can model heteroscedasticity naturally (uncertainty varies across the feature space)
- Simultaneous multi-quantile training is more efficient than fitting separate models

**Non-crossing:** Post-sort at inference. The network does not enforce ordering during training — if frequent crossing is detected, consider reducing model complexity or adding a monotonicity regulariser (roadmap item).

**Multivariate:** Supported via Gaussian copula.

**Limitations:**
- Requires more data than tree-based models (recommend ≥ 5,000 training samples)
- Longer training time
- More hyperparameters to tune
- No mathematical coverage guarantee

**Backend:** scikit-learn–compatible MLP in v1. PyTorch training loop in roadmap (v2).

**Example:**
```python
from uncertainty_flow.models import DeepQuantileNet

model = DeepQuantileNet(
    targets="price",
    hidden_layers=[128, 64, 32],
    quantile_levels=[0.05, 0.1, 0.5, 0.9, 0.95],
    max_iter=500,
)
model.fit(df_train, target="price")
pred = model.predict(df_test)
pred.interval(0.9)
```

---

## Choosing a Model

```
Is a mathematical coverage guarantee required?
├── Yes → ConformalRegressor (tabular) or ConformalForecaster (time series)
└── No → continue

Is the data time series?
├── Yes → ConformalForecaster or QuantileForestForecaster
└── No (tabular) → ConformalRegressor or QuantileForestForecaster

Is fast inference critical (e.g., real-time)?
├── Yes → QuantileForestForecaster (leaf lookup is very fast)
└── No → any model

Is the dataset large (> 5,000 samples) with complex nonlinear patterns?
├── Yes → DeepQuantileNet
└── No → QuantileForestForecaster or ConformalRegressor

Do you already have a trained sklearn model?
└── Yes → ConformalRegressor or ConformalForecaster (wrap it directly)
```

---

## Roadmap Models (not in v1)

| Model | Description | Target Version |
|---|---|---|
| `BayesianQuantileRegressor` | Full posterior via MCMC | v2 |
| `TransformerForecaster` | Pre-trained/fine-tuned temporal transformer (e.g. Chronos, TimesFM) with uncertainty head | v2 |
| `EnsembleUncertaintyModel` | Disagreement-based uncertainty from model ensembles | v2 |
