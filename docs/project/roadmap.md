# ROADMAP.md — Planned Features & Parking Lot

This document tracks what is intentionally *not* in v1 and why, along with rough prioritisation for future versions. Items here were parked after deliberate design discussion — not forgotten.

---

## v1 Scope (Current)

| Feature | Status |
|---|---|
| `ConformalRegressor` (tabular) | ✅ v1 |
| `ConformalForecaster` (time series, multivariate) | ✅ v1 |
| `QuantileForestForecaster` | ✅ v1 |
| `DeepQuantileNet` (sklearn MLP backend) | ✅ v1 |
| Polars I/O with LazyFrame support | ✅ v1 |
| `DistributionPrediction` object (`.quantile()`, `.interval()`, `.mean()`, `.plot()`) | ✅ v1 |
| Calibration report (Polars DataFrame + visual) | ✅ v1 |
| Residual correlation analysis (`uncertainty_drivers_`) | ✅ v1 |
| Gaussian copula for multivariate joint intervals | ✅ v1 |
| Holdout + cross-conformal calibration split | ✅ v1 |
| Pinball loss, Winkler score, coverage metrics (standalone) | ✅ v1 |
| Warning system (UF-W/E codes) | ✅ v1 |

---

## v2 Priorities (All Implemented - March 2026)

### 1. PyTorch Backend for `DeepQuantileNet` (DONE - March 2026)
**What:** Replace sklearn MLP backend with full PyTorch training loop.

**Implementation:**
- `DeepQuantileNetTorch` class with GPU support
- `QuantileNetTorch` module with shared trunk + quantile heads
- Monotonicity loss for non-crossing at training time
- Ensemble support (`n_estimators`)

**Note:** PyTorch is now an optional dependency (`pip install uncertainty-flow[torch]`).

---

### 2. `DistributionPrediction.sample()` (DONE - March 2026)
**What:** Draw n samples from the predicted distribution for each input row, enabling downstream Monte Carlo simulations.  
**Implementation:** Fit a spline-interpolated CDF per row from the quantile matrix; sample via inverse CDF using `scipy.interpolate.interp1d`. Uniform samples are clipped to quantile level bounds to prevent extrapolation.  
**Returns:** Polars DataFrame with `sample_id` column (original row index, repeated n times) plus one column per target.

---

### 3. Non-Crossing at Training Time (DONE - March 2026)
**What:** Enforce quantile monotonicity *during* training via a monotonicity loss term, rather than relying on post-sort at inference.  
**Implementation:** `MonotonicityLoss` class + `monotonicity_weight` hyperparameter in `DeepQuantileNetTorch`. Penalises violations of Q_i ≤ Q_{i+1} during backprop.  
**Usage:** Set `monotonicity_weight > 0` when instantiating the model.

---

### 4. Pre-trained Transformer Forecasters (DONE - March 2026)
**What:** Integration with Chronos-2 (Amazon, Oct 2025) — state-of-the-art universal time series foundation model on HuggingFace.  
**Implementation:** `TransformerForecaster` class in `models/`. Wraps Chronos-2 from HuggingFace via `chronos-forecasting>=2.0`. Conformal calibration for coverage guarantees. Fine-tune uncertainty head only (freeze trunk).  
**Note:** `chronos-forecasting` is an optional dependency (`pip install uncertainty-flow[transformers]`).

---

### 5. Quantile SHAP — Uncertainty Feature Attribution (DONE - March 2026)
**What:** Run SHAP on the upper and lower quantile models separately; compute the difference in feature importances to identify what *drives interval width* (not just point predictions).  
**Implementation:** `uncertainty_shap()` function in `calibration/shap_values.py`. Uses `shap.KernelExplainer` for model-agnostic attribution. Returns Polars DataFrame with SHAP values per feature per quantile level, plus interval width contribution.  
**Usage:** `model.uncertainty_shap(X, background=X_train[:100])`.  
**Note:** `shap` is an optional dependency (`pip install uncertainty-flow[shap]`).

---

### 6. Richer Copula Families (DONE - March 2026)
**What:** Beyond the Gaussian copula (v1), support Clayton, Frank, and Gumbel copulas for modelling tail dependence between targets.  
**Implementation:** New copula classes in `multivariate/copula.py`: `ClaytonCopula`, `GumbelCopula`, `FrankCopula`, plus `auto_select_copula()` for BIC-based selection. Updated `ConformalForecaster` and `QuantileForestForecaster` to accept `copula_family` parameter (default: `"auto"`).  
**Tail dependence:** Clayton (lower), Gumbel (upper), Frank (symmetric).  
**Note:** No new dependencies — uses `scipy.stats` for all copula families.

---

## Long-Term Ideas (No Timeline)

| Idea | Notes |
|---|---|
| `BayesianQuantileRegressor` | Full posterior via MCMC (NumPyro / PyMC). Powerful but slow — niche use case. |
| Ensemble uncertainty | Disagreement across ensemble members as uncertainty signal. |
| Conformal Risk Control | Extends conformal prediction to complex loss functions (e.g., LLM-as-judge scoring). MAPIE already does this — potential thin wrapper. |
| Interactive calibration dashboard | Web-based (Streamlit / Panel) version of the calibration report. |
| `uncertainty_flow` for classification | Prediction sets (classification conformal prediction). Out of scope currently but a natural extension. |
| Async / Streaming Inference | An async `.predict()` interface for real-time applications (e.g., FastAPI endpoints). No strong v1 requirement. |
| Model-Agnostic AutoCalibration | Post-hoc calibration step (isotonic regression on quantiles). Redundant given conformal prediction's coverage guarantees. |

---

## Deprecation Policy

- Features will not be deprecated without a full version cycle of deprecation warnings.
- Breaking API changes require a major version bump.
- The `DistributionPrediction` object interface is considered stable from v1 onwards — new methods may be added, existing signatures will not change without deprecation notice.
