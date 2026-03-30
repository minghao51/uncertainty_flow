# DESIGN.md — Design Decisions & Rationale

This document records *why* key decisions were made for `uncertainty_flow`. It is a living log — new decisions should be appended with a date and context.

---

## Core Philosophy

1. **Distribution-first, not point-first.** The default return type of every model is a `DistributionPrediction` object. Quantiles and intervals are derived from the distribution, not the primary output.
2. **Honest about uncertainty.** Every model clearly documents whether its coverage guarantee is *mathematical* (conformal) or *empirical* (best-effort). We never overstate guarantees.
3. **Polars I/O, NumPy spine.** User-facing input and output is Polars (including LazyFrame support). Internal computation uses NumPy arrays for compatibility with scikit-learn and other numerical backends.
4. **Usability over completeness.** A clean `fit/predict` API that data scientists can use in 10 minutes is worth more than a theoretically complete but complex interface.

---

## Decision Log

### [D-001] Package name: `uncertainty_flow`
**Decision:** Name the package `uncertainty_flow` (PyPI: `uncertainty-flow`).
**Rationale:** The original working name `quantileflow` leaned too heavily on quantile regression, while the core identity is now distribution-first. `uncertainty_flow` better signals the full probabilistic scope.
**Trade-off:** Less immediately searchable for users looking specifically for "quantile regression" — mitigated by strong documentation keywords.

---

### [D-002] Input/Output layer: Polars; internal compute: NumPy
**Decision:** Accept Polars DataFrames and LazyFrames at the API boundary. Convert to NumPy internally before model fitting/inference.
**Rationale:** scikit-learn, scipy, and most ML libraries require array-like inputs. Polars' ML ecosystem support is still maturing. Forcing Polars internally would break compatibility with the entire sklearn ecosystem.
**Trade-off:** Conversion overhead at the boundary. For very large datasets, `.collect()` on a LazyFrame before conversion is a materialisation cost. Documented clearly in performance notes.
**Lazy evaluation:** LazyFrame queries are passed through and only materialised (`.collect()`) immediately before the NumPy conversion step, preserving lazy evaluation benefits for upstream data pipelines.

---

### [D-003] Non-crossing guarantee: documentation + post-sort, not training constraint
**Decision:** Do not enforce non-crossing at training time. Instead, document per-model whether crossing can occur and apply a lightweight post-sort at inference where applicable.
**Rationale:** Training-time non-crossing constraints (e.g., incremental softplus offsets) are architecture-specific and cannot be applied to arbitrary wrapped base models (e.g., a user's existing XGBoost). Post-sort is model-agnostic.
**Known limitation:** Post-sort corrects the output but does not improve the model's calibration. If crossing is frequent, it signals a model quality issue — surfaced as a warning.
**Documentation contract:** See `MODELS.md` — each model has an explicit "Non-Crossing" and "Coverage Guarantee" entry.

---

### [D-004] Calibration set strategy: holdout default
**Decision:** Default calibration split is a holdout (last 20% for time series, random 20% for tabular). Cross-conformal is available as `calibration_method='cross'`.
**Rationale:** Holdout is fast, predictable, and easy to reason about. Speed is the default priority.
**Guard:** Hard warning if the resulting calibration set has fewer than 50 samples. Error (not warning) if fewer than 20.
**Time series note:** For temporal data, the holdout is always the *last* n% of observations — never a random split, which would leak future information.

---

### [D-005] Multivariate uncertainty: marginal CDFs + Gaussian copula
**Decision:** v1 supports marginal CDFs per target. Joint intervals are constructed via a Gaussian copula fit on the residual correlation structure.
**Rationale:** Full nonparametric joint CDFs (e.g., vine copulas) are powerful but very complex and computationally expensive. A Gaussian copula captures linear inter-target correlation — which covers the majority of real-world cases — with minimal complexity overhead.
**v1 scope:** `target_correlation='auto'` fits the Gaussian copula automatically. `target_correlation='independent'` uses marginal CDFs only.
**Roadmap:** Richer copula families (Clayton, Frank) are parked in v2.

---

### [D-006] Heteroscedasticity detection: residual correlation analysis
**Decision:** After fitting, automatically run a residual correlation analysis to detect which features correlate with squared residuals. Results are stored in `model.uncertainty_drivers_` and surfaced in the calibration report.
**Rationale:** Quantile SHAP is more precise but computationally heavy. Residual correlation is fast, interpretable, and sufficient for the majority of use cases.
**Bidirectional design:** Users can provide `uncertainty_features` as a hint. The model also independently detects drivers. Both signals are shown in the calibration report — the user's hints are validated, not just accepted.
**Unknown unknowns:** If residual correlation analysis finds *no* significant drivers, a warning is emitted: "Interval width drivers could not be identified. Intervals may be uniformly conservative." This surfaces the unknown rather than hiding it.

---

### [D-007] CDP object v1 surface: `.quantile()`, `.interval()`, `.mean()`, `.plot()`
**Decision:** `DistributionPrediction` exposes `quantile()`, `interval()`, `mean()`, and `plot()` in v1. `.sample()` is parked for v2.
**Rationale:** `.sample()` requires a full generative model or kernel density estimate over the predicted distribution, which adds significant complexity. The v1 surface covers the vast majority of practical use cases.

---

### [D-008] PyTorch backend: roadmap only
**Decision:** v1 targets scikit-learn compatible models only. PyTorch training loops are parked for v2.
**Rationale:** scikit-learn covers the large majority of the target user's existing workflow. Pre-trained/fine-tuned transformer-based forecasters (e.g., TimesFM, Chronos) are compelling but require a substantially different integration path. Better to do it right than fast.

---

## Non-Goals (v1)

These are explicitly out of scope for v1 to keep the implementation focused:

- Async / streaming inference
- LLM-as-a-judge uncertainty (MAPIE Risk Control style)
- Vine copula / nonparametric joint distributions
- Quantile SHAP
- `.sample()` on `DistributionPrediction`
- PyTorch training loops
- Bayesian / MCMC methods
- AutoML-style model selection

---

## Open Questions

| ID | Question | Status |
|---|---|---|
| OQ-001 | Should `DistributionPrediction.plot()` use matplotlib or a Polars-native plot backend? | Open |
| OQ-002 | For cross-conformal mode, what is the default k (number of folds)? | Tentative: 5 |
| OQ-003 | Should `.calibration_report()` render an HTML visual by default, or only when explicitly called? | Open |
