# CALIBRATION.md — Calibration Deep Dive

Calibration is a first-class citizen in `uncertainty_flow`. This document explains what calibration means, how to interpret the calibration report, and what to do when your model is miscalibrated.

---

## What Is Calibration?

A model is **well-calibrated** if its stated confidence matches its empirical accuracy. For uncertainty quantification:

> "If I ask for 90% prediction intervals, exactly 90% of true values should fall within those intervals."

A model that achieves 70% coverage when asked for 90% is *underconfident* (intervals are too narrow). A model that achieves 98% coverage when asked for 90% is *overconfident in interval width* (intervals are too wide and therefore uninformative).

**Both are problems.** Narrow intervals give false confidence. Wide intervals are useless for decision-making.

---

## The Calibration Report

Every model exposes `.calibration_report(data, target)`. It returns a Polars DataFrame:

```
┌────────────┬──────────────────┬───────────────────┬──────────┬───────────────┐
│ quantile   │ requested_coverage│ achieved_coverage │ sharpness│ winkler_score │
│ f64        │ f64               │ f64               │ f64      │ f64           │
╞════════════╪══════════════════╪═══════════════════╪══════════╪═══════════════╡
│ 0.80       │ 0.80             │ 0.83              │ 12.4     │ 18.2          │
│ 0.90       │ 0.90             │ 0.88              │ 17.1     │ 22.7          │
│ 0.95       │ 0.95             │ 0.91              │ 21.3     │ 28.4          │
└────────────┴──────────────────┴───────────────────┴──────────┴───────────────┘
```

### Column Definitions

| Column | Definition |
|---|---|
| `quantile` | The confidence level requested (e.g., 0.90 = 90% prediction interval) |
| `requested_coverage` | Same as `quantile` — the target |
| `achieved_coverage` | Fraction of test observations that actually fell within the interval |
| `sharpness` | Mean interval width (lower = more informative) |
| `winkler_score` | Winkler interval score — penalises both width *and* coverage violations (lower = better) |

### Interpreting the Report

| `achieved_coverage` vs `requested_coverage` | Diagnosis |
|---|---|
| `achieved` ≈ `requested` (within 2–3%) | ✅ Well-calibrated |
| `achieved` < `requested` by > 5% | ⚠️ Undercoverage — intervals are too narrow. Risk of overconfident decisions. |
| `achieved` > `requested` by > 5% | ⚠️ Overcoverage — intervals are too wide. Model is conservative but uninformative. |

`uncertainty_flow` emits `UF-W003` if the gap exceeds 5% at any quantile level.

---

## Metrics

### Pinball Loss (Quantile Loss)

The standard training objective for quantile regression. Also used for evaluation.

```
L_q(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)
```

Interpretation: asymmetric penalty. For q = 0.9, underpredicting is penalised 9x more than overpredicting.

```python
from uncertainty_flow.metrics import pinball_loss

pinball_loss(y_true, y_pred_q90, quantile=0.9)
```

### Winkler Score

Evaluates interval quality in a single number. Penalises both width and coverage violations.

```
W(l, u, y, α) = (u - l) + (2/α) * max(l - y, 0) + (2/α) * max(y - u, 0)
```

Where α = 1 - confidence (e.g., 0.1 for a 90% interval), l = lower bound, u = upper bound.

- Lower Winkler score = better
- A wide interval with perfect coverage will score higher than a sharp interval with good coverage
- Penalises coverage failures heavily (the 2/α multiplier)

```python
from uncertainty_flow.metrics import winkler_score

winkler_score(y_true, lower, upper, confidence=0.9)
```

### Empirical Coverage

```python
from uncertainty_flow.metrics import coverage_score

coverage_score(y_true, lower, upper)
# Returns fraction of y_true within [lower, upper]
```

---

## Calibration Strategies

### Holdout (Default)

A held-out portion of the data (last n% for time series, random n% for tabular) is reserved as the calibration set. The model never sees this data during training.

**Advantages:** Simple, fast, easy to reason about.  
**Disadvantages:** Wastes some training data; calibration estimate has higher variance on small datasets.  
**Time series note:** The holdout is always the *last* n% of observations. Random splits are never used for temporal data.

### Cross-Conformal

Uses k-fold cross-validation to produce calibration residuals, making more efficient use of data.

```python
model = ConformalRegressor(
    base_model=RandomForestRegressor(),
    calibration_method="cross",
)
```

**Advantages:** More data-efficient; lower variance calibration estimate.  
**Disadvantages:** Slower to fit (k training runs); more complex to reason about.  
**When to use:** Small datasets where holdout wastes too much data.

---

## Uncertainty Driver Detection

After fitting, `uncertainty_flow` automatically analyses which features correlate with prediction error magnitude (residual correlation analysis). This detects **heteroscedasticity** — where uncertainty is not uniform across the feature space.

### How it works

1. Fit the base model on training data
2. Compute squared residuals on the calibration set: `e_i² = (y_i - ŷ_i)²`
3. Compute Pearson correlation between each feature and `e_i²`
4. Test for significance (Bonferroni-corrected p-value)
5. Store results in `model.uncertainty_drivers_`

### Reading `uncertainty_drivers_`

```
┌─────────────────┬─────────────────────┬─────────┐
│ feature         │ residual_correlation │ p_value │
│ str             │ f64                  │ f64     │
╞═════════════════╪═════════════════════╪═════════╡
│ volatility      │ 0.71                 │ 0.001   │  ← strong driver
│ days_since_event│ 0.43                 │ 0.012   │  ← moderate driver
│ region          │ 0.08                 │ 0.34    │  ← not significant
└─────────────────┴─────────────────────┴─────────┘
```

### Bidirectional hints

You can provide your own hints, and the model will validate them against the residual analysis:

```python
model = ConformalRegressor(
    base_model=GradientBoostingRegressor(),
    uncertainty_features=["volatility", "age"],
)
model.fit(df_train, target="price")

# The calibration report will show:
# - Which of your hints were confirmed by residual analysis
# - Any additional drivers the model found that you didn't flag
# - Any hints that were NOT confirmed (potential red flag)
```

### Unknown unknowns

If the residual analysis finds *no* significant drivers, `UF-W004` is emitted:

> ⚠️ UF-W004: Residual correlation analysis found no significant uncertainty drivers. Intervals may be uniformly conservative or the model may be well-specified. Consider checking for distribution shift.

This is not necessarily a problem — it can mean the model is well-specified. But it means you cannot rely on heteroscedastic interval adaptation.

---

## What to Do When Miscalibrated

### Scenario: `achieved_coverage` < `requested_coverage` (undercoverage)

1. **Check calibration set size.** If < 50 samples, increase it (`calibration_size=0.3`).
2. **Check for distribution shift.** Is the test data from the same distribution as training/calibration?
3. **Switch to cross-conformal.** `calibration_method='cross'` may give a better calibration estimate.
4. **Use a conformal wrapper.** If you're using `QuantileForestForecaster` or `DeepQuantileNet` (no coverage guarantee), wrapping with conformal prediction will force coverage.

### Scenario: `achieved_coverage` > `requested_coverage` (overcoverage / wide intervals)

1. **Check sharpness.** Wide intervals may indicate the base model has high variance or the calibration set is dominated by easy cases.
2. **Review uncertainty drivers.** Are there features that explain most of the interval width? You may be able to build a more targeted model.
3. **Increase base model complexity.** Underfitting → high residuals → wide calibration residuals → wide intervals.

### Scenario: Coverage is good but Winkler score is high

Intervals are covering correctly but are too wide (low sharpness). 

1. **Improve the base model** (reduce residuals → narrower intervals).
2. **Feature engineering.** Better features → better point predictions → narrower uncertainty.
3. **Add `uncertainty_features`** to guide heteroscedastic adaptation — narrow intervals where the model is confident, widen only where uncertainty is genuinely high.
