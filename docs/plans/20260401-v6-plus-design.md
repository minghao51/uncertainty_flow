# v6+ Design: Bayesian, Causal, and Multi-Modal Uncertainty

**Date:** 2026-04-01
**Status:** Approved

---

## Scope

Three new modules for the v6+ Long-Term Vision phase:

1. **Bayesian Quantile Regression** — NumPyro-based posterior inference
2. **Causal Uncertainty Quantification** — Conformal + doubly-robust CATE estimation
3. **Multi-Modal Uncertainty** — Cross-feature uncertainty aggregation for tabular + TS

Not in scope: Federated learning, image/text modalities.

---

## Module Structure

```
uncertainty_flow/
├── bayesian/
│   ├── __init__.py
│   └── numpyro_model.py       # BayesianQuantileRegressor
├── causal/
│   ├── __init__.py
│   └── estimator.py           # CausalUncertaintyEstimator
└── multimodal/
    ├── __init__.py
    └── aggregator.py          # CrossModalAggregator
```

### New Optional Dependencies

```toml
[project.optional-dependencies]
numpyro = ["numpyro>=0.14.0", "jax>=0.4.0"]
causal = []  # Uses existing conformal infrastructure, no new deps
# multimodal: no new deps, uses existing copula module
```

### API Integration

All modules extend `DistributionPrediction` with new attributes/methods rather than creating subclasses:

- `DistributionPrediction.posterior` — Raw MCMC samples (numpy array)
- `DistributionPrediction.posterior_samples()` — Samples from posterior
- `DistributionPrediction.credible_interval(confidence)` — Bayesian credible interval
- `DistributionPrediction.rhat()` — Gelman-Rubin convergence
- `DistributionPrediction.posterior_summary()` — Summary DataFrame
- `DistributionPrediction.group_uncertainty()` — Per-group contribution
- `DistributionPrediction.group_intervals(confidence)` — Per-group intervals
- `DistributionPrediction.cross_group_correlation()` — Cross-group correlation
- `DistributionPrediction.treatment_effect()` — CATE point estimates
- `DistributionPrediction.average_treatment_effect()` — ATE with CI
- `DistributionPrediction.heterogeneity_score()` — CATE variance

---

## 1. Bayesian Quantile Regression

### API

```python
from uncertainty_flow.bayesian import BayesianQuantileRegressor

model = BayesianQuantileRegressor(
    quantiles=[0.1, 0.5, 0.9],
    n_warmup=500,
    n_samples=1000,
    kernel="nuts",
    prior_width=1.0,
)
model.fit(train_data, target="demand")
pred = model.predict(test_data)

# Standard
pred.quantile(0.5)
pred.interval(0.9)

# Bayesian-specific
pred.posterior_samples()
pred.credible_interval(0.9)
pred.rhat()
pred.posterior_summary()
```

### Internals

- **Likelihood:** Asymmetric Laplace distribution (pinball loss in probabilistic form)
- **Priors:** Weakly informative horseshoe prior on coefficients
- **Sampler:** NUTS with 4 chains by default; SVI fallback for large datasets
- **Posterior storage:** Converted from JAX DeviceArray to numpy for downstream compatibility
- **Convergence:** Warnings surfaced via Python warnings module

### NumPyro Model

```python
def model(X, y=None):
    n_features = X.shape[1]
    # Horseshoe prior for sparsity
    tau = numpyro.sample("tau", dist.HalfCauchy(0.1))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones(n_features)))
    beta = numpyro.sample("beta", dist.Normal(0, tau * lam))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    # Asymmetric Laplace likelihood for quantile regression
    quantile = 0.5  # configured per instance
    mu = X @ beta
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", AsymmetricLaplace(mu, sigma, quantile), obs=y)
```

---

## 2. Causal Uncertainty Quantification

### API

```python
from uncertainty_flow.causal import CausalUncertaintyEstimator

model = CausalUncertaintyEstimator(
    outcome_model=ConformalRegressor(base_model=GradientBoostingRegressor()),
    propensity_model=ConformalRegressor(base_model=GradientBoostingRegressor()),
    treatment_col="intervention",
    method="doubly_robust",
)
model.fit(train_data, target="outcome")
pred = model.predict(test_data)

pred.treatment_effect()
pred.average_treatment_effect()
pred.heterogeneity_score()
pred.interval(0.95)
```

### Doubly-Robust Algorithm

1. Split data into outcome-training and calibration sets
2. Fit outcome model μ̂(X, T) on control + treatment with treatment indicator
3. Fit propensity model ê(X) = P(T=1|X)
4. Compute DR scores for each sample i:
   `DR_i = μ̂₁(X_i) - μ̂₀(X_i) + T_i(Y_i - μ̂₁(X_i))/ê(X_i) - (1-T_i)(Y_i - μ̂₀(X_i))/(1-ê(X_i))`
5. Apply conformal prediction on DR scores → valid CATE intervals
6. Supports `s_learner` and `t_learner` methods as alternatives

### Key Decisions

- Reuses `ConformalRegressor` directly — no new inference engine
- `PropensityClassifier` thin wrapper for logistic-based propensity estimation
- `treatment_col` specified at `fit()` time so same model tests different treatments
- No new dependencies required

---

## 3. Multi-Modal Uncertainty (Tabular + TS)

### API

```python
from uncertainty_flow.multimodal import CrossModalAggregator

model = CrossModalAggregator(
    feature_groups={
        "demographics": ["age", "income", "region"],
        "temporal": ["lag_1", "lag_7", "day_of_week"],
        "weather": ["temperature", "humidity", "pressure"],
    },
    aggregation="product",  # or "copula", "independent"
)
model.fit(train_data, target="demand", base_model=ConformalForecaster(...))
pred = model.predict(test_data)

pred.group_uncertainty()
pred.group_intervals(0.9)
pred.cross_group_correlation()
```

### Aggregation Strategies

1. **product**: Multiply density estimates per group (assumes conditional independence given target)
2. **copula**: Use existing copula module to model cross-group dependence
3. **independent**: Simple average of quantile predictions across groups

### Internal Flow

1. For each feature group, fit a separate model on just those features
2. Compute per-group `DistributionPrediction` objects
3. Aggregate using selected strategy
4. Store per-group attribution in `DistributionPrediction._group_predictions`

### Key Decisions

- Reuses existing copula infrastructure for `copula` aggregation
- No new dependencies
- Feature groups can be auto-detected via correlation clustering (optional)
- Falls back to base model full prediction if no groups specified

---

## Testing Strategy

- **Unit tests** per module: model fitting, prediction, edge cases
- **Integration tests**: Bayesian prediction works with calibration report
- **Convergence tests**: Bayesian model warns on non-convergence
- **Causal validity tests**: DR estimator coverage on semi-synthetic data with known CATE
- **Multi-modal tests**: Group attribution sums to total uncertainty within tolerance

## Documentation Updates

- `docs/guides/models.md`: Add Bayesian, Causal, Multi-Modal sections
- `docs/architecture/overview.md`: Update module diagram
- New guides: `docs/guides/bayesian.md`, `docs/guides/causal.md`
