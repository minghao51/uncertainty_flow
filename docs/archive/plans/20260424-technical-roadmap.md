# Technical Roadmap: Decision-Centric UQ

This document outlines the technical implementation strategy for the `uncertainty_flow` decision-centric API, designed to bridge the gap between probabilistic forecasts and business actions.

## 1. Core Philosophy: "Distribution to Action"
The goal is to provide a seamless transition from a `DistributionPrediction` object to an optimal business decision without requiring the user to perform manual integration or optimization.

### Key API Pattern:
```python
prediction = model.predict(X)
decision = prediction.decide(Strategy(parameters))
```

---

## 2. Mathematical Methods & Archetypes

### A. Asymmetric Loss Optimization (The Newsvendor Problem)
Used for problems where the cost of over-prediction (excess) differs from the cost of under-prediction (shortage).

**Method:**
For a given overstock cost $c_o$ and stockout cost $c_u$, the optimal decision $y^*$ is the quantile at level $	au$:
$$	au = rac{c_u}{c_u + c_o}$$

**Implementation:**
- **Technique:** Leverage the existing `DistributionPrediction.quantile()` method.
- **Complexity:** $O(1)$ lookup/interpolation on the predicted quantile matrix.

### B. Threshold-Based Decisions (Safety Triggers)
Used for binary actions based on the probability of exceeding a critical threshold (e.g., fraud, failure).

**Method:**
Calculate $P(Y > Threshold)$. If this probability exceeds a given confidence level, trigger the action.

**Implementation:**
- **Technique:** Search the quantile levels to find the level $	au$ where $Q(	au) pprox Threshold$.
- **Result:** $1 - 	au$ gives the exceedance probability.

### C. Target Optimizer (Goal Seeking)
Used to find the value required to hit a specific target with a requested confidence level.

**Method:**
Identify the value $V$ such that $P(Y \ge Target) \ge Confidence$.

---

## 3. Implementation Architecture

### `uncertainty_flow.decisions` Module
- `DecisionStrategy`: Abstract base class defining the `resolve(prediction)` interface.
- `DecisionResult`: Dataclass containing `optimal_value`, `strategy_metadata`, and `expected_risk`.

### Extensions to `DistributionPrediction`
- `.decide(strategy: DecisionStrategy | Callable)`: The entry point for decision logic.
- **Monte Carlo Fallback:** For custom cost functions, `decide` will utilize `.sample()` to perform Monte Carlo integration:
  $$E[Loss(d)] = rac{1}{N} \sum_{i=1}^N Loss(d, y_i)$$
  It will then minimize this expected loss over the sample space.

---

## 4. Scalability & Performance
- **Zero-Copy Optimization:** Decisions based on quantiles (like `AsymmetricLoss`) will use direct NumPy views of the `_quantiles` matrix.
- **Lazy Evaluation:** Integration with Polars allows decision results to be returned as LazyFrames for large-scale batch processing.
