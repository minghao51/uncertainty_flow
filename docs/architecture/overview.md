# ARCHITECTURE.md — Package Structure & Data Flow

---

## Package Structure

```
uncertainty_flow/
│
├── core/
│   ├── base.py                  # BaseUncertaintyModel ABC
│   ├── distribution.py          # DistributionPrediction class
│   └── types.py                 # Shared type aliases and enums
│
├── models/
│   ├── __init__.py
│   ├── quantile_forest.py       # QuantileForestForecaster
│   └── deep_quantile.py         # DeepQuantileNet (sklearn-compatible MLP)
│
├── wrappers/
│   ├── __init__.py
│   ├── conformal.py             # ConformalRegressor (tabular)
│   └── conformal_ts.py          # ConformalForecaster (time series, temporal split)
│
├── multivariate/
│   ├── __init__.py
│   ├── marginal.py              # Per-target marginal CDF fitting
│   └── copula.py                # Gaussian copula for joint intervals
│
├── calibration/
│   ├── __init__.py
│   ├── report.py                # calibration_report() → Polars DataFrame
│   ├── plot.py                  # Fan charts, coverage plots
│   └── residual_analysis.py     # uncertainty_drivers_ computation
│
├── metrics/
│   ├── __init__.py
│   ├── pinball.py               # Pinball / quantile loss
│   ├── winkler.py               # Winkler interval score
│   └── coverage.py              # Empirical coverage checks
│
├── utils/
│   ├── __init__.py
│   ├── polars_bridge.py         # Polars → NumPy conversion, LazyFrame handling
│   ├── split.py                 # Calibration set splitting strategies
│   └── warnings.py              # Standardised warning messages
│
└── __init__.py                  # Public API surface
```

---

## Data Flow

```
User Input (Polars DataFrame / LazyFrame)
        │
        ▼
┌─────────────────────┐
│   polars_bridge.py  │  ← LazyFrame.collect() if needed
│   Polars → NumPy    │    Column validation, dtype checks
└─────────┬───────────┘
          │  np.ndarray (X), np.ndarray (y)
          ▼
┌─────────────────────┐
│  BaseUncertaintyModel│  ← .fit(), .predict()
│  (ABC)              │    Dispatches to model/wrapper
└─────────┬───────────┘
          │
    ┌─────┴──────────────────────────────┐
    │                                    │
    ▼                                    ▼
ConformalRegressor               QuantileForestForecaster
(sklearn wrapper)                (native quantile model)
    │                                    │
    └──────────────┬─────────────────────┘
                   │  raw quantile arrays (np.ndarray)
                   ▼
        ┌─────────────────────┐
        │ DistributionPrediction│  ← wraps raw arrays
        │  .quantile()          │    converts outputs back to Polars
        │  .interval()          │
        │  .mean()              │
        │  .plot()              │
        └─────────────────────┘
                   │
                   ▼
        User Output (Polars Series / DataFrame)
```

---

## Core Classes

### `BaseUncertaintyModel` (ABC)

All models inherit from this. Enforces the contract:

```python
class BaseUncertaintyModel(ABC):

    @abstractmethod
    def fit(self, data, target: str | list[str], **kwargs) -> "BaseUncertaintyModel":
        """Accepts Polars DataFrame or LazyFrame."""
        ...

    @abstractmethod
    def predict(self, data) -> "DistributionPrediction":
        """Returns a DistributionPrediction object."""
        ...

    def calibration_report(self, data, target: str | list[str]) -> pl.DataFrame:
        """Runs calibration diagnostics. Returns Polars DataFrame."""
        ...

    @property
    def uncertainty_drivers_(self) -> pl.DataFrame | None:
        """Set post-fit. Returns feature-residual correlation table or None."""
        ...
```

---

### `DistributionPrediction`

The core output object. Stores predicted quantile arrays internally; exposes a clean interface.

```python
class DistributionPrediction:
    """
    Holds predicted distributions for N samples.
    All outputs are returned as Polars Series or DataFrames.
    """

    def __init__(
        self,
        quantile_matrix: np.ndarray,   # shape (N, Q) — N samples, Q quantile levels
        quantile_levels: list[float],  # e.g. [0.05, 0.1, ..., 0.95]
        target_names: list[str],       # for multivariate: one DP per target, or stacked
        index: pl.Series | None = None # original row index from input
    ): ...

    def quantile(self, q: float | list[float]) -> pl.DataFrame:
        """Extract one or more quantile levels. Returns Polars DataFrame."""
        ...

    def interval(self, confidence: float = 0.9) -> pl.DataFrame:
        """
        Returns DataFrame with columns: lower, upper.
        Derives symmetric interval: (1-confidence)/2 and (1+confidence)/2 quantiles.
        """
        ...

    def mean(self) -> pl.Series:
        """Returns the 0.5 quantile (median) as a Polars Series."""
        ...

    def sample(self, n: int, random_state: int | None = None) -> pl.DataFrame:
        """
        Draw n samples per input row via spline-interpolated inverse CDF.
        Returns DataFrame with (n * n_samples) rows and columns: sample_id, plus one column per target.
        """
        ...

    def plot(self, actuals: pl.Series | None = None) -> None:
        """
        Fan chart of quantile bands.
        If actuals provided, overlays calibration coverage.
        """
        ...
```

---

### Polars Bridge

The single seam where Polars meets NumPy. All conversions happen here and nowhere else.

```python
# utils/polars_bridge.py

def to_numpy(data: pl.DataFrame | pl.LazyFrame, columns: list[str]) -> np.ndarray:
    """
    Accepts DataFrame or LazyFrame.
    Materialises LazyFrame only when necessary (.collect()).
    Returns np.ndarray with float64 dtype.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.select(columns).collect()
    return data.select(columns).to_numpy(allow_copy=False)


def to_polars(array: np.ndarray, columns: list[str], index: pl.Series | None = None) -> pl.DataFrame:
    """Converts NumPy array back to Polars DataFrame, restoring index if provided."""
    ...
```

---

## Calibration Split Strategies

Managed by `utils/split.py`:

| Strategy | Class | Default |
|---|---|---|
| Holdout (last n%) | `TemporalHoldoutSplit` | ✅ Time series default |
| Holdout (random n%) | `RandomHoldoutSplit` | ✅ Tabular default |
| Cross-conformal | `CrossConformalSplit` | Optional (`calibration_method='cross'`) |

**Guards (enforced in all strategies):**
- `n_calibration < 20` → `raise ValueError`
- `n_calibration < 50` → `warnings.warn` (undercoverage risk)

---

## Multivariate Flow (Gaussian Copula)

When `targets` is a list and `target_correlation != 'independent'`:

```
Per-target marginal CDFs fitted independently
        │
        ▼
Residuals extracted per target
        │
        ▼
Gaussian copula fitted on residual correlation matrix
        │
        ▼
Joint intervals derived from copula samples
        │
        ▼
DistributionPrediction (multivariate) returned
```

---

## Dependency Map

```
uncertainty_flow
    ├── polars            ← I/O layer
    ├── numpy             ← internal compute spine
    ├── scikit-learn      ← base model interface + conformal wrappers
    ├── scipy             ← Gaussian copula, statistical tests
    ├── matplotlib        ← plot() visualisations (optional, soft dep)
    └── mapie             ← conformal prediction implementation (optional, soft dep)
```

`matplotlib` and `mapie` are optional soft dependencies — `uncertainty_flow` imports them lazily and raises a clear `ImportError` with install instructions if they are missing when needed.
