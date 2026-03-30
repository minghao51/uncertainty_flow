# CONTRIBUTING.md

Thank you for contributing to `uncertainty_flow`. This guide covers dev setup, conventions, and how to add a new model.

---

## Dev Setup

We use [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repo
git clone https://github.com/your-org/uncertainty-flow.git
cd uncertainty-flow

# Install uv if you don't have it
curl -Lf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install all dependencies
uv sync --all-extras

# Activate the environment
source .venv/bin/activate

# Verify setup
python -c "import uncertainty_flow; print('Setup OK')"
```

### Optional dependencies

```bash
# For plot() support
uv add --optional matplotlib

# For conformal wrappers using MAPIE backend
uv add --optional mapie
```

---

## Running Tests

```bash
# Run full test suite
uv run pytest

# Run with coverage report
uv run pytest --cov=uncertainty_flow --cov-report=term-missing

# Run a specific module
uv run pytest tests/test_conformal.py

# Run only fast tests (skip slow integration tests)
uv run pytest -m "not slow"
```

Tests are in `tests/`. Each module in `uncertainty_flow/` has a corresponding `tests/test_<module>.py`.

---

## Code Style

We use `ruff` for linting and formatting.

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Both (recommended before committing)
uv run ruff check . && uv run ruff format .
```

Type hints are required on all public functions and methods. We use `mypy` for static type checking:

```bash
uv run mypy uncertainty_flow/
```

---

## Project Conventions

### Polars / NumPy boundary
- **Public API:** always accepts and returns Polars DataFrames, Series, or LazyFrames.
- **Internal compute:** always uses NumPy arrays. Conversion happens *only* in `utils/polars_bridge.py`.
- Never call `.to_numpy()` or `pl.from_numpy()` outside of `polars_bridge.py`.

### Warnings
- All user-facing warnings use the constants in `utils/warnings.py`. Do not write raw warning strings inline.
- Use `warnings.warn(UF_W001.format(...), UncertaintyFlowWarning)` pattern.
- Warning codes are documented in `API_SPEC.md`.

### DistributionPrediction
- All `.predict()` methods must return a `DistributionPrediction` object. Never return raw arrays from a public method.

### Guarantee documentation
- Every new model class **must** have a docstring that explicitly states:
  - Coverage guarantee: `GUARANTEED` or `EMPIRICAL ONLY`
  - Non-crossing: `BY CONSTRUCTION` / `POST-SORT` / `NOT GUARANTEED`
  - Any assumptions required for the guarantee to hold
- This is also reflected in `MODELS.md` — update that file when adding a model.

---

## Adding a New Model

### Step 1: Create the model file

Add a new file in `uncertainty_flow/models/` or `uncertainty_flow/wrappers/`:

```python
# uncertainty_flow/models/my_model.py

from uncertainty_flow.core.base import BaseUncertaintyModel
from uncertainty_flow.core.distribution import DistributionPrediction
from uncertainty_flow.utils.polars_bridge import to_numpy, to_polars
import polars as pl
import numpy as np


class MyModel(BaseUncertaintyModel):
    """
    One-line description.

    Coverage guarantee: EMPIRICAL ONLY / GUARANTEED (state assumption)
    Non-crossing: POST-SORT / BY CONSTRUCTION

    Parameters
    ----------
    targets : str or list[str]
        Target column name(s).
    ...
    """

    def __init__(self, targets: str | list[str], ...): 
        ...

    def fit(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> "MyModel":
        X, y = to_numpy(data, feature_cols), to_numpy(data, target_cols)
        # ... fit logic ...
        self._run_residual_analysis(X, y)  # always call this post-fit
        return self

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> DistributionPrediction:
        X = to_numpy(data, feature_cols)
        # ... predict logic → quantile_matrix (np.ndarray shape N x Q) ...
        return DistributionPrediction(
            quantile_matrix=quantile_matrix,
            quantile_levels=self.quantile_levels,
            target_names=self._target_names,
        )
```

### Step 2: Export from `__init__.py`

```python
# uncertainty_flow/models/__init__.py
from .my_model import MyModel
```

### Step 3: Add to the public API

```python
# uncertainty_flow/__init__.py
from .models import MyModel
```

### Step 4: Write tests

Create `tests/test_my_model.py`:

```python
import polars as pl
import numpy as np
import pytest
from uncertainty_flow.models import MyModel

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    n = 300
    return pl.DataFrame({
        "feature_1": rng.normal(size=n),
        "feature_2": rng.uniform(size=n),
        "target": rng.normal(loc=5.0, scale=2.0, size=n),
    })

def test_fit_predict(sample_data):
    model = MyModel(targets="target")
    model.fit(sample_data[:200])
    pred = model.predict(sample_data[200:])
    assert pred is not None

def test_interval_shape(sample_data):
    model = MyModel(targets="target")
    model.fit(sample_data[:200])
    pred = model.predict(sample_data[200:])
    interval = pred.interval(0.9)
    assert "lower" in interval.columns
    assert "upper" in interval.columns
    assert len(interval) == 100

def test_non_crossing(sample_data):
    model = MyModel(targets="target")
    model.fit(sample_data[:200])
    pred = model.predict(sample_data[200:])
    q = pred.quantile([0.05, 0.5, 0.95])
    assert (q["q_0.05"] <= q["q_0.5"]).all()
    assert (q["q_0.5"] <= q["q_0.95"]).all()

def test_calibration_report(sample_data):
    model = MyModel(targets="target")
    model.fit(sample_data[:200])
    report = model.calibration_report(sample_data[200:], target="target")
    assert "achieved_coverage" in report.columns
    assert "winkler_score" in report.columns

def test_small_calibration_warns(sample_data):
    with pytest.warns(match="UF-W001"):
        model = MyModel(targets="target", calibration_size=0.05)
        model.fit(sample_data[:200])

def test_too_small_calibration_raises(sample_data):
    with pytest.raises(ValueError, match="UF-E001"):
        model = MyModel(targets="target", calibration_size=0.01)
        model.fit(sample_data[:100])
```

### Step 5: Update `MODELS.md`

Add your model to the guarantee matrix and write a detailed entry. Be honest about what is and isn't guaranteed.

### Step 6: Open a PR

- Title format: `feat(models): Add MyModel`
- PR description should include: what the model does, guarantee level, any known limitations, and a link to a relevant paper if applicable.

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(models): Add QuantileForestForecaster
fix(wrappers): Correct temporal split for ConformalForecaster
docs(api): Clarify interval() confidence parameter
test(calibration): Add coverage for edge case < 20 samples
refactor(bridge): Consolidate LazyFrame materialisation logic
```

---

## PR Checklist

Before opening a PR:

- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format .` applied
- [ ] `uv run mypy uncertainty_flow/` passes
- [ ] `uv run pytest` passes with no regressions
- [ ] New model has a docstring with guarantee statements
- [ ] `MODELS.md` updated if a new model was added
- [ ] Warning codes in `API_SPEC.md` updated if new warnings added
