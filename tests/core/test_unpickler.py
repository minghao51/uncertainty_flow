import io
import pickle

import numpy as np
import pytest

from uncertainty_flow.core._persistence import _ModelUnpickler


class TestModelUnpickler:
    def test_allowed_builtin_roundtrip(self):
        data = {"key": [1, 2, 3], "nested": (4, 5.0)}
        buf = pickle.dumps(data)
        result = _ModelUnpickler(io.BytesIO(buf)).load()
        assert result == data

    def test_allowed_numpy_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        buf = pickle.dumps(arr)
        result = _ModelUnpickler(io.BytesIO(buf)).load()
        np.testing.assert_array_equal(result, arr)

    def test_forbidden_os_system(self):
        import os

        class Malicious:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        buf = pickle.dumps(Malicious())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
            _ModelUnpickler(io.BytesIO(buf)).load()

    def test_forbidden_subprocess(self):
        import subprocess

        class Malicious:
            def __reduce__(self):
                return (subprocess.Popen, (["ls"],))

        buf = pickle.dumps(Malicious())
        with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
            _ModelUnpickler(io.BytesIO(buf)).load()

    def test_uncertainty_flow_model_roundtrip(self):
        import os
        import tempfile

        import polars as pl

        from uncertainty_flow.models import QuantileForestForecaster

        rng = np.random.default_rng(42)
        n = 100
        df = pl.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": 2.0 * rng.standard_normal(n),
            }
        )
        model = QuantileForestForecaster(
            targets="y", horizon=1, n_estimators=10, auto_tune=False, random_state=42
        )
        model.fit(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.uf")
            model.save(path)
            loaded = type(model).load(path)
            pred = loaded.predict(df.head(5))
            assert pred._quantiles.shape[0] == 5
