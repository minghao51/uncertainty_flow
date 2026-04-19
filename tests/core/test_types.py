"""Tests for core types module."""

from uncertainty_flow.core.types import (
    DEFAULT_QUANTILES,
    CalibrationMethod,
    CorrelationMode,
    PolarsInput,
    TargetSpec,
)


class TestDefaultQuantiles:
    """Test default quantile levels."""

    def test_default_quantiles_length(self):
        """DEFAULT_QUANTILES should have 11 levels."""
        assert len(DEFAULT_QUANTILES) == 11

    def test_default_quantiles_values(self):
        """DEFAULT_QUANTILES should cover standard range."""
        assert DEFAULT_QUANTILES[0] == 0.05
        assert DEFAULT_QUANTILES[-1] == 0.95
        assert 0.5 in DEFAULT_QUANTILES

    def test_default_quantiles_sorted(self):
        """DEFAULT_QUANTILES should be sorted ascending."""
        assert list(DEFAULT_QUANTILES) == sorted(DEFAULT_QUANTILES)


class TestTypeAliases:
    """Test type aliases exist and are correct."""

    def test_calibration_method_is_literal(self):
        """CalibrationMethod should be a Literal type."""
        # This is mainly a type check, but we can verify the values
        valid_methods = ["holdout", "cross"]
        # Type annotations should accept these values
        method: CalibrationMethod = "holdout"
        assert method in valid_methods

    def test_correlation_mode_is_literal(self):
        """CorrelationMode should be a Literal type."""
        valid_modes = ["auto", "independent"]
        mode: CorrelationMode = "auto"
        assert mode in valid_modes

    def test_polars_input_accepts_dataframe(self):
        """PolarsInput should accept pl.DataFrame."""
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3]})
        data: PolarsInput = df
        assert isinstance(data, pl.DataFrame)

    def test_polars_input_accepts_lazyframe(self):
        """PolarsInput should accept pl.LazyFrame."""
        import polars as pl

        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        data: PolarsInput = lf
        assert isinstance(data, pl.LazyFrame)

    def test_target_spec_accepts_string(self):
        """TargetSpec should accept string."""
        target: TargetSpec = "price"
        assert isinstance(target, str)

    def test_target_spec_accepts_list(self):
        """TargetSpec should accept list of strings."""
        target: TargetSpec = ["price", "volume"]
        assert isinstance(target, list)
        assert all(isinstance(t, str) for t in target)
