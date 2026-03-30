"""Tests for configuration system."""

import pytest

from uncertainty_flow.core.config import (
    QuantileConfig,
    get_config,
    reset_config,
    set_config,
)


class TestQuantileConfig:
    """Test QuantileConfig validation and defaults."""

    def test_default_quantiles(self):
        """Default quantiles should be standard list."""
        config = QuantileConfig()
        expected = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        assert config.default_quantiles == expected

    def test_min_calibration_size_default(self):
        """Default min calibration size should be 20."""
        config = QuantileConfig()
        assert config.min_calibration_size == 20

    def test_warn_calibration_size_default(self):
        """Default warn calibration size should be 50."""
        config = QuantileConfig()
        assert config.warn_calibration_size == 50

    def test_custom_quantiles(self):
        """Should accept custom quantile list."""
        custom = [0.1, 0.5, 0.9]
        config = QuantileConfig(default_quantiles=custom)
        assert config.default_quantiles == custom

    def test_custom_calibration_sizes(self):
        """Should accept custom calibration sizes."""
        config = QuantileConfig(
            min_calibration_size=10,
            warn_calibration_size=30,
        )
        assert config.min_calibration_size == 10
        assert config.warn_calibration_size == 30


class TestQuantileValidation:
    """Test quantile validation."""

    def test_empty_quantiles_raises_error(self):
        """Empty quantile list should raise ValueError."""
        with pytest.raises(ValueError, match="Quantile list cannot be empty"):
            QuantileConfig(default_quantiles=[])

    def test_quantile_out_of_range_raises_error(self):
        """Quantiles outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            QuantileConfig(default_quantiles=[0.0, 0.5, 1.0])

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            QuantileConfig(default_quantiles=[-0.1, 0.5, 0.9])

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            QuantileConfig(default_quantiles=[0.1, 0.5, 1.1])

    def test_duplicate_quantiles_warning(self):
        """Duplicate quantiles should trigger warning and be deduplicated."""
        with pytest.warns(UserWarning, match="Duplicate quantiles"):
            config = QuantileConfig(default_quantiles=[0.1, 0.5, 0.5, 0.9])
        assert config.default_quantiles == [0.1, 0.5, 0.9]

    def test_quantiles_sorted(self):
        """Quantiles should be sorted even if provided unsorted."""
        config = QuantileConfig(default_quantiles=[0.9, 0.1, 0.5])
        assert config.default_quantiles == [0.1, 0.5, 0.9]


class TestCalibrationSizeValidation:
    """Test calibration size validation."""

    def test_warn_threshold_less_than_min_raises_error(self):
        """warn_calibration_size must be >= min_calibration_size."""
        with pytest.raises(
            ValueError,
            match="warn_calibration_size \\(30\\) must be >= min_calibration_size \\(50\\)",
        ):
            QuantileConfig(
                min_calibration_size=50,
                warn_calibration_size=30,
            )


class TestGlobalConfig:
    """Test global configuration instance."""

    def test_get_config_returns_singleton(self):
        """get_config should return same instance on subsequent calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_changes_global(self):
        """set_config should change the global instance."""
        custom = QuantileConfig(default_quantiles=[0.1, 0.5, 0.9])

        set_config(custom)
        assert get_config() is custom
        assert get_config().default_quantiles == [0.1, 0.5, 0.9]

        # Reset to defaults
        reset_config()
        assert get_config() is not custom

    def test_reset_config_restores_defaults(self):
        """reset_config should restore default configuration."""
        custom = QuantileConfig(default_quantiles=[0.1, 0.5, 0.9])
        set_config(custom)

        reset_config()
        config = get_config()

        assert config.default_quantiles == [
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
        ]


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_env_var_default_quantiles(self, monkeypatch):
        """UNCERTAINTY_FLOW_DEFAULT_QUANTILES should override defaults."""
        # Use JSON array format for env var
        monkeypatch.setenv("UNCERTAINTY_FLOW_DEFAULT_QUANTILES", "[0.1,0.5,0.9]")
        reset_config()  # Reset to pick up new env var
        config = get_config()
        assert config.default_quantiles == [0.1, 0.5, 0.9]
        reset_config()  # Clean up

    def test_env_var_min_calibration_size(self, monkeypatch):
        """UNCERTAINTY_FLOW_MIN_CALIBRATION_SIZE should override default."""
        monkeypatch.setenv("UNCERTAINTY_FLOW_MIN_CALIBRATION_SIZE", "30")
        reset_config()
        config = get_config()
        assert config.min_calibration_size == 30
        reset_config()

    def test_env_var_warn_calibration_size(self, monkeypatch):
        """UNCERTAINTY_FLOW_WARN_CALIBRATION_SIZE should override default."""
        monkeypatch.setenv("UNCERTAINTY_FLOW_WARN_CALIBRATION_SIZE", "40")
        reset_config()
        config = get_config()
        assert config.warn_calibration_size == 40
        reset_config()
