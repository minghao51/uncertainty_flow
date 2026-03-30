"""Calibration report generation."""

# Re-export from utils for backward compatibility
# The actual implementation is in utils/calibration_utils.py to avoid circular dependency
from ..utils.calibration_utils import calibration_report

__all__ = ["calibration_report"]
