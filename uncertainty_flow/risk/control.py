"""Conformal risk control for arbitrary risk functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from ..utils.exceptions import error_invalid_data

if TYPE_CHECKING:
    from ..core.base import BaseUncertaintyModel


def _prediction_mean(prediction) -> np.ndarray:
    """Return point predictions as a NumPy array."""
    mean_value = prediction.mean()
    if isinstance(mean_value, pl.Series):
        return mean_value.to_numpy()

    first_target = prediction._targets[0]
    return mean_value[first_target].to_numpy()


def _interval_half_width(prediction, confidence: float = 0.9) -> np.ndarray:
    """Return half-widths for the first target prediction interval."""
    interval = prediction.interval(confidence)
    if len(prediction._targets) == 1:
        lower = interval["lower"].to_numpy()
        upper = interval["upper"].to_numpy()
    else:
        first_target = prediction._targets[0]
        lower = interval[f"{first_target}_lower"].to_numpy()
        upper = interval[f"{first_target}_upper"].to_numpy()

    return (upper - lower) / 2


class ConformalRiskControl:
    """
    Conformal prediction for controlling arbitrary risk functions.

    Unlike traditional conformal prediction that controls coverage probability,
    this class controls expected risk for arbitrary user-defined risk functions.

    Parameters
    ----------
    base_model : BaseUncertaintyModel
        Fitted uncertainty model with predict() method
    risk_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Risk function that takes (y_true, y_pred) and returns risk values
    target_risk : float, default=0.1
        Target expected risk level to control
    calibration_method : str, default="quantile"
        Method for computing risk threshold ("quantile" or "mean")
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> import polars as pl
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from uncertainty_flow.wrappers import ConformalRegressor
    >>> from uncertainty_flow.risk import ConformalRiskControl, asymmetric_loss
    >>>
    >>> # Base model
    >>> base_model = GradientBoostingRegressor(random_state=42)
    >>> conformal_model = ConformalRegressor(base_model)
    >>> conformal_model.fit(train_data, target="y")
    >>>
    >>> # Wrap with risk control
    >>> risk_model = ConformalRiskControl(
    ...     base_model=conformal_model,
    ...     risk_function=asymmetric_loss(underprediction_penalty=2.0),
    ...     target_risk=0.1,
    ... )
    >>>
    >>> # Predictions are risk-calibrated
    >>> pred = risk_model.predict(test_data)
    """

    def __init__(
        self,
        base_model: "BaseUncertaintyModel",
        risk_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        target_risk: float = 0.1,
        calibration_method: str = "quantile",
        random_state: int | None = None,
    ):
        self.base_model = base_model
        self.risk_function = risk_function
        self.target_risk = target_risk
        self.calibration_method = calibration_method
        self.random_state = random_state

        # Fitted attributes
        self._risk_threshold: float | None = None
        self._calibration_risks: np.ndarray | None = None
        self._calibration_proxy: np.ndarray | None = None
        self._proxy_grid: np.ndarray | None = None
        self._risk_curve: np.ndarray | None = None

    def fit(
        self,
        data: pl.DataFrame,
        target: str,
    ) -> "ConformalRiskControl":
        """
        Fit risk calibration using calibration data.

        Computes risk threshold needed to achieve target risk level.

        Args:
            data: Calibration data with features and target
            target: Target column name

        Returns
        -------
        self
            Fitted ConformalRiskControl instance
        """
        from ..utils.polars_bridge import to_numpy

        features = data.drop(target)
        y_true = to_numpy(data, [target]).flatten()

        prediction = self.base_model.predict(features)
        y_pred = _prediction_mean(prediction)
        proxy = _interval_half_width(prediction)

        risks = self.risk_function(y_true, y_pred)
        if risks.shape != proxy.shape:
            raise ValueError(
                "risk_function must return one scalar risk value per sample. "
                f"Got shape {risks.shape}, expected {proxy.shape}."
            )

        metric_fn = self._risk_metric_fn()
        order = np.argsort(proxy)
        sorted_proxy = proxy[order]
        sorted_risks = risks[order]
        unique_proxy, unique_idx = np.unique(sorted_proxy, return_index=True)

        risk_curve = np.empty_like(unique_proxy, dtype=float)
        threshold = float(unique_proxy[0])

        for i, start_idx in enumerate(unique_idx):
            accepted_risks = sorted_risks[: start_idx + 1]
            metric_value = float(metric_fn(accepted_risks))
            risk_curve[i] = metric_value
            if metric_value <= self.target_risk:
                threshold = float(unique_proxy[i])

        self._calibration_risks = risks
        self._calibration_proxy = proxy
        self._proxy_grid = unique_proxy
        self._risk_curve = risk_curve
        self._risk_threshold = threshold

        if self._risk_threshold is None:
            raise ValueError(
                f"Unknown calibration_method: {self.calibration_method}. Use 'quantile' or 'mean'."
            )

        return self

    def predict(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Generate risk-calibrated predictions.

        Args:
            data: Feature DataFrame for prediction

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
                - prediction: Point predictions
                - risk: Expected risk for each prediction
                - exceeds_threshold: Whether risk exceeds calibrated threshold

        Raises
        ------
        InvalidDataError
            If model has not been fitted
        """
        if self._risk_threshold is None:
            error_invalid_data(
                "ConformalRiskControl must be fitted before prediction. Call fit() first."
            )

        prediction = self.base_model.predict(data)
        y_pred = _prediction_mean(prediction)
        proxy = _interval_half_width(prediction)
        risks = self._estimate_risk(proxy)

        return pl.DataFrame(
            {
                "prediction": y_pred,
                "risk": risks,
                "exceeds_threshold": proxy > self._risk_threshold,
            }
        )

    def _risk_metric_fn(self) -> Callable[[np.ndarray], float]:
        """Return the calibration metric used to map proxy to realized risk."""
        if self.calibration_method == "mean":
            return lambda values: float(np.mean(values))
        if self.calibration_method == "quantile":
            quantile_level = min(max(1 - self.target_risk, 0.5), 0.99)
            return lambda values: float(np.quantile(values, quantile_level))

        raise ValueError(
            f"Unknown calibration_method: {self.calibration_method}. Use 'quantile' or 'mean'."
        )

    def _estimate_risk(self, proxy: np.ndarray) -> np.ndarray:
        """Estimate realized risk from the calibration proxy curve."""
        if self._proxy_grid is None or self._risk_curve is None:
            error_invalid_data(
                "ConformalRiskControl must be fitted before prediction. Call fit() first."
            )

        risk_curve = self._risk_curve
        proxy_grid = self._proxy_grid
        assert risk_curve is not None and proxy_grid is not None
        return np.interp(
            proxy,
            proxy_grid,
            risk_curve,
            left=float(risk_curve[0]),
            right=float(risk_curve[-1]),
        )

    def risk_threshold(self) -> float:
        """
        Return the calibrated risk threshold.

        Returns
        -------
        float
            Risk threshold used for predictions

        Raises
        ------
        InvalidDataError
            If model has not been fitted
        """
        if self._risk_threshold is None:
            error_invalid_data(
                "ConformalRiskControl must be fitted before accessing risk_threshold. "
                "Call fit() first."
            )
        assert self._risk_threshold is not None
        return self._risk_threshold

    def summary(self) -> dict[str, object]:
        """
        Return summary of the risk control configuration.

        Returns
        -------
        dict
            Dictionary with configuration and calibration results
        """
        return {
            "target_risk": self.target_risk,
            "calibration_method": self.calibration_method,
            "risk_threshold": self._risk_threshold,
            "n_calibration_samples": (
                len(self._calibration_risks) if self._calibration_risks is not None else None
            ),
        }
