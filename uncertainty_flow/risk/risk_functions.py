"""Pre-built risk functions for conformal risk control."""

from typing import Callable

import numpy as np


def asymmetric_loss(
    overprediction_penalty: float = 1.0,
    underprediction_penalty: float = 2.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Asymmetric loss function for different penalties on over/under prediction.

    Useful when overpredictions and underpredictions have different costs.

    Parameters
    ----------
    overprediction_penalty : float, default=1.0
        Penalty coefficient for overpredictions (pred > true)
    underprediction_penalty : float, default=2.0
        Penalty coefficient for underpredictions (pred < true)

    Returns
    -------
    Callable
        Risk function that computes asymmetric loss

    Examples
    --------
    >>> import numpy as np
    >>> from uncertainty_flow.risk import asymmetric_loss
    >>>
    >>> risk_fn = asymmetric_loss(overprediction_penalty=1.0, underprediction_penalty=2.0)
    >>> y_true = np.array([10, 20, 30])
    >>> y_pred = np.array([12, 18, 32])
    >>> risk = risk_fn(y_true, y_pred)
    """

    def _risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        errors = y_pred - y_true
        loss = np.where(
            errors > 0,
            overprediction_penalty * errors,
            -underprediction_penalty * errors,
        )
        return loss

    return _risk


def threshold_penalty(
    threshold: float,
    penalty_above: float = 10.0,
    penalty_below: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Threshold-based penalty function.

    Applies higher penalty when error exceeds threshold.

    Parameters
    ----------
    threshold : float
        Error threshold for penalty escalation
    penalty_above : float, default=10.0
        Penalty when error exceeds threshold
    penalty_below : float, default=1.0
        Base penalty when error is within threshold

    Returns
    -------
    Callable
        Risk function that computes threshold penalty

    Examples
    --------
    >>> import numpy as np
    >>> from uncertainty_flow.risk import threshold_penalty
    >>>
    >>> risk_fn = threshold_penalty(threshold=5.0)
    >>> y_true = np.array([100, 100, 100])
    >>> y_pred = np.array([95, 105, 120])
    >>> risk = risk_fn(y_true, y_pred)
    """

    def _risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        errors = np.abs(y_true - y_pred)
        loss = np.where(
            errors > threshold,
            penalty_above * (errors - threshold),
            penalty_below * errors,
        )
        return loss

    return _risk


def inventory_cost(
    holding_cost: float = 1.0,
    stockout_cost: float = 10.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Inventory management cost function.

    Models the cost of holding excess inventory vs. stockouts.
    Useful for demand forecasting optimization.

    Parameters
    ----------
    holding_cost : float, default=1.0
        Cost per unit of overpredicted demand (excess inventory)
    stockout_cost : float, default=10.0
        Cost per unit of underpredicted demand (stockout)

    Returns
    -------
    Callable
        Risk function that computes inventory cost

    Examples
    --------
    >>> import numpy as np
    >>> from uncertainty_flow.risk import inventory_cost
    >>>
    >>> risk_fn = inventory_cost(holding_cost=1.0, stockout_cost=10.0)
    >>> demand = np.array([100, 150, 200])
    >>> forecast = np.array([110, 140, 210])
    >>> cost = risk_fn(demand, forecast)
    """

    def _risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Overprediction: holding cost for excess inventory
        over = np.maximum(y_pred - y_true, 0) * holding_cost
        # Underprediction: stockout cost for missed demand
        under = np.maximum(y_true - y_pred, 0) * stockout_cost
        return over + under

    return _risk


def financial_var(
    var_level: float = 0.95,
    excess_penalty: float = 10.0,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Financial Value-at-Risk (VaR) style risk function.

    Penalizes predictions that exceed VaR threshold.

    Parameters
    ----------
    var_level : float, default=0.95
        VaR confidence level (e.g., 0.95 for 95% VaR)
    excess_penalty : float, default=10.0
        Multiplier for excess loss beyond VaR threshold

    Returns
    -------
    Callable
        Risk function that computes VaR-based penalty

    Examples
    --------
    >>> import numpy as np
    >>> from uncertainty_flow.risk import financial_var
    >>>
    >>> risk_fn = financial_var(var_level=0.95)
    >>> y_true = np.array([100, 100, 100])
    >>> y_pred = np.array([95, 105, 120])
    >>> risk = risk_fn(y_true, y_pred)
    """

    def _risk(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        losses = np.abs(y_true - y_pred)
        var_threshold = np.quantile(losses, var_level)
        excess_loss = np.maximum(losses - var_threshold, 0)
        return losses + excess_penalty * excess_loss

    return _risk
