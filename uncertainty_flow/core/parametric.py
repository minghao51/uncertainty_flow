"""Parametric distribution fitting from quantile predictions."""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


class ParametricDistribution:
    """
    Parametric distribution fitted from quantile predictions.

    Constructed from a ``DistributionPrediction`` (or raw quantile data)
    by matching the empirical CDF (interpolated through quantile knots) to a
    parametric family via least-squares minimisation.

    Supported families: ``"normal"``, ``"student_t"``, ``"lognormal"``,
    ``"gamma"``, ``"auto"`` (selects best by KS distance).
    """

    _FAMILIES = ("normal", "student_t", "lognormal", "gamma")

    def __init__(
        self,
        family: str,
        params: dict[str, float],
        quantile_values: np.ndarray | None = None,
        quantile_levels: np.ndarray | None = None,
    ):
        if family not in self._FAMILIES:
            raise ValueError(f"Unknown family {family!r}. Choose from {self._FAMILIES}")
        self.family = family
        self._params = params
        self._quantile_values = quantile_values
        self._quantile_levels = quantile_levels

    @property
    def mean(self) -> float:
        return float(self._rv().mean())

    @property
    def variance(self) -> float:
        return float(self._rv().var())

    @property
    def shape_params(self) -> dict[str, float]:
        return dict(self._params)

    def pdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self._rv().pdf(x)

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self._rv().cdf(x)

    def ppf(self, q: np.ndarray | float) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return self._rv().ppf(q)

    def logpdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self._rv().logpdf(x)

    def sample(self, n: int, random_state: int | None = None) -> np.ndarray:
        return self._rv().rvs(size=n, random_state=random_state)

    def _rv(self) -> stats.rv_continuous:
        p = self._params
        if self.family == "normal":
            return stats.norm(loc=p["loc"], scale=p["scale"])
        if self.family == "student_t":
            return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"])
        if self.family == "lognormal":
            return stats.lognorm(s=p["s"], loc=p["loc"], scale=p["scale"])
        if self.family == "gamma":
            return stats.gamma(a=p["a"], loc=p["loc"], scale=p["scale"])
        raise ValueError(f"Unhandled family: {self.family}")

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self._params.items())
        return f"ParametricDistribution(family={self.family!r}, {parts})"


def _fit_family(
    family: str,
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float]:
    qv = np.sort(quantile_values)

    if family == "normal":
        loc = float(np.mean(qv))
        scale = float(max(np.std(qv), 1e-8))
        return {"loc": loc, "scale": scale}

    if family == "student_t":
        loc = float(np.mean(qv))
        scale = float(max(np.std(qv), 1e-8))
        return {"df": 5.0, "loc": loc, "scale": scale}

    if family == "lognormal":
        pos = qv[qv > 0]
        if len(pos) < 2:
            pos = np.abs(qv) + 1e-8
        log_data = np.log(pos)
        mu = float(np.mean(log_data))
        sigma = float(max(np.std(log_data), 1e-8))
        return {"s": sigma, "loc": 0.0, "scale": float(np.exp(mu))}

    if family == "gamma":
        pos = qv[qv > 0]
        if len(pos) < 2:
            pos = np.abs(qv) + 1e-8
        mean_val = float(np.mean(pos))
        var_val = float(max(np.var(pos), 1e-8))
        a = mean_val**2 / var_val
        scale = var_val / mean_val
        return {"a": max(a, 1e-8), "loc": 0.0, "scale": max(scale, 1e-8)}

    raise ValueError(f"Unknown family: {family}")


def _refine_params(
    family: str,
    initial_params: dict[str, float],
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float]:
    qv = np.sort(quantile_values)
    ql = quantile_levels

    def _objective(x: np.ndarray, fam: str) -> float:
        try:
            if fam == "normal":
                rv = stats.norm(loc=x[0], scale=max(x[1], 1e-8))
            elif fam == "student_t":
                rv = stats.t(df=max(x[0], 1.0), loc=x[1], scale=max(x[2], 1e-8))
            elif fam == "lognormal":
                rv = stats.lognorm(s=max(x[0], 1e-8), loc=x[1], scale=max(x[2], 1e-8))
            elif fam == "gamma":
                rv = stats.gamma(a=max(x[0], 1e-8), loc=x[1], scale=max(x[2], 1e-8))
            else:
                return np.inf
            predicted = rv.ppf(ql)
            return float(np.sum((predicted - qv) ** 2))
        except (ValueError, OverflowError):
            return np.inf

    if family == "normal":
        x0 = np.array([initial_params["loc"], initial_params["scale"]])
        bounds: list[tuple[float | None, float | None]] = [(None, None), (1e-8, None)]
    elif family == "student_t":
        x0 = np.array([initial_params["df"], initial_params["loc"], initial_params["scale"]])
        bounds = [(1.01, 100.0), (None, None), (1e-8, None)]
    elif family == "lognormal":
        x0 = np.array([initial_params["s"], initial_params["loc"], initial_params["scale"]])
        bounds = [(1e-8, None), (None, None), (1e-8, None)]
    elif family == "gamma":
        x0 = np.array([initial_params["a"], initial_params["loc"], initial_params["scale"]])
        bounds = [(1e-8, None), (None, None), (1e-8, None)]
    else:
        return initial_params

    try:
        result = optimize.minimize(
            _objective,
            x0,
            args=(family,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
        x = result.x
        if family == "normal":
            return {"loc": float(x[0]), "scale": float(max(x[1], 1e-8))}
        elif family == "student_t":
            return {
                "df": float(max(x[0], 1.01)),
                "loc": float(x[1]),
                "scale": float(max(x[2], 1e-8)),
            }
        elif family == "lognormal":
            return {
                "s": float(max(x[0], 1e-8)),
                "loc": float(x[1]),
                "scale": float(max(x[2], 1e-8)),
            }
        elif family == "gamma":
            return {
                "a": float(max(x[0], 1e-8)),
                "loc": float(x[1]),
                "scale": float(max(x[2], 1e-8)),
            }
    except (ValueError, OverflowError, RuntimeError):
        pass

    return initial_params


def _ks_distance(family: str, params: dict[str, float], qv: np.ndarray, ql: np.ndarray) -> float:
    if family == "normal":
        rv = stats.norm(loc=params["loc"], scale=params["scale"])
    elif family == "student_t":
        rv = stats.t(df=params["df"], loc=params["loc"], scale=params["scale"])
    elif family == "lognormal":
        rv = stats.lognorm(s=params["s"], loc=params["loc"], scale=params["scale"])
    elif family == "gamma":
        rv = stats.gamma(a=params["a"], loc=params["loc"], scale=params["scale"])
    else:
        return np.inf

    try:
        predicted = rv.ppf(ql)
        return float(np.max(np.abs(predicted - qv)))
    except (ValueError, OverflowError):
        return np.inf


def fit_parametric(
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
    family: str = "auto",
) -> ParametricDistribution:
    """
    Fit a parametric distribution from quantile knots.

    Args:
        quantile_values: 1-D array of predicted quantile values (sorted).
        quantile_levels: 1-D array of quantile levels in (0, 1).
        family: Distribution family or ``"auto"`` for best selection.

    Returns:
        Fitted ``ParametricDistribution``.
    """
    qv = np.sort(np.asarray(quantile_values, dtype=np.float64))
    ql = np.asarray(quantile_levels, dtype=np.float64)

    if family == "auto":
        best_dist = None
        best_ks = np.inf
        for fam in ParametricDistribution._FAMILIES:
            try:
                init = _fit_family(fam, qv, ql)
                refined = _refine_params(fam, init, qv, ql)
                ks = _ks_distance(fam, refined, qv, ql)
                if ks < best_ks:
                    best_ks = ks
                    best_dist = ParametricDistribution(fam, refined, qv, ql)
            except (ValueError, OverflowError, RuntimeError):
                continue
        if best_dist is None:
            init = _fit_family("normal", qv, ql)
            best_dist = ParametricDistribution("normal", init, qv, ql)
        return best_dist

    init = _fit_family(family, qv, ql)
    refined = _refine_params(family, init, qv, ql)
    return ParametricDistribution(family, refined, qv, ql)
