"""Parametric distribution fitting from quantile predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats


class ParametricDistribution:
    """
    Parametric distribution fitted from quantile predictions.

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
        return _FAMILY_REGISTRY[self.family].rv_factory(self._params)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self._params.items())
        return f"ParametricDistribution(family={self.family!r}, {parts})"


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FamilySpec:
    name: str
    param_keys: list[str]
    rv_factory: Callable
    initial_fit: Callable
    bounds: list[tuple[float | None, float | None]]
    x0_from_params: Callable
    params_from_x: Callable


def _normal_rv(p: dict[str, float]) -> stats.rv_continuous:
    return stats.norm(loc=p["loc"], scale=p["scale"])


def _student_t_rv(p: dict[str, float]) -> stats.rv_continuous:
    return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"])


def _lognormal_rv(p: dict[str, float]) -> stats.rv_continuous:
    return stats.lognorm(s=p["s"], loc=p["loc"], scale=p["scale"])


def _gamma_rv(p: dict[str, float]) -> stats.rv_continuous:
    return stats.gamma(a=p["a"], loc=p["loc"], scale=p["scale"])


def _fit_normal(qv: np.ndarray) -> dict[str, float]:
    loc = float(np.mean(qv))
    scale = float(max(np.std(qv), 1e-8))
    return {"loc": loc, "scale": scale}


def _fit_student_t(qv: np.ndarray) -> dict[str, float]:
    loc = float(np.mean(qv))
    scale = float(max(np.std(qv), 1e-8))
    return {"df": 5.0, "loc": loc, "scale": scale}


def _fit_lognormal(qv: np.ndarray) -> dict[str, float]:
    pos = qv[qv > 0]
    if len(pos) < 2:
        pos = np.abs(qv) + 1e-8
    log_data = np.log(pos)
    mu = float(np.mean(log_data))
    sigma = float(max(np.std(log_data), 1e-8))
    return {"s": sigma, "loc": 0.0, "scale": float(np.exp(mu))}


def _fit_gamma(qv: np.ndarray) -> dict[str, float]:
    pos = qv[qv > 0]
    if len(pos) < 2:
        pos = np.abs(qv) + 1e-8
    mean_val = float(np.mean(pos))
    var_val = float(max(np.var(pos), 1e-8))
    a = mean_val**2 / var_val
    scale = var_val / mean_val
    return {"a": max(a, 1e-8), "loc": 0.0, "scale": max(scale, 1e-8)}


def _normal_x0(p: dict[str, float]) -> np.ndarray:
    return np.array([p["loc"], p["scale"]])


def _student_t_x0(p: dict[str, float]) -> np.ndarray:
    return np.array([p["df"], p["loc"], p["scale"]])


def _lognormal_x0(p: dict[str, float]) -> np.ndarray:
    return np.array([p["s"], p["loc"], p["scale"]])


def _gamma_x0(p: dict[str, float]) -> np.ndarray:
    return np.array([p["a"], p["loc"], p["scale"]])


def _normal_from_x(x: np.ndarray) -> dict[str, float]:
    return {"loc": float(x[0]), "scale": float(max(x[1], 1e-8))}


def _student_t_from_x(x: np.ndarray) -> dict[str, float]:
    return {"df": float(max(x[0], 1.01)), "loc": float(x[1]), "scale": float(max(x[2], 1e-8))}


def _lognormal_from_x(x: np.ndarray) -> dict[str, float]:
    return {"s": float(max(x[0], 1e-8)), "loc": float(x[1]), "scale": float(max(x[2], 1e-8))}


def _gamma_from_x(x: np.ndarray) -> dict[str, float]:
    return {"a": float(max(x[0], 1e-8)), "loc": float(x[1]), "scale": float(max(x[2], 1e-8))}


_FAMILY_REGISTRY: dict[str, _FamilySpec] = {
    "normal": _FamilySpec(
        name="normal",
        param_keys=["loc", "scale"],
        rv_factory=_normal_rv,
        initial_fit=_fit_normal,
        bounds=[(None, None), (1e-8, None)],
        x0_from_params=_normal_x0,
        params_from_x=_normal_from_x,
    ),
    "student_t": _FamilySpec(
        name="student_t",
        param_keys=["df", "loc", "scale"],
        rv_factory=_student_t_rv,
        initial_fit=_fit_student_t,
        bounds=[(1.01, 100.0), (None, None), (1e-8, None)],
        x0_from_params=_student_t_x0,
        params_from_x=_student_t_from_x,
    ),
    "lognormal": _FamilySpec(
        name="lognormal",
        param_keys=["s", "loc", "scale"],
        rv_factory=_lognormal_rv,
        initial_fit=_fit_lognormal,
        bounds=[(1e-8, None), (None, None), (1e-8, None)],
        x0_from_params=_lognormal_x0,
        params_from_x=_lognormal_from_x,
    ),
    "gamma": _FamilySpec(
        name="gamma",
        param_keys=["a", "loc", "scale"],
        rv_factory=_gamma_rv,
        initial_fit=_fit_gamma,
        bounds=[(1e-8, None), (None, None), (1e-8, None)],
        x0_from_params=_gamma_x0,
        params_from_x=_gamma_from_x,
    ),
}


# ---------------------------------------------------------------------------
# Internal fitting logic
# ---------------------------------------------------------------------------


def _refine_params_impl(
    family: str,
    initial_params: dict[str, float],
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float]:
    spec = _FAMILY_REGISTRY[family]
    qv = np.sort(quantile_values)
    ql = quantile_levels

    def _objective(x: np.ndarray) -> float:
        try:
            params = spec.params_from_x(x)
            rv = spec.rv_factory(params)
            predicted = rv.ppf(ql)
            return float(np.sum((predicted - qv) ** 2))
        except (ValueError, OverflowError):
            return np.inf

    x0 = spec.x0_from_params(initial_params)
    try:
        result = optimize.minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=spec.bounds,
            options={"maxiter": 200},
        )
        return spec.params_from_x(result.x)
    except (ValueError, OverflowError, RuntimeError):
        pass

    return initial_params


def _ks_distance_impl(
    spec: _FamilySpec,
    params: dict[str, float],
    qv: np.ndarray,
    ql: np.ndarray,
) -> float:
    try:
        rv = spec.rv_factory(params)
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
        quantile_values: 1-D array of predicted quantile values.
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
            spec = _FAMILY_REGISTRY[fam]
            try:
                init = spec.initial_fit(qv)
                refined = _refine_params_impl(fam, init, qv, ql)
                ks = _ks_distance_impl(spec, refined, qv, ql)
                if ks < best_ks:
                    best_ks = ks
                    best_dist = ParametricDistribution(fam, refined, qv, ql)
            except (ValueError, OverflowError, RuntimeError):
                continue
        if best_dist is None:
            spec = _FAMILY_REGISTRY["normal"]
            init = spec.initial_fit(qv)
            best_dist = ParametricDistribution("normal", init, qv, ql)
        return best_dist

    spec = _FAMILY_REGISTRY[family]
    init = spec.initial_fit(qv)
    refined = _refine_params_impl(family, init, qv, ql)
    return ParametricDistribution(family, refined, qv, ql)


# ---------------------------------------------------------------------------
# Backward-compatible public helpers (used by tests)
# ---------------------------------------------------------------------------


def _fit_family(
    family: str,
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float]:
    """Initial parameter fit for a distribution family (public for testing)."""
    return _FAMILY_REGISTRY[family].initial_fit(np.sort(quantile_values))


def _refine_params(
    family: str,
    initial_params: dict[str, float],
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
) -> dict[str, float]:
    """Refine parameters via optimization (public for testing)."""
    return _refine_params_impl(family, initial_params, quantile_values, quantile_levels)


def _ks_distance(
    family: str,
    params: dict[str, float],
    qv: np.ndarray,
    ql: np.ndarray,
) -> float:
    """KS distance for a fitted family (public for testing)."""
    spec = _FAMILY_REGISTRY[family]
    return _ks_distance_impl(spec, params, qv, ql)
