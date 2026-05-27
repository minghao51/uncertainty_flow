"""Bayesian mixin for DistributionPrediction."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import polars as pl

from ..utils.exceptions import InvalidDataError, QuantileError


class _BayesianHost(Protocol):
    _posterior: np.ndarray | dict[str, np.ndarray] | None
    _posterior_predictive: np.ndarray | None
    _posterior_chains: dict[str, np.ndarray] | None


class BayesianMixin:
    def posterior_samples(self: _BayesianHost) -> np.ndarray:
        if self._posterior is None:
            raise InvalidDataError(
                "posterior_samples() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        if isinstance(self._posterior, np.ndarray):
            return self._posterior

        matrices = []
        for arr in self._posterior.values():
            arr_np = np.asarray(arr)
            if arr_np.ndim == 1:
                matrices.append(arr_np[:, np.newaxis])
            else:
                matrices.append(arr_np.reshape(arr_np.shape[0], -1))
        if not matrices:
            raise InvalidDataError("posterior data is empty")
        return np.column_stack(matrices)

    def posterior_parameter_interval(self: _BayesianHost, confidence: float = 0.9) -> pl.DataFrame:
        if not (0 < confidence < 1):
            raise QuantileError(f"confidence must be in (0, 1), got {confidence}")
        samples = self.posterior_samples()
        alpha = (1 - confidence) / 2
        lower = np.quantile(samples, alpha, axis=0)
        upper = np.quantile(samples, 1 - alpha, axis=0)
        return pl.DataFrame({"lower": lower, "upper": upper}, orient="row")

    def credible_interval(self: _BayesianHost, confidence: float = 0.9) -> pl.DataFrame:
        import warnings

        if not (0 < confidence < 1):
            raise QuantileError(f"confidence must be in (0, 1), got {confidence}")

        if self._posterior_predictive is None:
            warnings.warn(
                "credible_interval() currently falls back to parameter intervals when "
                "posterior predictive draws are unavailable. Use "
                "posterior_parameter_interval() for explicit parameter intervals.",
                FutureWarning,
                stacklevel=2,
            )
            return self.posterior_parameter_interval(confidence)

        alpha = (1 - confidence) / 2
        lower = np.quantile(self._posterior_predictive, alpha, axis=1)
        upper = np.quantile(self._posterior_predictive, 1 - alpha, axis=1)
        return pl.DataFrame({"lower": lower, "upper": upper}, orient="row")

    def rhat(self: _BayesianHost) -> np.ndarray:
        if self._posterior_chains is None:
            raise InvalidDataError(
                "rhat() requires posterior chain data. "
                "Fit BayesianQuantileRegressor with num_chains > 1."
            )

        all_rhat = []
        for name, arr in self._posterior_chains.items():
            chains = np.asarray(arr)
            if chains.ndim < 2:
                raise InvalidDataError(f"posterior chain array for '{name}' must be at least 2D")
            n_chains = chains.shape[0]
            chain_len = chains.shape[1]
            if n_chains < 2:
                raise InvalidDataError(
                    f"rhat() requires at least 2 chains for '{name}', got {n_chains}."
                )
            if chain_len < 2:
                raise InvalidDataError(
                    f"rhat() requires at least 2 draws per chain for '{name}', got {chain_len}."
                )
            reshaped = chains.reshape(n_chains, chain_len, -1)
            chain_means = reshaped.mean(axis=1)
            b = chain_len * np.var(chain_means, axis=0, ddof=1)
            w = np.mean(np.var(reshaped, axis=1, ddof=1), axis=0)
            if np.any(w < 1e-10):
                raise InvalidDataError(
                    f"Within-chain variance too close to zero for R-hat calculation for '{name}'."
                )
            var_hat = (1 - 1 / chain_len) * w + (1 / chain_len) * b
            all_rhat.append(np.sqrt(var_hat / w))

        if not all_rhat:
            raise InvalidDataError(
                "rhat() requires at least one posterior chain parameter to evaluate."
            )
        return np.concatenate(all_rhat)

    def posterior_summary(self: _BayesianHost) -> pl.DataFrame:
        if self._posterior is None:
            raise InvalidDataError(
                "posterior_summary() requires posterior data. "
                "Use a BayesianQuantileRegressor to generate predictions with posteriors."
            )
        samples = self.posterior_samples()
        return pl.DataFrame(
            {
                "mean": np.mean(samples, axis=0),
                "std": np.std(samples, axis=0),
                "q025": np.quantile(samples, 0.025, axis=0),
                "q50": np.quantile(samples, 0.5, axis=0),
                "q975": np.quantile(samples, 0.975, axis=0),
            }
        )
