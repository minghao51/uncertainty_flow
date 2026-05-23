"""Deep benchmark orchestration flow."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from uncertainty_flow.metrics import coverage_score, pinball_loss, winkler_score
from uncertainty_flow.utils.exceptions import RECOVERABLE_EXCEPTIONS, DataError
from uncertainty_flow.utils.polars_bridge import to_numpy_series
from uncertainty_flow.utils.split import RollingOriginSplit

from .configs import BenchmarkConfig, ModelBuildConfig
from .datasets import DatasetInfo, load_dataset
from .providers import BenchmarkModelProvider, resolve_provider
from .results import BenchmarkResult, ModelResult
from .tuning import TuningConfig, auto_tune_model

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    """Dataset payload for flow execution."""

    df: pl.DataFrame
    ds_info: DatasetInfo
    target: str


class BenchmarkFlow:
    """Orchestrates benchmark run lifecycle behind a compact interface."""

    def __init__(
        self,
        config: BenchmarkConfig,
        providers: dict[str, BenchmarkModelProvider],
        class_registry: dict[str, type],
    ) -> None:
        self.config = config
        self.providers = providers
        self.class_registry = class_registry
        self.loaded: LoadedDataset | None = None
        self._tuning_cache: dict[str, dict[str, Any]] = {}
        self._tuning_result_cache: dict[str, Any] = {}

    def load_data(self) -> LoadedDataset:
        df, ds_info = load_dataset(
            self.config.dataset_name,
            n_samples=self.config.n_samples,
            revision=self.config.dataset_revision,
        )
        target = self.config.target_column or ds_info.default_target
        self.loaded = LoadedDataset(df=df, ds_info=ds_info, target=target)
        return self.loaded

    def run(self, model_names: list[str] | None, allow_partial: bool = False) -> BenchmarkResult:
        loaded = self.loaded or self.load_data()
        active_names = model_names or sorted(set(self.providers) | set(self.class_registry))

        results: list[ModelResult] = []
        errors: list[dict[str, str]] = []
        for model_name in active_names:
            try:
                results.append(self._run_one(model_name, loaded))
            except (KeyboardInterrupt, SystemExit):
                raise
            except RECOVERABLE_EXCEPTIONS as e:
                logger.exception("Benchmark model '%s' failed: %s", model_name, e)
                errors.append({"model": model_name, "error": str(e)})
                if not allow_partial:
                    raise RuntimeError(f"Benchmark failed for model '{model_name}': {e}") from e

        if not results:
            if errors:
                raise RuntimeError(
                    f"Benchmark produced no successful model results. Errors: {errors}"
                )
            raise RuntimeError("Benchmark produced no model results.")

        return BenchmarkResult(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=self.config.dataset_name,
            dataset_domain=loaded.ds_info.domain,
            n_samples=self.config.n_samples,
            horizon=self.config.horizon,
            models=results,
            errors=errors,
        )

    def _run_one(self, model_name: str, loaded: LoadedDataset) -> ModelResult:
        tune_df, train_df, test_df = self._train_test_split(loaded.df)
        tuning_result = self._get_tuning_result(model_name, tune_df, loaded.target)

        if self.config.rolling_origin:
            return self._run_one_rolling(model_name, loaded, tuning_result)

        return self._evaluate_single_split(
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            target=loaded.target,
            tuning_result=tuning_result,
        )

    def _train_test_split(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        n_total = len(df)

        if n_total < 3:
            raise DataError("Dataset must contain at least 3 rows for benchmark train/test split.")

        n_test = int(n_total * self.config.test_size)
        if n_test < 1:
            raise DataError(
                "test_size produced an empty test split. "
                "Increase test_size or n_samples so at least one test row is retained."
            )

        n_non_test = n_total - n_test
        if n_non_test < 2:
            raise DataError(
                "test_size produced an empty train split. "
                "Decrease test_size so at least two non-test rows are retained for tune/train."
            )

        n_tune = max(1, int(n_non_test * self.config.tune_size))
        if n_tune >= n_non_test:
            n_tune = max(1, n_non_test // 2)

        tune_df = df.head(n_tune)
        train_df = df.slice(n_tune, n_non_test - n_tune)
        test_df = df.tail(n_test)
        return tune_df, train_df, test_df

    def _get_tuning_result(self, model_name: str, tune_df: pl.DataFrame, target: str) -> Any | None:
        if model_name in self._tuning_result_cache:
            return self._tuning_result_cache[model_name]
        if not self.config.auto_tune:
            return None

        tune_config = TuningConfig(
            target_coverage=self.config.target_coverage,
            n_samples=self.config.tune_samples,
            timeout=self.config.tune_timeout,
            hybrid_validation=self.config.hybrid_validation,
        )
        tuning_result = auto_tune_model(
            model_name=model_name,
            df=tune_df,
            target=target,
            horizon=self.config.horizon,
            config=tune_config,
        )
        self._tuning_cache[model_name] = tuning_result.best_params
        self._tuning_result_cache[model_name] = tuning_result
        return tuning_result

    def _build_model(
        self,
        model_name: str,
        target: str,
        tuning_result: Any | None,
        provider: BenchmarkModelProvider | None = None,
    ):
        tuned_params = tuning_result.best_params if tuning_result is not None else {}
        if provider is None:
            provider = self._resolve_provider(model_name)
        build_config = ModelBuildConfig(
            model_name=model_name,
            target_column=target,
            horizon=self.config.horizon,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            tuned_params=tuned_params,
        )
        return provider.build(build_config), tuned_params

    def _resolve_provider(self, model_name: str) -> BenchmarkModelProvider:
        return resolve_provider(model_name, self.providers, self.class_registry)

    def _evaluate_single_split(
        self,
        model_name: str,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        target: str,
        tuning_result: Any | None,
    ) -> ModelResult:
        benchmark, tuned_params = self._build_model(model_name, target, tuning_result)
        was_tuned = bool(tuned_params)

        benchmark.fit(train_df, target)
        pred = benchmark.predict(test_df)

        n_pred = len(pred.interval(0.9))
        y_true = to_numpy_series(test_df[target])[-n_pred:]
        lower_90_s, upper_90_s = pred.interval_bounds(0.9)
        lower_80_s, upper_80_s = pred.interval_bounds(0.8)
        lower_90 = to_numpy_series(lower_90_s)
        upper_90 = to_numpy_series(upper_90_s)
        lower_80 = to_numpy_series(lower_80_s)
        upper_80 = to_numpy_series(upper_80_s)

        cov_90 = coverage_score(y_true, lower_90, upper_90)
        cov_80 = coverage_score(y_true, lower_80, upper_80)
        wink_90 = winkler_score(y_true, lower_90, upper_90, confidence=0.9)
        wink_80 = winkler_score(y_true, lower_80, upper_80, confidence=0.8)
        sharp_90 = float(np.mean(upper_90 - lower_90))
        sharp_80 = float(np.mean(upper_80 - lower_80))
        pinball = float(pinball_loss(y_true, lower_90, 0.1))

        return ModelResult(
            model_name=model_name,
            coverage_90=round(cov_90, 4),
            coverage_80=round(cov_80, 4),
            sharpness_90=round(sharp_90, 4),
            sharpness_80=round(sharp_80, 4),
            winkler_90=round(wink_90, 4),
            winkler_80=round(wink_80, 4),
            pinball_loss=round(pinball, 4),
            train_time_sec=round(float(getattr(benchmark, "train_time", 0.0)), 3),
            n_samples=n_pred,
            tuned_params=tuned_params,
            was_tuned=was_tuned,
            validation_coverage_90=self._tuning_attr(tuning_result, "coverage_90"),
            validation_sharpness_90=self._tuning_attr(tuning_result, "sharpness_90"),
            validation_winkler_90=self._tuning_attr(tuning_result, "winkler_90"),
            validation_split_type=self._tuning_attr_str(tuning_result, "validation_split_type"),
            validation_strategy=self._tuning_attr_str(tuning_result, "validation_strategy"),
            validation_n_splits=self._tuning_attr_int(tuning_result, "validation_n_splits"),
        )

    def _run_one_rolling(
        self, model_name: str, loaded: LoadedDataset, tuning_result: Any | None
    ) -> ModelResult:
        splitter = RollingOriginSplit(
            n_splits=self.config.rolling_n_splits,
            min_train_size=self.config.rolling_min_train,
            horizon=self.config.rolling_horizon,
        )
        splits = splitter.splits(loaded.df)

        provider = resolve_provider(model_name, self.providers, self.class_registry)
        _, tuned_params = self._build_model(model_name, loaded.target, tuning_result, provider)
        was_tuned = bool(tuned_params)

        cov_90s: list[float] = []
        cov_80s: list[float] = []
        wink_90s: list[float] = []
        wink_80s: list[float] = []
        sharp_90s: list[float] = []
        sharp_80s: list[float] = []
        pinballs: list[float] = []
        train_times: list[float] = []
        n_samples_list: list[int] = []

        for train_df, test_df in splits:
            benchmark, _ = self._build_model(model_name, loaded.target, tuning_result, provider)
            benchmark.fit(train_df, loaded.target)
            pred = benchmark.predict(test_df)

            n_pred = len(pred.interval(0.9))
            y_true = to_numpy_series(test_df[loaded.target])[-n_pred:]
            lower_90_s, upper_90_s = pred.interval_bounds(0.9)
            lower_80_s, upper_80_s = pred.interval_bounds(0.8)
            lower_90 = to_numpy_series(lower_90_s)
            upper_90 = to_numpy_series(upper_90_s)
            lower_80 = to_numpy_series(lower_80_s)
            upper_80 = to_numpy_series(upper_80_s)

            cov_90s.append(coverage_score(y_true, lower_90, upper_90))
            cov_80s.append(coverage_score(y_true, lower_80, upper_80))
            wink_90s.append(winkler_score(y_true, lower_90, upper_90, confidence=0.9))
            wink_80s.append(winkler_score(y_true, lower_80, upper_80, confidence=0.8))
            sharp_90s.append(float(np.mean(upper_90 - lower_90)))
            sharp_80s.append(float(np.mean(upper_80 - lower_80)))
            pinballs.append(float(pinball_loss(y_true, lower_90, 0.1)))
            train_times.append(float(getattr(benchmark, "train_time", 0.0)))
            n_samples_list.append(n_pred)

        return ModelResult(
            model_name=model_name,
            coverage_90=round(float(np.mean(cov_90s)), 4),
            coverage_80=round(float(np.mean(cov_80s)), 4),
            sharpness_90=round(float(np.mean(sharp_90s)), 4),
            sharpness_80=round(float(np.mean(sharp_80s)), 4),
            winkler_90=round(float(np.mean(wink_90s)), 4),
            winkler_80=round(float(np.mean(wink_80s)), 4),
            pinball_loss=round(float(np.mean(pinballs)), 4),
            train_time_sec=round(float(np.sum(train_times)), 3),
            n_samples=int(np.sum(n_samples_list)),
            tuned_params=tuned_params,
            was_tuned=was_tuned,
            test_split_type="rolling_origin",
        )

    @staticmethod
    def _tuning_attr(tuning_result: Any | None, name: str) -> float | None:
        if tuning_result is None or not hasattr(tuning_result, name):
            return None
        return round(float(getattr(tuning_result, name, 0.0)), 4)

    @staticmethod
    def _tuning_attr_str(tuning_result: Any | None, name: str) -> str | None:
        if tuning_result is None or not hasattr(tuning_result, name):
            return None
        value = getattr(tuning_result, name, None)
        return str(value) if value is not None else None

    @staticmethod
    def _tuning_attr_int(tuning_result: Any | None, name: str) -> int | None:
        if tuning_result is None or not hasattr(tuning_result, name):
            return None
        value = getattr(tuning_result, name, None)
        return int(value) if value is not None else None
