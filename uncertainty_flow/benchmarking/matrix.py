"""Registry-backed multi-model execution for verified benchmark runs."""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl

from .contracts import (
    ArtifactRef,
    ArtifactType,
    DegradationReason,
    ModelExecutionResult,
    ModelExecutionStatus,
    PipelineRunResult,
    RunIdentity,
    RunManifest,
    RunRequest,
    RunStatus,
    RunVerificationReport,
    VerificationCheck,
    VerificationSeverity,
    VerificationStatus,
)
from .dataflows.modeling import evaluate_metrics
from .driver import build_driver
from .identity import content_hash, derive_identity
from .lineage import materialize_dataset_lineage
from .model_contracts import ModelBuildConfig
from .operations import (
    RunLockManager,
    manifest_authenticity_valid,
    publication_secret,
    sign_manifest,
)
from .registry import default_model_registry
from .storage import LocalArtifactStore

MatrixRunResult = PipelineRunResult


def _model_frame(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [column for column in ("_split", "_fold", "id") if column in frame.columns]
    return frame.drop(columns)


def _rolling_fold_frames(
    frame: pl.DataFrame, plan: object
) -> list[tuple[int, pl.DataFrame, pl.DataFrame]]:
    """Materialize train/test frames for each persisted rolling-origin fold."""

    validation_plan = plan
    ids = (
        [str(value) for value in frame["id"].to_list()]
        if "id" in frame.columns
        else [str(index) for index in range(len(frame))]
    )
    indexed = frame.with_columns(pl.Series("__uf_row_id", ids))
    folds: list[tuple[int, pl.DataFrame, pl.DataFrame]] = []
    fold_ids = sorted({assignment.fold for assignment in validation_plan.assignments})
    for fold in fold_ids:
        if fold is None:
            continue
        assignments = [
            assignment for assignment in validation_plan.assignments if assignment.fold == fold
        ]
        train_ids = [item.observation_id for item in assignments if item.split == "train"]
        test_ids = [item.observation_id for item in assignments if item.split == "test"]
        folds.append(
            (
                fold,
                indexed.filter(pl.col("__uf_row_id").is_in(train_ids)).drop("__uf_row_id"),
                indexed.filter(pl.col("__uf_row_id").is_in(test_ids)).drop("__uf_row_id"),
            )
        )
    if not folds:
        raise ValueError("Rolling-origin validation produced no executable folds")
    return folds


def _persist_fitted_model(
    store: LocalArtifactStore, ref: ArtifactRef, adapter: object
) -> object | None:
    """Persist a provider's underlying `.uf` model archive when supported."""

    model = getattr(adapter, "model", None)
    save = getattr(model, "save", None)
    if not callable(save):
        return None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.uf"
        save(path)
        return store.write_bytes(ref, path.read_bytes())


class ModelMatrixCoordinator:
    """Execute every requested registered model with branch isolation."""

    def __init__(self, storage_root: str = "data", code_version: str = "uncertainty-flow-0.5.0"):
        self.store = LocalArtifactStore(storage_root)
        self.code_version = code_version
        self.locks = RunLockManager(storage_root)

    def run_with_lock(self, request: RunRequest, input_frame: pl.DataFrame) -> MatrixRunResult:
        """Execute the model matrix under the same run lock policy as single runs."""

        if not request.models:
            raise ValueError("At least one model is required for matrix execution")
        driver = build_driver()
        outputs = driver.execute(
            ["resolved_run_config", "validation_plan"],
            inputs={"run_request": request, "input_frame": input_frame},
        )
        self._apply_code_version(outputs)
        identity = RunIdentity(
            **derive_identity(
                source_checksum=content_hash(input_frame.to_dict(as_series=False)),
                ingestion_contract_version="bronze-v1",
                validation_contract={"strategy": outputs["validation_plan"].strategy.value},
                transformation_version="silver-v1",
                split_configuration=outputs["validation_plan"].model_dump(mode="json"),
                model_specification=outputs["resolved_run_config"].request.models,
                evaluation_specification=outputs["resolved_run_config"].request.evaluation,
                code_version=self.code_version,
                dataset_specification=outputs["resolved_run_config"].request.dataset,
            )
        )
        with self.locks.lock(identity.run_id):
            return self._execute(
                request,
                input_frame,
                default_model_registry(),
                driver,
                outputs,
                identity,
            )

    def run(self, request: RunRequest, input_frame: pl.DataFrame) -> MatrixRunResult:
        if not request.models:
            raise ValueError("At least one model is required for matrix execution")
        registry = default_model_registry()
        driver = build_driver()
        inputs = {"run_request": request, "input_frame": input_frame}
        preflight = driver.execute(["resolved_run_config", "validation_plan"], inputs=inputs)
        self._apply_code_version(preflight)
        resolved = preflight["resolved_run_config"]
        plan = preflight["validation_plan"]
        identity = RunIdentity(
            **derive_identity(
                source_checksum=content_hash(input_frame.to_dict(as_series=False)),
                ingestion_contract_version="bronze-v1",
                validation_contract={"strategy": plan.strategy.value},
                transformation_version="silver-v1",
                split_configuration=plan.model_dump(mode="json"),
                model_specification=resolved.request.models,
                evaluation_specification=resolved.request.evaluation,
                code_version=self.code_version,
                dataset_specification=resolved.request.dataset,
            )
        )
        return self._execute(request, input_frame, registry, driver, preflight, identity)

    def _apply_code_version(self, preflight: dict[str, Any]) -> None:
        """Keep persisted resolved configuration aligned with run identity."""

        preflight["resolved_run_config"] = preflight["resolved_run_config"].model_copy(
            update={"code_version": self.code_version}
        )

    def _execute(
        self,
        request: RunRequest,
        input_frame: pl.DataFrame,
        registry: Any,
        driver: Any,
        preflight: dict[str, Any],
        identity: RunIdentity,
    ) -> MatrixRunResult:
        """Execute from an already-resolved preflight without rebuilding the DAG."""

        inputs = {"run_request": request, "input_frame": input_frame}
        resolved = preflight["resolved_run_config"]
        plan = preflight["validation_plan"]
        rolling_origin = plan.strategy.value == "rolling_origin"
        if rolling_origin:
            from .dataflows.modeling import gold_dataset

            outputs = {"gold_dataset": gold_dataset(input_frame, plan)}
            train_frame = None
            test_frame = None
        else:
            outputs = driver.execute(
                [
                    "resolved_run_config",
                    "validation_plan",
                    "gold_dataset",
                    "train_dataset",
                    "test_dataset",
                ],
                inputs=inputs,
            )
            train_frame = outputs["train_dataset"]
            test_frame = outputs["test_dataset"]
        target = request.dataset.get("target")
        if not isinstance(target, str) or not target:
            raise ValueError("dataset.target is required for matrix execution")

        run_root = f"04_platinum/runs/{identity.run_id}"
        manifest_ref = ArtifactRef(
            artifact_type=ArtifactType.MANIFEST,
            path=f"{run_root}/manifest.json",
            schema_version="v1",
        )
        if self.store.exists(manifest_ref):
            if request.reuse_policy == "reuse_verified":
                existing = self._reuse_verified(manifest_ref, identity, request)
                if existing is not None:
                    return existing
            elif request.reuse_policy == "fail_if_exists":
                raise FileExistsError(f"Run {identity.run_id} already exists")
        self.store.clear_staging(identity.run_id)
        store = self.store.staging(identity.run_id)
        model_metrics: dict[str, dict[str, float]] = {}
        errors: list[dict[str, str]] = []
        model_records: dict[str, dict[str, object]] = {}
        materialization_verified: list[bool] = []
        artifacts = list(
            materialize_dataset_lineage(
                store,
                request,
                identity,
                input_frame,
                outputs["gold_dataset"],
                plan,
            )
        )
        materialization_verified.extend(
            store.verify(ref).status == VerificationStatus.PASSED for ref in artifacts
        )
        config_ref = ArtifactRef(
            artifact_type=ArtifactType.RESOLVED_CONFIG,
            path=f"{run_root}/resolved-config.json",
            schema_version="v1",
        )
        config_result = store.write_json(config_ref, resolved.model_dump(mode="json"))
        artifacts.append(config_result.ref)
        materialization_verified.append(config_result.verified)
        for spec in resolved.request.models:
            model_id = str(spec["id"])
            provider_name = str(spec["provider"])
            parameters: dict[str, object] = {}
            try:
                provider = registry.get(provider_name)
                parameters = dict(spec.get("parameters", {}))
                model_config = ModelBuildConfig(
                    model_name=provider_name,
                    target_column=target,
                    horizon=int(cast(int, parameters.get("horizon", 3))),
                    n_estimators=int(cast(int, parameters.get("n_estimators", 30))),
                    random_state=int(cast(int, parameters.get("random_state", 42))),
                    tuned_params=parameters,
                )
                if rolling_origin:
                    fold_metrics: dict[str, list[float]] = {}
                    fold_predictions: list[pl.DataFrame] = []
                    train_time_sec = 0.0
                    evaluation_row_count = 0
                    for fold, fold_train, fold_test in _rolling_fold_frames(input_frame, plan):
                        model = provider.build(model_config)
                        model.fit(_model_frame(fold_train), target)
                        prediction = model.predict(_model_frame(fold_test))
                        metrics = evaluate_metrics(
                            prediction, fold_test[target], request.evaluation
                        )
                        for metric_name, value in metrics.items():
                            fold_metrics.setdefault(metric_name, []).append(value)
                        prediction_frame = prediction.interval(0.9)
                        if isinstance(prediction_frame, pl.Series):
                            prediction_frame = prediction_frame.to_frame()
                        fold_predictions.append(
                            prediction_frame.with_columns(pl.lit(fold).alias("_fold"))
                        )
                        train_time_sec += float(getattr(model, "train_time", 0.0))
                        evaluation_row_count += len(prediction_frame)
                    metrics = {
                        metric_name: float(sum(values) / len(values))
                        for metric_name, values in fold_metrics.items()
                    }
                    prediction_frame = pl.concat(fold_predictions, how="vertical")
                else:
                    assert train_frame is not None and test_frame is not None
                    model = provider.build(model_config)
                    model.fit(_model_frame(train_frame), target)
                    prediction = model.predict(_model_frame(test_frame))
                    metrics = evaluate_metrics(prediction, test_frame[target], request.evaluation)
                    prediction_frame = prediction.interval(0.9)
                    if isinstance(prediction_frame, pl.Series):
                        prediction_frame = prediction_frame.to_frame()
                    train_time_sec = float(getattr(model, "train_time", 0.0))
                    evaluation_row_count = len(prediction_frame)
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise ValueError("model produced non-finite metrics")
                model_metrics[model_id] = metrics
                model_artifact_ref = None
                if not rolling_origin:
                    model_ref = ArtifactRef(
                        artifact_type=ArtifactType.MODEL,
                        path=f"{run_root}/models/{model_id}.uf",
                        schema_version="v1",
                    )
                    model_result = _persist_fitted_model(store, model_ref, model)
                    if model_result is not None:
                        artifacts.append(model_result.ref)
                        materialization_verified.append(model_result.verified)
                        model_artifact_ref = model_result.ref
                prediction_ref = ArtifactRef(
                    artifact_type=ArtifactType.PREDICTIONS,
                    path=f"{run_root}/predictions/{model_id}.parquet",
                    schema_version="v1",
                )
                prediction_result = store.write_table(prediction_ref, prediction_frame)
                artifacts.append(prediction_result.ref)
                materialization_verified.append(prediction_result.verified)
                model_records[model_id] = {
                    "provider": provider_name,
                    "required": bool(spec.get("required", True)),
                    "parameters": parameters,
                    "status": ModelExecutionStatus.SUCCESS,
                    "train_time_sec": train_time_sec,
                    "evaluation_row_count": evaluation_row_count,
                    "metrics": metrics,
                    "model_ref": model_artifact_ref,
                    "prediction_ref": prediction_result.ref,
                }
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                if bool(spec.get("required", True)) or request.fail_fast:
                    raise
                errors.append({"model": model_id, "provider": provider_name, "error": str(error)})
                model_records[model_id] = {
                    "provider": provider_name,
                    "required": False,
                    "parameters": parameters,
                    "status": ModelExecutionStatus.DEGRADED,
                    "error": str(error),
                }

        preliminary_model_results = tuple(
            ModelExecutionResult(
                model_id=model_id,
                provider=str(record["provider"]),
                status=cast(ModelExecutionStatus, record["status"]),
                required=bool(record["required"]),
                resolved_parameters=cast(dict[str, object], record["parameters"]),
                train_time_sec=float(cast(float, record.get("train_time_sec", 0.0))),
                evaluation_row_count=int(cast(int, record.get("evaluation_row_count", 0))),
                metrics={
                    str(key): float(cast(float, value))
                    for key, value in cast(dict[str, object], record.get("metrics", {})).items()
                },
                error=str(record["error"]) if "error" in record else None,
                degradation_reason=(
                    str(record["error"])
                    if record["status"] == ModelExecutionStatus.DEGRADED
                    else None
                ),
                prediction_artifact_ref=cast(ArtifactRef | None, record.get("prediction_ref")),
                model_artifact_ref=cast(ArtifactRef | None, record.get("model_ref")),
            )
            for model_id, record in model_records.items()
        )
        metrics_ref = ArtifactRef(
            artifact_type=ArtifactType.METRICS,
            path=f"{run_root}/metrics.json",
            schema_version="v1",
        )
        metrics_result = store.write_json(
            metrics_ref,
            {
                "models": model_metrics,
                "errors": errors,
                "model_results": [
                    result.model_dump(mode="json") for result in preliminary_model_results
                ],
            },
        )
        artifacts.append(metrics_result.ref)
        materialization_verified.append(metrics_result.verified)
        checks = (
            VerificationCheck(
                check_id="splits.leakage",
                status=(
                    VerificationStatus.PASSED
                    if plan.leakage_check.passed
                    else VerificationStatus.FAILED
                ),
                severity=VerificationSeverity.ERROR,
                target="validation_plan",
                failure_message=None if plan.leakage_check.passed else "Split leakage detected",
            ),
            VerificationCheck(
                check_id="models.completed",
                status=(VerificationStatus.PASSED if model_metrics else VerificationStatus.FAILED),
                severity=VerificationSeverity.ERROR,
                target=metrics_ref.path,
                evidence={"completed": str(len(model_metrics)), "failed": str(len(errors))},
                failure_message=None if model_metrics else "No model branch completed",
            ),
            VerificationCheck(
                check_id="artifacts.materialized",
                status=(
                    VerificationStatus.PASSED
                    if all(materialization_verified)
                    else VerificationStatus.FAILED
                ),
                severity=VerificationSeverity.ERROR,
                target=run_root,
                evidence={
                    "verified": str(sum(materialization_verified)),
                    "total": str(len(materialization_verified)),
                },
                failure_message=(
                    None
                    if all(materialization_verified)
                    else "Artifact materialization verification failed"
                ),
            ),
        )
        verification = RunVerificationReport(
            run_id=identity.run_id,
            checks=checks,
            passed=all(check.status == VerificationStatus.PASSED for check in checks),
        )
        verification_ref = ArtifactRef(
            artifact_type=ArtifactType.VERIFICATION,
            path=f"{run_root}/verification.json",
            schema_version="v1",
        )
        verification_result = store.write_json(
            verification_ref, verification.model_dump(mode="json")
        )
        artifacts.append(verification_result.ref)
        publication_verified = verification.passed and verification_result.verified
        if not publication_verified:
            raise RuntimeError(f"Run {identity.run_id} failed publication verification")
        degraded = bool(errors) and bool(model_metrics)
        manifest = RunManifest(
            identity=identity,
            dataset_id=str(resolved.request.dataset["id"]),
            dataset_domain=str(resolved.request.dataset.get("domain", "unknown")),
            status=(
                RunStatus.SUCCESS
                if publication_verified and not degraded
                else RunStatus.DEGRADED
                if publication_verified
                else RunStatus.FAILED
            ),
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            resolved_config_hash=resolved.config_hash,
            artifacts=tuple(artifact.path for artifact in artifacts),
            artifact_refs=tuple(artifacts),
            degradation_reasons=tuple(
                DegradationReason(
                    node=f"model:{error['model']}",
                    exception_category="ModelBranchFailure",
                    message=error["error"],
                    evidence_impact="That model has no published evidence",
                    remediation="Fix the model adapter or remove it from the request",
                )
                for error in errors
            ),
            verification_passed=publication_verified,
        )
        secret = publication_secret(request.publication)
        if secret is not None:
            manifest = manifest.model_copy(
                update={"manifest_signature": sign_manifest(manifest, secret)}
            )
        manifest_result = store.write_json(manifest_ref, manifest.model_dump(mode="json"))
        if not manifest_result.verified:
            raise RuntimeError(f"Run {identity.run_id} manifest failed verification")
        artifacts.append(manifest_result.ref)
        self.store.promote(store, identity.run_id)
        model_results = tuple(
            result.model_copy(update={"metric_artifact_ref": metrics_result.ref})
            for result in preliminary_model_results
        )
        return PipelineRunResult(
            manifest=manifest,
            verification=verification,
            artifacts=tuple(artifacts),
            model_results=model_results,
        )

    def _reuse_verified(
        self, manifest_ref: ArtifactRef, identity: RunIdentity, request: RunRequest
    ) -> PipelineRunResult | None:
        manifest = RunManifest.model_validate(self.store.read_json(manifest_ref))
        secret = publication_secret(request.publication)
        if (
            manifest.identity != identity
            or not manifest.verification_passed
            or not manifest_authenticity_valid(manifest, secret)
            or not manifest.artifact_refs
            or any(
                self.store.verify(ref).status != VerificationStatus.PASSED
                for ref in manifest.artifact_refs
            )
        ):
            return None
        verification_ref = next(
            (
                ref
                for ref in manifest.artifact_refs
                if ref.artifact_type == ArtifactType.VERIFICATION
            ),
            None,
        )
        metrics_ref = next(
            (ref for ref in manifest.artifact_refs if ref.artifact_type == ArtifactType.METRICS),
            None,
        )
        if verification_ref is None or metrics_ref is None:
            return None
        verification = RunVerificationReport.model_validate(self.store.read_json(verification_ref))
        if not verification.passed:
            return None
        payload = self.store.read_json(metrics_ref)
        serialized_results = payload.get("model_results")
        if not isinstance(serialized_results, list):
            return None
        model_results = tuple(
            ModelExecutionResult.model_validate(serialized).model_copy(
                update={"metric_artifact_ref": metrics_ref}
            )
            for serialized in serialized_results
        )
        return PipelineRunResult(
            manifest=manifest,
            verification=verification,
            artifacts=manifest.artifact_refs,
            model_results=model_results,
            reused=True,
        )
