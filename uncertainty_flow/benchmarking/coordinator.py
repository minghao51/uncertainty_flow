"""Operational coordinator for the first verified Hamilton benchmark run."""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from .driver import build_driver
from .identity import content_hash, derive_identity
from .lineage import materialize_dataset_lineage
from .operations import (
    RunLockManager,
    manifest_authenticity_valid,
    publication_secret,
    sign_manifest,
)
from .storage import LocalArtifactStore


class BenchmarkCoordinator:
    """Apply run policy around the Hamilton dataflow."""

    def __init__(self, storage_root: str = "data", code_version: str = "uncertainty-flow-0.5.0"):
        self.store = LocalArtifactStore(storage_root)
        self.code_version = code_version
        self.locks = RunLockManager(storage_root)

    def run_with_lock(self, request: RunRequest, input_frame: pl.DataFrame) -> PipelineRunResult:
        """Execute with a local run lock and stale-lock recovery policy."""

        source_checksum = content_hash(input_frame.to_dict(as_series=False))
        driver = build_driver()
        preflight = driver.execute(
            ["resolved_run_config", "validation_plan"],
            inputs={"run_request": request, "input_frame": input_frame},
        )
        self._apply_code_version(preflight)
        identity = self._derive_identity(
            preflight["resolved_run_config"].request,
            source_checksum,
            preflight["validation_plan"],
        )
        with self.locks.lock(identity.run_id):
            return self._execute(request, input_frame, driver, preflight, identity)

    def run(self, request: RunRequest, input_frame: pl.DataFrame) -> PipelineRunResult:
        """Execute, verify, and publish the initial local Platinum branch."""

        source_checksum = content_hash(input_frame.to_dict(as_series=False))
        driver = build_driver()
        preflight = driver.execute(
            ["resolved_run_config", "validation_plan"],
            inputs={"run_request": request, "input_frame": input_frame},
        )
        self._apply_code_version(preflight)
        resolved = preflight["resolved_run_config"]
        plan = preflight["validation_plan"]
        identity = self._derive_identity(resolved.request, source_checksum, plan)
        return self._execute(request, input_frame, driver, preflight, identity)

    def _apply_code_version(self, preflight: dict[str, Any]) -> None:
        """Keep persisted resolved configuration aligned with run identity."""

        preflight["resolved_run_config"] = preflight["resolved_run_config"].model_copy(
            update={"code_version": self.code_version}
        )

    def _execute(
        self,
        request: RunRequest,
        input_frame: pl.DataFrame,
        driver: Any,
        preflight: dict[str, Any],
        identity: RunIdentity,
    ) -> PipelineRunResult:
        """Execute from an already-resolved preflight without rebuilding the DAG."""

        resolved = preflight["resolved_run_config"]
        plan = preflight["validation_plan"]
        if plan.strategy.value == "rolling_origin":
            if len(request.models) != 1:
                raise ValueError("BenchmarkCoordinator rolling-origin execution requires one model")
            from .matrix import ModelMatrixCoordinator

            return ModelMatrixCoordinator(
                storage_root=str(self.store.root), code_version=self.code_version
            ).run(request, input_frame)
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

        outputs = driver.execute(
            [
                "resolved_run_config",
                "validation_plan",
                "gold_dataset",
                "test_dataset",
                "fitted_model",
                "distribution_predictions",
                "metric_results",
                "diagnostic_results",
            ],
            inputs={"run_request": request, "input_frame": input_frame},
        )
        metrics = outputs["metric_results"]
        predictions = outputs["distribution_predictions"]
        fitted_model = outputs["fitted_model"]
        diagnostic_results = outputs["diagnostic_results"]

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
        materialization_verified = [
            store.verify(ref).status == VerificationStatus.PASSED for ref in artifacts
        ]

        config_ref = ArtifactRef(
            artifact_type=ArtifactType.RESOLVED_CONFIG,
            path=f"{run_root}/resolved-config.json",
            schema_version="v1",
        )
        config_result = store.write_json(config_ref, resolved.model_dump(mode="json"))
        artifacts.append(config_result.ref)
        materialization_verified.append(config_result.verified)

        prediction_ref = ArtifactRef(
            artifact_type=ArtifactType.PREDICTIONS,
            path=f"{run_root}/predictions.parquet",
            schema_version="v1",
        )
        prediction_frame = predictions.interval(0.9)
        if isinstance(prediction_frame, pl.Series):
            prediction_frame = prediction_frame.to_frame()
        prediction_result = store.write_table(prediction_ref, prediction_frame)
        artifacts.append(prediction_result.ref)
        materialization_verified.append(prediction_result.verified)

        model_spec = resolved.request.models[0]
        model_id = str(model_spec.get("id", model_spec.get("provider", "model")))
        provider = str(model_spec.get("provider", model_id))
        model_artifact_ref = None
        model = getattr(fitted_model, "model", None)
        save = getattr(model, "save", None)
        if callable(save):
            model_ref = ArtifactRef(
                artifact_type=ArtifactType.MODEL,
                path=f"{run_root}/models/{model_id}.uf",
                schema_version="v1",
            )
            with tempfile.TemporaryDirectory() as directory:
                archive_path = Path(directory) / "model.uf"
                save(archive_path)
                model_artifact_result = store.write_bytes(model_ref, archive_path.read_bytes())
            artifacts.append(model_artifact_result.ref)
            materialization_verified.append(model_artifact_result.verified)
            model_artifact_ref = model_artifact_result.ref

        prediction_row_count = len(prediction_frame)
        preliminary_model_result = ModelExecutionResult(
            model_id=model_id,
            provider=provider,
            status=ModelExecutionStatus.SUCCESS,
            required=bool(model_spec.get("required", True)),
            resolved_parameters=dict(model_spec.get("parameters", {})),
            train_time_sec=float(getattr(fitted_model, "train_time", 0.0)),
            evaluation_row_count=prediction_row_count,
            metrics={str(key): float(value) for key, value in metrics.items()},
            model_artifact_ref=model_artifact_ref,
            prediction_artifact_ref=prediction_result.ref,
        )
        metrics_ref = ArtifactRef(
            artifact_type=ArtifactType.METRICS,
            path=f"{run_root}/metrics.json",
            schema_version="v1",
        )
        metrics_result = store.write_json(
            metrics_ref,
            {
                "models": {model_id: preliminary_model_result.metrics},
                "model_results": [preliminary_model_result.model_dump(mode="json")],
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
                evidence={"checked_rows": str(plan.leakage_check.checked_rows)},
                failure_message=None if plan.leakage_check.passed else "Split leakage detected",
            ),
            VerificationCheck(
                check_id="metrics.finite",
                status=(
                    VerificationStatus.PASSED
                    if all(math.isfinite(value) for value in metrics.values())
                    else VerificationStatus.FAILED
                ),
                severity=VerificationSeverity.ERROR,
                target=metrics_result.ref.path,
                failure_message="Metric output contains a non-finite value",
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
            passed=all(check.status == VerificationStatus.PASSED for check in checks)
            and all(materialization_verified),
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

        manifest = RunManifest(
            identity=identity,
            dataset_id=str(resolved.request.dataset["id"]),
            dataset_domain=str(resolved.request.dataset.get("domain", "unknown")),
            status=(
                RunStatus.FAILED
                if not publication_verified
                else RunStatus.DEGRADED
                if any(value.startswith("degraded:") for value in diagnostic_results.values())
                else RunStatus.SUCCESS
            ),
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            resolved_config_hash=resolved.config_hash,
            artifacts=tuple(artifact.path for artifact in artifacts),
            artifact_refs=tuple(artifacts),
            degradation_reasons=tuple(
                DegradationReason(
                    node=name,
                    exception_category="OptionalDiagnosticUnavailable",
                    message=value,
                    evidence_impact="Diagnostic evidence is incomplete",
                    remediation="Install or register the diagnostic adapter",
                )
                for name, value in diagnostic_results.items()
                if value.startswith("degraded:")
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
        self.store.promote(store, identity.run_id)
        artifacts.append(manifest_result.ref)
        model_result = preliminary_model_result.model_copy(
            update={"metric_artifact_ref": metrics_result.ref}
        )
        return PipelineRunResult(
            manifest=manifest,
            verification=verification,
            artifacts=tuple(artifacts),
            model_results=(model_result,),
        )

    def _derive_identity(self, request, source_checksum: str, plan) -> RunIdentity:
        identities = derive_identity(
            source_checksum=source_checksum,
            ingestion_contract_version="bronze-v1",
            validation_contract={"strategy": plan.strategy.value},
            transformation_version="silver-v1",
            split_configuration=plan.model_dump(mode="json"),
            model_specification=request.models,
            evaluation_specification=request.evaluation,
            code_version=self.code_version,
            dataset_specification=request.dataset,
        )
        return RunIdentity(**identities)

    def _reuse_verified(
        self, manifest_ref: ArtifactRef, identity: RunIdentity, request: RunRequest
    ) -> PipelineRunResult | None:
        manifest = RunManifest.model_validate(self.store.read_json(manifest_ref))
        secret = publication_secret(request.publication)
        if (
            manifest.identity != identity
            or not manifest.verification_passed
            or not manifest_authenticity_valid(manifest, secret)
        ):
            return None
        if not manifest.artifact_refs or any(
            self.store.verify(ref).status != VerificationStatus.PASSED
            for ref in manifest.artifact_refs
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
        if (
            verification_ref is None
            or metrics_ref is None
            or not self.store.exists(verification_ref)
            or not self.store.exists(metrics_ref)
        ):
            return None
        verification = RunVerificationReport.model_validate(self.store.read_json(verification_ref))
        if not verification.passed:
            return None
        payload = self.store.read_json(metrics_ref)
        serialized_results = payload.get("model_results")
        if not isinstance(serialized_results, list) or len(serialized_results) != 1:
            return None
        model_result = ModelExecutionResult.model_validate(serialized_results[0]).model_copy(
            update={"metric_artifact_ref": metrics_ref}
        )
        return PipelineRunResult(
            manifest=manifest,
            verification=verification,
            artifacts=manifest.artifact_refs,
            model_results=(model_result,),
            reused=True,
        )
