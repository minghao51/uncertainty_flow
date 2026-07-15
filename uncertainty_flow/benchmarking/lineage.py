"""Shared local materialization of Bronze, Silver, and Gold lineage artifacts."""

from __future__ import annotations

import polars as pl

from .contracts import ArtifactRef, ArtifactType, RunIdentity, RunRequest, ValidationPlan
from .storage import LocalArtifactStore


def _silver_frame(source_dataset: pl.DataFrame) -> pl.DataFrame:
    """Normalize the source into a stable row-identity-bearing Silver frame."""

    if "id" not in source_dataset.columns:
        return source_dataset.with_columns(
            pl.Series("id", [str(index) for index in range(len(source_dataset))])
        )
    return source_dataset.with_columns(pl.col("id").cast(pl.String))


def _schema(frame: pl.DataFrame) -> dict[str, str]:
    """Return a stable, human-readable schema contract for lineage metadata."""

    return {name: str(dtype) for name, dtype in frame.schema.items()}


def materialize_dataset_lineage(
    store: LocalArtifactStore,
    request: RunRequest,
    identity: RunIdentity,
    source_dataset: pl.DataFrame,
    gold_dataset: pl.DataFrame,
    validation_plan: ValidationPlan,
) -> tuple[ArtifactRef, ...]:
    """Write immutable local dataset layers and persisted split membership."""

    dataset_id = str(request.dataset["id"])
    bronze_root = f"01_bronze/{dataset_id}/{identity.dataset_version}"
    silver_root = f"02_silver/{dataset_id}/{identity.silver_version}"
    gold_root = f"03_gold/{dataset_id}/{identity.validation_plan_id}"
    artifacts: list[ArtifactRef] = []
    silver_dataset = _silver_frame(source_dataset)

    for artifact_type, path, frame in (
        (ArtifactType.BRONZE_DATASET, f"{bronze_root}/observations.parquet", source_dataset),
        (ArtifactType.SILVER_DATASET, f"{silver_root}/observations.parquet", silver_dataset),
        (ArtifactType.GOLD_DATASET, f"{gold_root}/observations.parquet", gold_dataset),
    ):
        ref = ArtifactRef(artifact_type=artifact_type, path=path, schema_version="v1")
        artifacts.append(store.write_table(ref, frame).ref)

    json_artifacts = (
        (
            ArtifactRef(
                artifact_type=ArtifactType.BRONZE_DATASET,
                path=f"{bronze_root}/manifest.json",
                schema_version="v1",
            ),
            {
                "dataset_id": dataset_id,
                "dataset_version": identity.dataset_version,
                "rows": len(source_dataset),
                "contract": "source_faithful_ingestion",
                "schema": _schema(source_dataset),
            },
        ),
        (
            ArtifactRef(
                artifact_type=ArtifactType.SILVER_VALIDATION,
                path=f"{silver_root}/validation.json",
                schema_version="v1",
            ),
            {
                "silver_version": identity.silver_version,
                "rows": len(silver_dataset),
                "contract": "normalized_typed_observations",
                "normalization": {
                    "row_identity": "existing id cast to string or generated from source position",
                    "source_schema": _schema(source_dataset),
                    "silver_schema": _schema(silver_dataset),
                },
            },
        ),
        (
            ArtifactRef(
                artifact_type=ArtifactType.GOLD_SPLITS,
                path=f"{gold_root}/splits.json",
                schema_version="v1",
            ),
            {
                **validation_plan.model_dump(mode="json"),
                "contract": "model_ready_validation_membership",
                "fold_count": len(
                    {assignment.fold for assignment in validation_plan.assignments} - {None}
                ),
            },
        ),
    )
    for ref, payload in json_artifacts:
        artifacts.append(store.write_json(ref, payload).ref)
    return tuple(artifacts)
