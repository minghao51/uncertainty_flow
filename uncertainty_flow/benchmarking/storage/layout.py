"""Medallion artifact path resolution."""

from __future__ import annotations

from pathlib import Path

from uncertainty_flow.benchmarking.contracts.artifacts import ArtifactType

_LAYER_BY_TYPE = {
    ArtifactType.BRONZE_DATASET: "01_bronze",
    ArtifactType.SILVER_DATASET: "02_silver",
    ArtifactType.SILVER_VALIDATION: "02_silver",
    ArtifactType.GOLD_DATASET: "03_gold",
    ArtifactType.GOLD_SPLITS: "03_gold",
    ArtifactType.MODEL: "04_platinum",
    ArtifactType.PREDICTIONS: "04_platinum",
    ArtifactType.METRICS: "04_platinum",
    ArtifactType.CALIBRATION: "04_platinum",
    ArtifactType.DIAGNOSTICS: "04_platinum",
    ArtifactType.MANIFEST: "04_platinum",
    ArtifactType.VERIFICATION: "04_platinum",
}


def artifact_path(
    root: Path,
    artifact_type: ArtifactType,
    *,
    dataset_id: str | None = None,
    version: str | None = None,
    run_id: str | None = None,
    name: str | None = None,
) -> Path:
    """Resolve a safe path in the medallion layout."""

    layer = _LAYER_BY_TYPE[artifact_type]
    if layer == "04_platinum":
        if not run_id:
            raise ValueError("run_id is required for Platinum artifacts")
        base = root / layer / "runs" / run_id
    else:
        if not dataset_id or not version:
            raise ValueError("dataset_id and version are required for dataset artifacts")
        base = root / layer / dataset_id / version
    return base / (name or f"{artifact_type.value}.json")
