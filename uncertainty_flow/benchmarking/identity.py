"""Canonical serialization and deterministic identity helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"__non_finite_float__": str(value)}
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes_hex__": bytes(value).hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a contract or mapping deterministically."""

    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def derive_identity(
    *,
    source_checksum: str,
    ingestion_contract_version: str,
    validation_contract: Any,
    transformation_version: str,
    split_configuration: Any,
    model_specification: Any,
    evaluation_specification: Any,
    code_version: str,
    dataset_specification: Any = None,
) -> dict[str, str]:
    """Derive stable upstream and run identities from non-volatile inputs."""

    dataset_version = content_hash(
        {
            "source_checksum": source_checksum,
            "contract_version": ingestion_contract_version,
            "dataset": dataset_specification,
        }
    )
    silver_version = content_hash(
        {
            "dataset_version": dataset_version,
            "validation_contract": validation_contract,
            "transformation_version": transformation_version,
        }
    )
    validation_plan_id = content_hash(
        {"silver_version": silver_version, "split_configuration": split_configuration}
    )
    run_id = content_hash(
        {
            "validation_plan_id": validation_plan_id,
            "model_specification": model_specification,
            "evaluation_specification": evaluation_specification,
            "code_version": code_version,
        }
    )
    return {
        "dataset_version": dataset_version,
        "silver_version": silver_version,
        "validation_plan_id": validation_plan_id,
        "run_id": run_id,
    }
