"""Private helpers for model persistence."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import platform
import warnings
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .types import DEFAULT_QUANTILES

MODEL_PAYLOAD_NAME = "model.pkl"
METADATA_PAYLOAD_NAME = "metadata.json"
MAX_ARCHIVE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
SUPPORTED_FORMAT_VERSIONS = {1}
DEPENDENCY_NAMES = {
    "polars": "polars",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
}


def _sha256_hex(payload: bytes) -> str:
    """Return SHA-256 hex digest for bytes payload."""
    return hashlib.sha256(payload).hexdigest()


def compute_archive_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest for an archive file."""
    archive_path = Path(path)
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_version(package_name: str) -> str | None:
    """Return installed package version when available."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _library_version() -> str:
    """Return the installed library version when possible."""
    return _safe_version("uncertainty-flow") or "0.1.0"


def _fitted_flag(model: Any) -> bool:
    """Extract a conventional fitted flag from a model."""
    return bool(getattr(model, "_fitted", False))


def _class_path(model_or_cls: Any) -> str:
    """Return a fully qualified class path."""
    cls = model_or_cls if isinstance(model_or_cls, type) else model_or_cls.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _target_names(model: Any) -> list[str] | None:
    """Discover model target names when available."""
    if hasattr(model, "targets"):
        targets = getattr(model, "targets")
        if isinstance(targets, list):
            return [str(target) for target in targets]
        if targets is not None:
            return [str(targets)]

    target_col = getattr(model, "_target_col_", None)
    if target_col:
        return [str(target_col)]

    target = getattr(model, "target", None)
    if target:
        return [str(target)]

    return None


def _quantile_levels(model: Any) -> list[float] | None:
    """Discover configured quantile levels when available."""
    quantile_levels = getattr(model, "quantile_levels", None)
    if quantile_levels is not None:
        return [float(level) for level in quantile_levels]

    if hasattr(model, "_quantiles_") or hasattr(model, "_leaf_distributions"):
        return [float(level) for level in DEFAULT_QUANTILES]

    return None


def build_metadata(model: Any, include_metadata: bool = True) -> dict[str, Any]:
    """Build portable metadata for a persisted model."""
    fitted = _fitted_flag(model)
    metadata_dict: dict[str, Any] = {
        "format_version": 1,
        "class_path": _class_path(model),
        "fitted": fitted,
    }

    if not include_metadata:
        return metadata_dict

    metadata_dict.update(
        {
            "library_version": _library_version(),
            "python_version": platform.python_version(),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                name: version
                for name, package in DEPENDENCY_NAMES.items()
                if (version := _safe_version(package)) is not None
            },
        }
    )

    if (targets := _target_names(model)) is not None:
        metadata_dict["target_names"] = targets

    if (levels := _quantile_levels(model)) is not None:
        metadata_dict["quantile_levels"] = levels

    for attr_name in ("horizon", "copula_family", "calibration_size", "tuned_params_"):
        value = getattr(model, attr_name, None)
        if value is not None:
            metadata_dict[attr_name] = value

    return metadata_dict


def save_model_archive(
    model: Any,
    path: str | Path,
    include_metadata: bool = True,
) -> dict[str, Any]:
    """Persist a model and metadata into a .uf archive."""
    archive_path = Path(path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_dict = build_metadata(model, include_metadata=include_metadata)
    if include_metadata:
        metadata_dict["model_payload_sha256"] = _sha256_hex(model_bytes)

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MODEL_PAYLOAD_NAME, model_bytes)
        archive.writestr(METADATA_PAYLOAD_NAME, json.dumps(metadata_dict, indent=2, sort_keys=True))

    return metadata_dict


def _warn_version_mismatches(saved_metadata: dict[str, Any]) -> None:
    """Warn on dependency version mismatches without blocking load."""
    dependency_versions = saved_metadata.get("dependencies", {})
    if not isinstance(dependency_versions, dict):
        return

    for name, saved_version in dependency_versions.items():
        package_name = DEPENDENCY_NAMES.get(name)
        if package_name is None:
            continue
        current_version = _safe_version(package_name)
        if current_version is not None and current_version != saved_version:
            warnings.warn(
                f"Loaded model was saved with {name}=={saved_version}, "
                f"current environment has {name}=={current_version}.",
                UserWarning,
                stacklevel=3,
            )


def load_model_archive(
    path: str | Path,
    *,
    expected_archive_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model and metadata from a .uf archive."""
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Model archive not found: {archive_path}")

    archive_size = archive_path.stat().st_size
    if archive_size > MAX_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            f"Model archive too large ({archive_size:,} bytes). "
            f"Maximum allowed: {MAX_ARCHIVE_SIZE_BYTES:,} bytes."
        )

    if expected_archive_sha256 is not None:
        actual_archive_sha256 = compute_archive_sha256(archive_path)
        if actual_archive_sha256.lower() != expected_archive_sha256.lower():
            raise ValueError(
                "Archive SHA-256 mismatch. "
                f"Expected {expected_archive_sha256.lower()}, got {actual_archive_sha256.lower()}."
            )

    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            names = set(archive.namelist())
            if MODEL_PAYLOAD_NAME not in names:
                raise ValueError(
                    f"Invalid model archive: missing required payload '{MODEL_PAYLOAD_NAME}'."
                )
            if METADATA_PAYLOAD_NAME not in names:
                raise ValueError(
                    f"Invalid model archive: missing required payload '{METADATA_PAYLOAD_NAME}'."
                )

            metadata_payload = archive.read(METADATA_PAYLOAD_NAME)
            model_payload = archive.read(MODEL_PAYLOAD_NAME)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid model archive: {archive_path} is not a valid zip file.") from exc

    try:
        metadata_dict = json.loads(metadata_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid model archive: metadata.json is not valid JSON.") from exc

    format_version = metadata_dict.get("format_version")
    if format_version is None:
        raise ValueError(
            "Invalid model archive: metadata.json missing required 'format_version' field."
        )
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported archive format version: {format_version}. "
            f"Supported versions: {SUPPORTED_FORMAT_VERSIONS}."
        )

    expected_model_sha256 = metadata_dict.get("model_payload_sha256")
    if expected_model_sha256 is not None:
        actual_model_sha256 = _sha256_hex(model_payload)
        if actual_model_sha256 != str(expected_model_sha256).lower():
            raise ValueError(
                "Invalid model archive: model payload checksum mismatch. "
                "The archive may be corrupted or tampered with."
            )

    try:
        model = pickle.load(io.BytesIO(model_payload))
    except (pickle.PickleError, AttributeError, ImportError, EOFError) as exc:
        raise ValueError("Invalid model archive: failed to deserialize model payload.") from exc

    _warn_version_mismatches(metadata_dict)
    setattr(model, "_metadata", metadata_dict)
    return model, metadata_dict
