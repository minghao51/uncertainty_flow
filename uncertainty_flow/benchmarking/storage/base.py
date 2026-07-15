"""Artifact store protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from ..contracts.artifacts import ArtifactRef, MaterializationResult
from ..contracts.verification import VerificationCheck


class ArtifactStore(Protocol):
    """Persistence boundary used by pipeline nodes and the coordinator."""

    root: Path

    def exists(self, ref: ArtifactRef) -> bool: ...

    def read_json(self, ref: ArtifactRef) -> Mapping[str, Any]: ...

    def write_json(self, ref: ArtifactRef, value: Mapping[str, Any]) -> MaterializationResult: ...

    def write_bytes(self, ref: ArtifactRef, value: bytes) -> MaterializationResult: ...

    def read_table(self, ref: ArtifactRef) -> pl.DataFrame: ...

    def write_table(self, ref: ArtifactRef, value: pl.DataFrame) -> MaterializationResult: ...

    def verify(self, ref: ArtifactRef) -> VerificationCheck: ...
