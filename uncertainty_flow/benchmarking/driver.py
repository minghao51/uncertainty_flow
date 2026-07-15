"""Hamilton driver construction for the benchmarking dataflows."""

from __future__ import annotations

from typing import Any

from .dataflows import diagnostics, modeling, vertical


def build_driver(*, config: dict[str, Any] | None = None) -> Any:
    """Build the first benchmark Hamilton driver.

    Hamilton is intentionally imported here, rather than from core model modules,
    so the ordinary library API remains independent of the benchmarking extra.
    """

    try:
        from hamilton import driver
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "Hamilton support requires the optional benchmarking extra: "
            "uv sync --extra benchmarking"
        ) from error

    return (
        driver.Builder()
        .with_config(config or {})
        .with_modules(vertical, modeling, diagnostics)
        .build()
    )


def available_outputs() -> tuple[str, ...]:
    """Return stable output names for the initial vertical slice."""

    return (
        "resolved_run_config",
        "source_dataset",
        "validation_plan",
        "gold_dataset",
        "fitted_model",
        "distribution_predictions",
        "metric_results",
        "diagnostic_results",
    )
