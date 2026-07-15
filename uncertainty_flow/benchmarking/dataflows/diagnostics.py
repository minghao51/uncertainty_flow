"""Optional diagnostic branch with explicit degraded-run semantics."""

from __future__ import annotations

from uncertainty_flow.benchmarking.contracts.runs import ResolvedRunConfig


def diagnostic_results(resolved_run_config: ResolvedRunConfig) -> dict[str, str]:
    """Report optional diagnostics without hiding unavailable implementations."""

    requested = resolved_run_config.request.evaluation.get("diagnostics", {})
    if not isinstance(requested, dict):
        raise ValueError("evaluation.diagnostics must be a mapping")
    results: dict[str, str] = {}
    for name, policy in requested.items():
        if policy == "disabled":
            results[str(name)] = "disabled"
        elif policy == "optional":
            results[str(name)] = "degraded: diagnostic adapter unavailable"
        else:
            raise ValueError(f"Unknown diagnostic policy {policy!r} for {name!r}")
    return results
