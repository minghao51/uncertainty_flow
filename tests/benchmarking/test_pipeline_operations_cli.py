"""CLI coverage for Phase 6 cleanup controls."""

from __future__ import annotations

import json

import polars as pl
from click.testing import CliRunner

from uncertainty_flow.cli import _load_pipeline_request, cli


def test_pipeline_gc_defaults_to_safe_dry_run(tmp_path) -> None:
    result = CliRunner().invoke(cli, ["pipeline", "gc", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "Would remove 0 unverified run(s)" in result.output


def test_pipeline_verify_lineage_and_list_runs(tmp_path) -> None:
    data_path = tmp_path / "input.parquet"
    pl.DataFrame({"x": list(range(150)), "y": list(range(150))}).write_parquet(data_path)
    config = tmp_path / "run.yaml"
    config.write_text(
        "\n".join(
            (
                "dataset:",
                "  id: cli-fixture",
                f"  uri: {data_path}",
                "  target: y",
                "validation:",
                "  strategy: random_holdout",
                "  test_size: 0.2",
                "models:",
                "  - id: conformal-regressor",
                "    provider: conformal-regressor",
                "evaluation:",
                "  metrics: [coverage]",
                "storage:",
                f"  root: {tmp_path}",
            )
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    run = runner.invoke(cli, ["pipeline", "run", "--config", str(config)])

    assert run.exit_code == 0, run.output
    run_id = next(
        line.split(": ", 1)[1] for line in run.output.splitlines() if line.startswith("Run:")
    )
    for command, expected in (
        (["pipeline", "verify", run_id, "--root", str(tmp_path)], "verified"),
        (["pipeline", "lineage", run_id, "--root", str(tmp_path)], "01_bronze"),
        (["pipeline", "list-runs", "--root", str(tmp_path)], run_id),
    ):
        result = runner.invoke(cli, command)
        assert result.exit_code == 0, result.output
        assert expected in result.output


def test_pipeline_list_runs_reports_a_corrupt_manifest(tmp_path) -> None:
    manifest = tmp_path / "04_platinum" / "runs" / "broken" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"status": "success"}), encoding="utf-8")

    result = CliRunner().invoke(cli, ["pipeline", "list-runs", "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Invalid manifest" in result.output


def test_pipeline_request_precedence_uses_cli_environment_file_defaults(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "request.yaml"
    config.write_text(
        "storage:\n  root: from-file\nmodels: []\ndataset: {}\nvalidation: {}\nevaluation: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("UNCERTAINTY_FLOW_PIPELINE_STORAGE_ROOT", "from-environment")
    request = _load_pipeline_request(config)
    assert request.storage["root"] == "from-environment"

    request = _load_pipeline_request(config, storage_root="from-cli")
    assert request.storage["root"] == "from-cli"
    assert request.reuse_policy == "reuse_verified"
