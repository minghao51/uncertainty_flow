"""CLI tests for the first Hamilton pipeline commands."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from click.testing import CliRunner

from uncertainty_flow.cli import cli


def _config(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "input.parquet"
    pl.DataFrame(
        {
            "feature": [float(i % 5) for i in range(150)],
            "y": [float(i) + 0.25 for i in range(150)],
        }
    ).write_parquet(dataset_path)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        f"""
pipeline:
  mode: benchmark
dataset:
  id: fixture
  uri: {dataset_path}
  target: y
validation:
  strategy: random_holdout
  test_size: 0.2
models:
  - id: conformal-regressor
    provider: conformal-regressor
evaluation:
  metrics: [coverage, winkler, pinball]
storage:
  provider: local
  root: {tmp_path}
""",
        encoding="utf-8",
    )
    return config_path


def test_pipeline_plan_has_no_side_effects(tmp_path) -> None:
    config = _config(tmp_path)
    result = CliRunner().invoke(cli, ["pipeline", "plan", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "Pipeline plan valid" in result.output
    assert "Side effects: none" in result.output
    assert not list(tmp_path.glob("04_platinum"))


def test_pipeline_plan_rejects_semantically_invalid_metrics(tmp_path) -> None:
    config = _config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "metrics: [coverage, winkler, pinball]", "metrics: []"
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["pipeline", "plan", "--config", str(config)])

    assert result.exit_code == 1
    assert "non-empty list" in result.output


def test_pipeline_run_publishes_verified_run(tmp_path) -> None:
    config = _config(tmp_path)
    result = CliRunner().invoke(cli, ["pipeline", "run", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "Status: success" in result.output
    assert "Verified: True" in result.output


def test_pipeline_export_site_exports_verified_evidence(tmp_path) -> None:
    config = _config(tmp_path)
    run_result = CliRunner().invoke(cli, ["pipeline", "run", "--config", str(config)])
    output = tmp_path / "evidence"
    export_result = CliRunner().invoke(
        cli,
        ["pipeline", "export-site", "--root", str(tmp_path), "--output", str(output)],
    )

    assert run_result.exit_code == 0, run_result.output
    assert export_result.exit_code == 0, export_result.output
    assert (output / "index.json").exists()
