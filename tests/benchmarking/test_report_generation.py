"""Tests for the pipeline-native benchmark report reader."""

from __future__ import annotations

import json

from benchmarks.generate_report import create_comparison_table, load_results


def test_report_reader_consumes_pipeline_result_shape(tmp_path) -> None:
    payload = {
        "manifest": {"dataset_id": "weather", "dataset_domain": "time-series"},
        "model_results": [
            {
                "model_id": "conformal-regressor",
                "train_time_sec": 0.25,
                "metrics": {
                    "coverage_90": 0.9,
                    "winkler_90": 1.2,
                    "calibration_error_90": 0.0,
                    "pinball": 0.1,
                },
            }
        ],
    }
    (tmp_path / "weather.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "unrelated.json").write_text('{"unrelated": true}', encoding="utf-8")

    results = load_results(str(tmp_path))
    table = create_comparison_table(results)

    assert len(results) == 1
    assert table.row(0, named=True)["dataset"] == "weather"
    assert table.row(0, named=True)["domain"] == "time-series"
    assert table.row(0, named=True)["model"] == "conformal-regressor"
    assert table.row(0, named=True)["pinball"] == 0.1
