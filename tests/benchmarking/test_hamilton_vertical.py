"""DAG construction and validation tests for the initial Hamilton slice."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from uncertainty_flow.benchmarking.contracts import RunRequest
from uncertainty_flow.benchmarking.driver import available_outputs, build_driver


def _request(test_size: float = 0.2) -> RunRequest:
    return RunRequest(
        dataset={"id": "example", "target": "y"},
        validation={"strategy": "random_holdout", "test_size": test_size, "random_seed": 42},
        models=({"id": "conformal-regressor", "provider": "conformal-regressor"},),
        evaluation={"metrics": ["coverage"]},
        storage={"provider": "local", "root": "data"},
    )


def test_initial_driver_has_stable_outputs() -> None:
    driver = build_driver()

    available = {node.name for node in driver.list_available_variables()}
    assert set(available_outputs()).issubset(available)


def test_validation_plan_executes_without_side_effects(tmp_path) -> None:
    driver = build_driver()
    frame = pl.DataFrame({"id": ["a", "b", "c", "d", "e"], "y": [1, 2, 3, 4, 5]})

    result = driver.execute(
        ["resolved_run_config", "validation_plan"],
        inputs={"run_request": _request(), "input_frame": frame},
    )

    plan = result["validation_plan"]
    assert plan.leakage_check.passed is True
    assert len(plan.assignments) == len(frame)
    assert list(tmp_path.iterdir()) == []


def test_invalid_split_fails_before_any_model_step() -> None:
    driver = build_driver()
    frame = pl.DataFrame({"id": ["a", "b", "c"], "y": [1, 2, 3]})

    with pytest.raises(ValueError, match="test_size"):
        driver.execute(
            ["validation_plan"],
            inputs={"run_request": _request(test_size=1.0), "input_frame": frame},
        )


def test_resolved_config_rejects_empty_model_list() -> None:
    frame = pl.DataFrame({"id": ["a", "b", "c"], "y": [1, 2, 3]})

    with pytest.raises(ValueError, match="At least one model"):
        build_driver().execute(
            ["resolved_run_config"],
            inputs={
                "run_request": _request().model_copy(update={"models": ()}),
                "input_frame": frame,
            },
        )


def test_validation_rejects_null_observation_ids() -> None:
    frame = pl.DataFrame({"id": ["a", None, "c"], "y": [1, 2, 3]})

    with pytest.raises(ValueError, match="cannot contain nulls"):
        build_driver().execute(
            ["validation_plan"],
            inputs={"run_request": _request(), "input_frame": frame},
        )


def test_temporal_and_rolling_plans_preserve_tail_membership() -> None:
    driver = build_driver()
    frame = pl.DataFrame(
        {
            "id": [f"row-{i}" for i in range(10)],
            "timestamp": list(reversed(range(10))),
            "y": list(range(10)),
        }
    )

    request = _request().model_copy(
        update={
            "validation": {
                "strategy": "temporal_holdout",
                "test_size": 0.2,
                "timestamp_column": "timestamp",
            }
        }
    )
    result = driver.execute(
        ["validation_plan"],
        inputs={"run_request": request, "input_frame": frame},
    )
    assignments = result["validation_plan"].assignments
    assert [item.observation_id for item in assignments if item.split == "test"] == [
        "row-0",
        "row-1",
    ]

    rolling_request = request.model_copy(
        update={"validation": {"strategy": "rolling_origin", "test_size": 0.2}}
    )
    rolling_plan = driver.execute(
        ["validation_plan"],
        inputs={"run_request": rolling_request, "input_frame": frame},
    )["validation_plan"]
    assert rolling_plan.strategy.value == "rolling_origin"
    assert {item.fold for item in rolling_plan.assignments} == {0, 1, 2}
    assert all(item.fold is not None for item in rolling_plan.assignments)


def test_temporal_plan_accepts_real_datetime_values() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(days=index) for index in range(20)],
            "y": [float(index) for index in range(20)],
        }
    )
    request = _request().model_copy(
        update={
            "validation": {
                "strategy": "temporal_holdout",
                "test_size": 0.2,
                "timestamp_column": "timestamp",
            }
        }
    )

    outputs = build_driver().execute(
        ["resolved_run_config", "validation_plan"],
        inputs={"run_request": request, "input_frame": frame},
    )

    assert outputs["validation_plan"].strategy.value == "temporal_holdout"


def test_temporal_plan_requires_timestamp_or_explicit_source_order() -> None:
    frame = pl.DataFrame({"y": [float(index) for index in range(20)]})
    temporal = _request().model_copy(
        update={"validation": {"strategy": "temporal_holdout", "test_size": 0.2}}
    )

    with pytest.raises(ValueError, match="timestamp_column or preserve_order"):
        build_driver().execute(
            ["validation_plan"],
            inputs={"run_request": temporal, "input_frame": frame},
        )

    ordered = temporal.model_copy(
        update={
            "validation": {
                "strategy": "temporal_holdout",
                "test_size": 0.2,
                "preserve_order": True,
            }
        }
    )
    plan = build_driver().execute(
        ["validation_plan"],
        inputs={"run_request": ordered, "input_frame": frame},
    )["validation_plan"]

    assert [item.observation_id for item in plan.assignments if item.split == "test"] == [
        "16",
        "17",
        "18",
        "19",
    ]


def test_random_holdout_uses_the_configured_seed() -> None:
    driver = build_driver()
    frame = pl.DataFrame({"id": [f"row-{i}" for i in range(20)], "y": list(range(20))})
    first = driver.execute(
        ["validation_plan"],
        inputs={
            "run_request": _request().model_copy(
                update={
                    "validation": {"strategy": "random_holdout", "test_size": 0.2, "random_seed": 1}
                }
            ),
            "input_frame": frame,
        },
    )["validation_plan"]
    second = driver.execute(
        ["validation_plan"],
        inputs={
            "run_request": _request().model_copy(
                update={
                    "validation": {"strategy": "random_holdout", "test_size": 0.2, "random_seed": 2}
                }
            ),
            "input_frame": frame,
        },
    )["validation_plan"]
    assert first.assignments != second.assignments
