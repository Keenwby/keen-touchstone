"""Artifact serialization must satisfy the canonical JSON Schemas."""

import jsonschema
import numpy as np
import pytest

from keen_touchstone.aggregate import build_suite_result
from keen_touchstone.artifacts import SUITE_TASK_KEY, ReliabilityAggregate, load_schema
from keen_touchstone.stats import TaskTrials


def _tasks(n_tasks: int = 8, n: int = 10, seed: int = 5) -> list[TaskTrials]:
    rng = np.random.default_rng(seed)
    return [
        TaskTrials(
            task_key=f"task-{i}",
            n=n,
            c=int(rng.binomial(n, rng.uniform(0.4, 0.95))),
            tokens=tuple(int(x) for x in rng.integers(200, 900, size=n)),
        )
        for i in range(n_tasks)
    ]


def test_suite_and_tasks_validate_against_schema() -> None:
    result = build_suite_result(
        _tasks(), context="offline", model="mockllm/model", agent_config_hash="abc123"
    )
    schema = load_schema("reliability-aggregate")
    for agg in [result.suite, *result.tasks]:
        jsonschema.validate(agg.to_schema_dict(), schema)  # raises on violation


def test_suite_headline_matches_last_curve_point() -> None:
    result = build_suite_result(
        _tasks(), context="offline", model="m", agent_config_hash="h", seed=7
    )
    suite = result.suite
    last = suite.reliability_decay_curve[-1]
    assert suite.headline_k == last.k == 10  # min(n_i)
    assert suite.pass_hat_k == pytest.approx(last.pass_hat_k)
    assert suite.pass_hat_k_ci_low == pytest.approx(last.ci_low)
    assert suite.pass_hat_k_ci_high == pytest.approx(last.ci_high)
    assert suite.task_key == SUITE_TASK_KEY
    assert suite.n_rollouts == 80
    assert suite.token_mean is not None


def test_task_key_source_omitted_when_none() -> None:
    result = build_suite_result(
        _tasks(2), context="online", model="m", agent_config_hash="h"
    )
    data = result.suite.to_schema_dict()
    assert "task_key_source" not in data
    tagged = build_suite_result(
        _tasks(2), context="online", model="m", agent_config_hash="h",
        task_key_source="declared_tag",
    )
    assert tagged.suite.to_schema_dict()["task_key_source"] == "declared_tag"


def test_schema_rejects_extra_fields() -> None:
    data = build_suite_result(
        _tasks(2), context="offline", model="m", agent_config_hash="h"
    ).suite.to_schema_dict()
    data["sneaky_extra"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, load_schema("reliability-aggregate"))


def test_warnings_for_small_suites_and_single_trials() -> None:
    small = build_suite_result(
        _tasks(2), context="offline", model="m", agent_config_hash="h"
    )
    assert any("between-task tail" in w for w in small.warnings)
    single_trial = build_suite_result(
        [TaskTrials("a", 1, 1), TaskTrials("b", 5, 3)],
        context="offline", model="m", agent_config_hash="h",
    )
    assert single_trial.suite.headline_k == 1
    assert any("more epochs" in w for w in single_trial.warnings)


def test_k_max_narrows_headline() -> None:
    result = build_suite_result(
        _tasks(6, n=12), context="offline", model="m", agent_config_hash="h", k_max=4
    )
    assert result.suite.headline_k == 4
    assert len(result.suite.reliability_decay_curve) == 4


def test_pydantic_forbids_extras_too() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReliabilityAggregate(
            task_key="t", agent_config_hash="h", model="m", n_rollouts=1,
            context="offline", pass_rate=0.5, bogus_field=1,
        )
