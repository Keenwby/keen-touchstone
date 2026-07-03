"""OTel trace adapter: grouping, outcome rules, exclusion accounting, truth recovery."""

import json

import jsonschema
import pytest

from keen_touchstone.adapters.otel_traces import (
    DeclaredTagStrategy,
    read_spans_jsonl,
    trials_from_traces,
)
from keen_touchstone.demo.tracegen import DEMO_TRACE_TASKS, gen_spans, write_spans_jsonl


def test_generated_spans_validate_against_span_schema() -> None:
    from keen_touchstone.artifacts import load_schema

    schema = load_schema("span")
    for span in gen_spans(seed=1, runs_per_task=2, unsigned_runs=1):
        jsonschema.validate(span, schema)


def test_generator_deterministic() -> None:
    assert gen_spans(seed=7) == gen_spans(seed=7)
    assert gen_spans(seed=7) != gen_spans(seed=8)


def test_ingest_groups_and_excludes(tmp_path) -> None:
    path = write_spans_jsonl(tmp_path / "t.jsonl", gen_spans(seed=3, runs_per_task=10, unsigned_runs=2))
    ingest = trials_from_traces(read_spans_jsonl(path))
    assert sorted(t.task_key for t in ingest.tasks) == sorted(DEMO_TRACE_TASKS)
    assert all(t.n == 10 for t in ingest.tasks)
    assert ingest.excluded == {"no_task_signature": 2}
    assert any("excluded 2 run(s)" in w for w in ingest.warnings)
    assert ingest.model == "demo/agent-model"
    assert ingest.agent_config_hash == "democfg00003"  # declared, not defaulted
    assert all(t.tokens is not None for t in ingest.tasks)
    assert ingest.n_runs == 52


def test_truth_recovery_within_ci() -> None:
    """The whole point: estimates from traces we didn't author recover the
    known true reliability. Deterministic via seed."""
    from keen_touchstone.aggregate import build_suite_result

    spans = gen_spans(seed=11, runs_per_task=30, unsigned_runs=0)
    ingest = trials_from_traces(spans)
    result = build_suite_result(
        ingest.tasks, context="online", model=ingest.model,
        agent_config_hash=ingest.agent_config_hash, seed=11,
    )
    ps = list(DEMO_TRACE_TASKS.values())
    for point in result.suite.reliability_decay_curve:
        true_agg = sum(p**point.k for p in ps) / len(ps)
        assert point.ci_low - 1e-9 <= true_agg <= point.ci_high + 1e-9, (
            f"k={point.k}: true {true_agg:.3f} outside [{point.ci_low:.3f}, {point.ci_high:.3f}]"
        )


def test_outcome_priority_and_variants() -> None:
    def run_spans(**root_extra):
        return [{
            "trace_id": "t1", "span_id": "s1", "parent_span_id": None,
            "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
            "harness.task_signature": "sig", "gen_ai.request.model": "m",
            **root_extra,
        }]

    # declared outcome wins over evaluation score
    both = run_spans(**{"harness.outcome": "failure", "gen_ai.evaluation.score.value": 1.0})
    assert trials_from_traces(both).tasks[0].c == 0
    # evaluation score value with threshold
    assert trials_from_traces(run_spans(**{"gen_ai.evaluation.score.value": 0.9})).tasks[0].c == 0
    assert trials_from_traces(run_spans(**{"gen_ai.evaluation.score.value": 1.0})).tasks[0].c == 1
    # label variant
    assert trials_from_traces(run_spans(**{"gen_ai.evaluation.score.label": "pass"})).tasks[0].c == 1
    # boolean declaration
    assert trials_from_traces(run_spans(**{"harness.outcome": True})).tasks[0].c == 1
    # regex fallback
    regex_run = run_spans(**{"output.messages": "Final answer: DONE ✅"})
    assert trials_from_traces(regex_run, outcome_regex="DONE").tasks[0].c == 1
    # nothing resolvable -> excluded, and with zero usable runs that's an error
    with pytest.raises(ValueError, match="no usable runs"):
        trials_from_traces(run_spans())


def test_mixed_models_and_configs_rejected() -> None:
    spans = gen_spans(seed=5, runs_per_task=2, unsigned_runs=0)
    other = gen_spans(seed=6, runs_per_task=2, unsigned_runs=0, model="other/model")
    with pytest.raises(ValueError, match="multiple"):
        trials_from_traces(spans + other)


def test_default_config_hash_warns() -> None:
    spans = [{
        "trace_id": "t1", "span_id": "s1", "parent_span_id": None,
        "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
        "harness.task_signature": "sig", "gen_ai.request.model": "m",
        "harness.outcome": "success",
    }]
    ingest = trials_from_traces(spans)
    assert any("no harness.agent_config_hash" in w for w in ingest.warnings)
    assert len(ingest.agent_config_hash) == 12


def test_strategy_interface_is_swappable() -> None:
    class EverythingIsOneTask:
        name = "constant"

        def signature(self, run):
            return "the-only-task"

    spans = gen_spans(seed=2, runs_per_task=3, unsigned_runs=2)
    ingest = trials_from_traces(spans, strategy=EverythingIsOneTask())
    assert [t.task_key for t in ingest.tasks] == ["the-only-task"]
    assert ingest.excluded == {}  # constant strategy signs everything


def test_cli_ingest_demo_end_to_end(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.artifacts import load_schema
    from keen_touchstone.cli import main

    out = tmp_path / "ingest-out"
    result = CliRunner().invoke(main, ["ingest", "--demo", "--out", str(out), "--resamples", "300"])
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "aggregate.json").read_text())
    jsonschema.validate(payload["suite"], load_schema("reliability-aggregate"))
    assert payload["suite"]["context"] == "online"
    assert payload["suite"]["task_key_source"] == "declared_tag"
    assert (out / "traces.demo.jsonl").exists()


def test_token_coercion_tolerates_malformed_usage() -> None:
    """Adversarial-review regression: a float-as-string token value aborted
    the whole ingest; malformed usage now counts 0, never crashes."""
    spans = [{
        "trace_id": "t1", "span_id": "s1", "parent_span_id": None,
        "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
        "harness.task_signature": "sig", "gen_ai.request.model": "m",
        "harness.outcome": "success",
        "gen_ai.usage.input_tokens": "300.5", "gen_ai.usage.output_tokens": "garbage",
    }]
    ingest = trials_from_traces(spans)
    assert ingest.tasks[0].tokens == (300,)


def test_cli_clean_error_on_bad_ingest(tmp_path) -> None:
    """Domain errors reach the user as a clean message, not a traceback."""
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trace_id": "t", "span_id": "s"}\n')  # no signature/outcome
    result = CliRunner().invoke(main, ["ingest", str(bad)])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_read_spans_jsonl_errors(tmp_path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"trace_id": "a", "span_id": "s"}\nnot-json\n')
    with pytest.raises(ValueError, match="not valid JSON"):
        read_spans_jsonl(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no spans"):
        read_spans_jsonl(empty)
    assert DeclaredTagStrategy.name == "declared_tag"
