"""Derived task identity: normalization goldens, truth recovery, honesty readout."""

from collections import Counter

import pytest

from keen_touchstone.adapters.otel_traces import trials_from_traces
from keen_touchstone.adapters.signatures import (
    HYPOTHESIS_CAVEAT,
    TemplateStrategy,
    ToolSequenceStrategy,
    grouping_readout,
    normalize_template,
    runs_from_spans,
)
from keen_touchstone.demo.tracegen import gen_spans_labeled

# ------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Refund order #7042 for customer C842", "refund order <num> for customer <id>"),
        ("email confirmation to user842@example.com", "email confirmation to <email>"),
        ("since 2026-06-14 check https://example.com/x?y=1", "since <date> check <url>"),
        ("incident INC-7042 and uuid 123e4567-e89b-12d3-a456-426614174000",
         "incident <id> and uuid <id>"),
        ("amounts 1,234.50 and 99", "amounts <num> and <num>"),
        ("   collapse\n whitespace  ", "collapse whitespace"),
    ],
)
def test_normalize_template_goldens(raw: str, expected: str) -> None:
    assert normalize_template(raw) == expected


def test_same_task_different_ids_share_a_signature() -> None:
    spans, _ = gen_spans_labeled(seed=1, runs_per_task=5, unsigned_runs=0,
                                 include_signatures=False)
    runs = runs_from_spans(spans)
    strategy = TemplateStrategy()
    sigs = [strategy.signature(r) for r in runs]
    assert all(s is not None for s in sigs)
    # 5 true tasks × 5 runs -> exactly 5 template clusters of size 5
    counts = Counter(sigs)
    assert len(counts) == 5
    assert set(counts.values()) == {5}


def test_truth_recovery_purity_on_unlabeled_traffic() -> None:
    """The keystone claim: derived grouping recovers the true task structure
    on traffic that carries no tags at all."""
    spans, labels = gen_spans_labeled(seed=7, runs_per_task=12, unsigned_runs=0,
                                      include_signatures=False)
    runs = runs_from_spans(spans)
    strategy = TemplateStrategy()
    cluster_truth: dict[str, list[str]] = {}
    for run in runs:
        sig = strategy.signature(run)
        assert sig is not None
        cluster_truth.setdefault(sig, []).append(labels[run.trace_id]["task"])
    # purity: each derived cluster is dominated by one true task
    total = pure = 0
    for members in cluster_truth.values():
        top = Counter(members).most_common(1)[0][1]
        total += len(members)
        pure += top
    assert pure / total >= 0.95
    # and no over-merging: number of clusters == number of true tasks
    assert len(cluster_truth) == 5


def test_runs_without_input_are_unsigned_not_guessed() -> None:
    spans, _ = gen_spans_labeled(seed=3, runs_per_task=3, unsigned_runs=2,
                                 include_signatures=False)
    # unsigned_runs have no template input (no task) -> strategy returns None
    strategy = TemplateStrategy()
    readout = grouping_readout(spans, strategy)
    assert readout.n_unsigned == 2
    assert readout.n_grouped == 15


def test_tool_sequence_strategy_groups_and_collapses() -> None:
    spans = []
    for tid, tools in (("t1", ["search", "search", "sql"]), ("t2", ["search", "sql"])):
        spans.append({"trace_id": tid, "span_id": "r", "parent_span_id": None,
                      "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0})
        for i, tool in enumerate(tools):
            spans.append({"trace_id": tid, "span_id": f"s{i}", "parent_span_id": "r",
                          "gen_ai.operation.name": "execute_tool",
                          "gen_ai.tool.name": tool, "harness.step_id": i + 1})
    runs = runs_from_spans(spans)
    strategy = ToolSequenceStrategy()
    sigs = {r.trace_id: strategy.signature(r) for r in runs}
    # consecutive repeat collapsed -> both runs share search>sql
    assert sigs["t1"] == sigs["t2"] is not None
    assert strategy.templates[sigs["t1"]] == "search>sql"
    # a run with no tools is unsigned
    no_tools = runs_from_spans([{"trace_id": "t3", "span_id": "r", "parent_span_id": None,
                                 "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0}])
    assert strategy.signature(no_tools[0]) is None


def test_readout_honesty_fields() -> None:
    # keep declared tags so purity-vs-declared is computable
    spans, _ = gen_spans_labeled(seed=5, runs_per_task=6, unsigned_runs=0)
    readout = grouping_readout(spans, TemplateStrategy())
    assert readout.caveat == HYPOTHESIS_CAVEAT
    assert readout.n_clusters == 5
    assert readout.singleton_rate == 0.0
    assert readout.purity_vs_declared == pytest.approx(1.0)
    assert readout.declared_coverage == pytest.approx(1.0)
    top = readout.clusters[0]
    assert top.template is not None and "<" in top.template  # readable mask
    assert top.exemplars and top.majority_declared is not None


def test_derived_strategy_feeds_ingest() -> None:
    spans, labels = gen_spans_labeled(seed=9, runs_per_task=8, unsigned_runs=0,
                                      include_signatures=False)
    ingest = trials_from_traces(spans, strategy=TemplateStrategy())
    assert len(ingest.tasks) == 5
    assert all(t.n == 8 for t in ingest.tasks)
    # outcomes still come from the declared outcome attrs, grouping is derived
    truth_by_task = Counter()
    ok_by_task = Counter()
    for lab in labels.values():
        truth_by_task[lab["task"]] += 1
        ok_by_task[lab["task"]] += int(lab["ok"])
    assert sorted(t.c for t in ingest.tasks) == sorted(ok_by_task.values())


def test_deterministic_across_instances() -> None:
    spans, _ = gen_spans_labeled(seed=11, runs_per_task=4, include_signatures=False)
    a = {r.trace_id: TemplateStrategy().signature(r) for r in runs_from_spans(spans)}
    b = {r.trace_id: TemplateStrategy().signature(r) for r in runs_from_spans(spans)}
    assert a == b


def test_cli_ingest_with_template_strategy(tmp_path) -> None:
    import json

    from click.testing import CliRunner

    from keen_touchstone.cli import main
    from keen_touchstone.demo.tracegen import write_spans_jsonl

    spans, _ = gen_spans_labeled(seed=13, runs_per_task=6, unsigned_runs=1,
                                 include_signatures=False)
    traces = write_spans_jsonl(tmp_path / "t.jsonl", spans)
    result = CliRunner().invoke(main, [
        "ingest", str(traces), "--signature-strategy", "template",
        "--out", str(tmp_path / "out"), "--resamples", "300",
    ])
    assert result.exit_code == 0, result.output
    assert "task-identity readout" in result.output
    assert "HYPOTHESIS" in result.output
    payload = json.loads((tmp_path / "out" / "aggregate.json").read_text())
    assert payload["suite"]["task_key_source"] == "task_signature"
    assert payload["grouping"]["strategy"] == "template"
    assert payload["grouping"]["n_clusters"] == 5
    assert any("HYPOTHESIS" in w for w in payload["warnings"])
