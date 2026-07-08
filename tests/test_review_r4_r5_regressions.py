"""Regressions for the pipeline review rounds r4 (Phase 4) + r5 (Phase 5).

Each test pins one reviewer finding using the reviewer's own reproduction
logic. Tagged [r4-Fn]/[r5-Fn] to match the reports in
~/workspace/claude-projects/adversarial-review/reviews/.
"""

import json

import pytest
from click.testing import CliRunner
from rich.console import Console

from keen_touchstone.cli import main
from keen_touchstone.demo.tracegen import gen_spans_labeled
from keen_touchstone.online import watch_stream
from keen_touchstone.online.watch import parse_stamp

SURE_TASKS = {sig: 1.0 for sig in (
    "support/refund-flow", "support/address-change", "ops/log-triage",
    "ops/incident-summary", "research/competitor-scan",
)}


def _strip_timestamps(spans):
    return [{k: v for k, v in s.items() if k != "start_time"} for s in spans]


def test_r4_f1_untimestamped_drift_still_breaches() -> None:
    """[r4-F1] the drift signal must survive a stream with no timestamps at
    all (was: lexicographic trace-id scramble smeared it into all-warnings)."""
    spans, _ = gen_spans_labeled(
        seed=2, tasks=SURE_TASKS, runs_per_task=16, unsigned_runs=0,
        drift={"task": "*", "after": 8, "p": 0.0},
    )
    stripped = _strip_timestamps(spans)
    report = watch_stream(stripped, window_size=20, slo="0.5@2")
    assert report.breached, [w.status for w in report.windows]
    assert any(w.status == "breach" for w in report.windows)
    assert any("no start_time" in w for w in report.warnings)


def test_r4_f2_abstention_never_clears_a_standing_breach() -> None:
    """[r4-F2] a breach followed by an insufficient_n window stays standing."""
    spans, _ = gen_spans_labeled(
        seed=3, tasks=SURE_TASKS, runs_per_task=8, unsigned_runs=0,
        drift={"task": "*", "after": 0, "p": 0.0},  # everything fails
    )
    # second window can't answer at k=4 (2 trials/task per window of 20)
    report = watch_stream(spans, window_size=20, slo="0.5@2")
    statuses = [w.status for w in report.windows]
    assert statuses[0] == "breach"
    assert report.breached  # regardless of what follows
    if report.latest_full_status != "breach":
        assert any("STILL STANDING" in w for w in report.warnings)


def test_r4_f3_domain_errors_exit_3_not_1(tmp_path) -> None:
    """[r4-F3] CI must distinguish 'gate fired' (1) from 'pipeline broken' (3)."""
    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text("not-json\n")
    runner = CliRunner()
    result = runner.invoke(main, ["watch", str(garbage), "--slo", "0.5@2"])
    assert result.exit_code == 3, result.output
    not_agg = tmp_path / "x.json"
    not_agg.write_text("{}")
    result = runner.invoke(main, ["slo-gate", str(not_agg), "--slo", "0.5@2"])
    assert result.exit_code == 3
    result = runner.invoke(main, ["compare", str(not_agg), str(not_agg), "--at-k", "2"])
    assert result.exit_code == 3


def test_r4_f4_timestamps_parse_to_a_real_timeline() -> None:
    """[r4-F4] mixed UTC offsets and numeric epochs order chronologically."""
    a = parse_stamp("2026-07-01T10:00:00+02:00")  # = 08:00Z
    b = parse_stamp("2026-07-01T09:00:00Z")
    assert a < b  # string comparison said otherwise
    early, late = parse_stamp(999_999_999), parse_stamp(1_000_000_000)
    assert early < late  # digit-count string sort said otherwise
    assert parse_stamp(1_700_000_000_000) is not None  # milliseconds
    assert parse_stamp(1.7e18) is not None  # OTLP nanoseconds
    assert parse_stamp("NOT-A-DATE") is None
    assert parse_stamp(None) is None


def test_r4_f5_hex_ids_mask_and_shredding_warns() -> None:
    """[r4-F5] git-SHA-shaped ids mask now; total under-merge triggers the
    shred warning instead of parading 100% purity."""
    from keen_touchstone.adapters.signatures import (
        TemplateStrategy,
        grouping_readout,
        normalize_template,
    )

    assert normalize_template("Review commit deadbeefc4fe") == \
        normalize_template("Review commit cafebabe12aa")

    # a still-splitting shape (plural/singular is a documented limit) →
    # singletons dominate → the readout must warn that purity is meaningless
    spans = []
    for i in range(12):
        spans.append({
            "trace_id": f"t{i}", "span_id": "r", "parent_span_id": None,
            "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
            "harness.task_signature": "one-true-task",
            "input.messages": f"handle case {'x' * (i + 1)}",  # unmaskable variety
            "harness.outcome": "success",
        })
    readout = grouping_readout(spans, TemplateStrategy())
    assert readout.singleton_rate == 1.0
    assert readout.purity_vs_declared == 1.0  # trivially perfect…
    assert readout.shred_warning is not None  # …and called out as meaningless
    assert "SHREDDING" in readout.shred_warning


def test_r4_f6_slo_gate_refuses_to_gate_without_ci(tmp_path) -> None:
    """[r4-F6] no CI at the SLO's k → refuse (default mode), not vacuous pass."""
    from keen_touchstone.online import slo_gate

    payload = {
        "suite": {"reliability_decay_curve": [
            {"k": 1, "pass_hat_k": 0.01, "ci_low": None, "ci_high": None},
        ]},
        "tasks": [{"task_key": "t", "n_rollouts": 4, "pass_rate": 0.0}],
    }
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cannot assess"):
        slo_gate(path, "0.99@1")
    strict = slo_gate(path, "0.99@1", strict=True)  # strict gates on the point
    assert not strict.ok


def _demo_cells(tmp_path):
    from keen_touchstone.demo.attribution_demo import run_attribution_demo

    run_attribution_demo(tmp_path, console=Console(quiet=True))
    return {name: str(tmp_path / name / "report" / "aggregate.json")
            for name in ("baseline", "model_swap", "harness_swap", "both_swap")}


@pytest.fixture(scope="module")
def attr_cells(tmp_path_factory):
    return _demo_cells(tmp_path_factory.mktemp("r5"))


def test_r5_f1_same_file_and_identical_identity_rejected(attr_cells) -> None:
    """[r5-F1] cell contamination is refused, not silently analyzed."""
    runner = CliRunner()
    same_file = runner.invoke(main, [
        "attribute", "--baseline", attr_cells["baseline"],
        "--model-swap", attr_cells["baseline"],  # same file twice
        "--harness-swap", attr_cells["harness_swap"],
    ])
    assert same_file.exit_code == 3
    assert "SAME" in same_file.output

    swapped = runner.invoke(main, [
        "attribute", "--baseline", attr_cells["baseline"],
        "--model-swap", attr_cells["harness_swap"],  # mislabeled cells
        "--harness-swap", attr_cells["model_swap"],
        "--out", "/tmp/kt-r5-swapped",
    ])
    # mislabeled swap cells: model tags contradict the 2x2 design → warned
    assert swapped.exit_code == 0
    assert "check your cells" in swapped.output


def test_r5_f2_out_of_range_share_emits_null_not_zero(tmp_path) -> None:
    """[r5-F2] a negative share must not read as 'measured: no effect'."""
    from keen_touchstone.aggregate import build_suite_result
    from keen_touchstone.report import RunMeta, emit
    from keen_touchstone.stats import TaskTrials

    def cell(name, model, fail_ids):
        tasks = [TaskTrials(f"t{i}", 4, 0 if i in fail_ids else 4) for i in range(1, 11)]
        result = build_suite_result(tasks, context="offline", model=model,
                                    agent_config_hash=name)
        emit(result, tmp_path / name, RunMeta(source=name, task_name="t", model=model),
             console=Console(quiet=True))
        return str(tmp_path / name / "aggregate.json")

    baseline = cell("base", "m1", set())          # nothing fails
    worse = cell("worse", "m2", {1, 2, 3, 4, 5, 6})  # the "better" model fails 6
    hswap = cell("hswap", "m1-h2", set())
    result = CliRunner().invoke(main, [
        "attribute", "--baseline", baseline, "--model-swap", worse,
        "--harness-swap", hswap, "--out", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "out" / "attribution.json").read_text())
    assert payload["attribution"]["model_share"] is None  # null, not 0.0
    assert payload["detail"]["model_share"]["delta"] == pytest.approx(-0.6)
    assert "WORSE" in result.output


def test_r5_f3_invalid_rows_are_clean_errors(tmp_path) -> None:
    """[r5-F3] n=0, out-of-range pass_rate, and missing keys → ValueError,
    never a raw traceback or an impossible published number."""
    from keen_touchstone.online import load_aggregate_tasks

    def write(rows):
        path = tmp_path / "agg.json"
        path.write_text(json.dumps({"suite": {}, "tasks": rows}))
        return path

    with pytest.raises(ValueError, match="n_rollouts=0"):
        load_aggregate_tasks(write([{"task_key": "t", "n_rollouts": 0, "pass_rate": 0.0}]))
    with pytest.raises(ValueError, match="outside"):
        load_aggregate_tasks(write([{"task_key": "t", "n_rollouts": 4, "pass_rate": 1.5}]))
    with pytest.raises(ValueError, match="missing"):
        load_aggregate_tasks(write([{"task_key": "t", "pass_rate": 0.5}]))


def test_r5_f5_tool_error_markers_surface(tmp_path) -> None:
    """[r5-F5] the error-marker rule was dead code — now it must be the TOP
    hypothesis reason when the tool output screams failure."""
    from keen_touchstone.attribution import diagnose_cassette
    from keen_touchstone.cassette import RecordingIO

    def agent(io):
        io.tool_call("uploader", {"batch": 7},
                     lambda x: {"status": "3 uploads failed, aborting", "error": "disk full"})
        raise RuntimeError("giving up")

    rec = RecordingIO(tmp_path, task_input={})
    with pytest.raises(RuntimeError), rec as io:
        agent(io)
    report = diagnose_cassette(tmp_path / "cassettes" / f"{rec.run_id}.cassette.jsonl")
    reasons = [h.reason for h in report.hypotheses]
    assert any("error markers" in r for r in reasons), reasons


def test_r5_f6_tiny_n_annotates_instead_of_fake_ci() -> None:
    """[r5-F6] a width-zero bootstrap bracket over one task is false precision."""
    from keen_touchstone.attribution import decompose

    single = {"only": (2, 0)}
    fixed = {"only": (2, 2)}
    result = decompose(single, fixed, dict(single))
    assert result.underpowered
    assert "[too few tasks for a CI]" in result.sentence()
    assert "[+" not in result.sentence().split("harness")[0]  # no fake bracket on model share


def test_r5_f7_unequal_trials_across_cells_noted() -> None:
    """[r5-F7] baseline n=2 vs swap n=50 must be said, not implied equal."""
    from keen_touchstone.attribution import decompose

    baseline = {f"t{i}": (2, 0) for i in range(1, 9)}
    mswap = {f"t{i}": (50, 50) for i in range(1, 9)}
    hswap = {f"t{i}": (2, 2) for i in range(1, 9)}
    result = decompose(baseline, mswap, hswap)
    assert any("UNEQUAL trial counts" in n for n in result.notes)


def test_r4_f1_watch_windows_use_file_order_when_untimestamped(tmp_path) -> None:
    """[r4-F1 corollary] runs_from_spans preserves insertion order."""
    from keen_touchstone.adapters.signatures import runs_from_spans

    spans = []
    for tid in ("zz-first", "aa-second", "mm-third"):  # anti-lexicographic
        spans.append({"trace_id": tid, "span_id": "r", "parent_span_id": None,
                      "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0})
    assert [r.trace_id for r in runs_from_spans(spans)] == ["zz-first", "aa-second", "mm-third"]
