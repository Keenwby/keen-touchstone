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


def _watch_run(sig, outcome, stamp, tid):
    return {
        "trace_id": tid, "span_id": "0001", "parent_span_id": None,
        "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
        "harness.agent_config_hash": "cfg", "gen_ai.request.model": "m",
        "harness.task_signature": sig, "harness.outcome": outcome,
        "start_time": stamp,
    }


def test_r4_f2_abstention_never_clears_a_standing_breach() -> None:
    """[r4-F2] a breach followed by an insufficient_n window stays standing.

    Fixture rebuilt per fix-verification R2-2: the original fixture produced
    ['breach', 'breach'] and never exercised the abstention branch (theater —
    green pre-fix, green under a state-machine revert). This construction is
    lifted from the reviewer's reproducer: window 1 carries a singleton-trial
    task, so it genuinely abstains at k=2 — and the statuses are pinned so the
    fixture can't drift again."""
    spans = []
    # window 0: A,A,B,B — every run fails, both tasks at n=2 → confident breach
    for i, (sig, out) in enumerate([("A", "failure"), ("A", "failure"),
                                    ("B", "failure"), ("B", "failure")]):
        spans.append(_watch_run(sig, out, f"2026-07-01T08:0{i}:00+00:00", f"w0r{i}"))
    # window 1: A,A,B,C — C has one trial → insufficient_n (still all failing!)
    for i, (sig, out) in enumerate([("A", "failure"), ("A", "failure"),
                                    ("B", "failure"), ("C", "failure")]):
        spans.append(_watch_run(sig, out, f"2026-07-01T09:0{i}:00+00:00", f"w1r{i}"))
    report = watch_stream(spans, window_size=4, slo="0.5@2")
    statuses = [w.status for w in report.windows]
    assert statuses == ["breach", "insufficient_n"], statuses
    assert report.breached  # the abstaining window must NOT retract the page
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
    """[r5-F5] the error-marker rule was dead code — it must now fire (ranked
    among the hypotheses) when the tool output screams failure. (Docstring
    aligned with the any() assertion per fix-verification quarantine note.)"""
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


# ---------------------------------------------------------------------------
# Fix-verification round 2 (reports kt-phase4-fixes / kt-phase5-fixes):
# the fixes above plugged each reviewed PoC; these pin the same-root-cause
# siblings the verification sweep found one door over.
# ---------------------------------------------------------------------------


def _mini_aggregate(tmp_path, name, model, cfg_hash, tasks_rows, curve=None):
    payload = {
        "suite": {
            "model": model, "agent_config_hash": cfg_hash,
            "reliability_decay_curve": curve or [],
        },
        "tasks": tasks_rows,
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_n1_identity_contamination_refused_through_every_door(attr_cells, tmp_path) -> None:
    """[fixver N1] a COPY of a cell (different file, identical identity) must
    be refused whichever flag it arrives under — the original fix guarded only
    the baseline/model_swap pair."""
    import shutil

    runner = CliRunner()
    baseline_copy = tmp_path / "baseline-copy.json"
    shutil.copy(attr_cells["baseline"], baseline_copy)
    mswap_copy = tmp_path / "mswap-copy.json"
    shutil.copy(attr_cells["model_swap"], mswap_copy)

    # V3: baseline copy as --harness-swap — used to publish harness_share 0.0
    v3 = runner.invoke(main, [
        "attribute", "--baseline", attr_cells["baseline"],
        "--model-swap", attr_cells["model_swap"],
        "--harness-swap", str(baseline_copy),
    ])
    assert v3.exit_code == 3, v3.output
    assert "SAME" in v3.output

    # V4: model_swap copy as --both-swap — used to fabricate an interaction
    v4 = runner.invoke(main, [
        "attribute", "--baseline", attr_cells["baseline"],
        "--model-swap", attr_cells["model_swap"],
        "--harness-swap", attr_cells["harness_swap"],
        "--both-swap", str(mswap_copy),
    ])
    assert v4.exit_code == 3, v4.output
    assert "SAME" in v4.output

    # guard-pinning (the mutation the verifier's f1b probe survived):
    # identity-identical through the ORIGINAL door, but via a copied file —
    # the same-file rule can't catch it, only the identity rule can
    v0 = runner.invoke(main, [
        "attribute", "--baseline", attr_cells["baseline"],
        "--model-swap", str(baseline_copy),
        "--harness-swap", attr_cells["harness_swap"],
    ])
    assert v0.exit_code == 3, v0.output
    assert "SAME" in v0.output


def test_n1_identity_less_cells_fail_closed_with_honest_wording(tmp_path) -> None:
    """[fixver quarantine V6] cells carrying no identity metadata fail closed,
    and the message says 'cannot be verified' — not 'same identity'."""
    rows = [{"task_key": "t", "n_rollouts": 2, "pass_rate": 0.5}]
    a = _mini_aggregate(tmp_path, "a", None, None, rows)
    b = _mini_aggregate(tmp_path, "b", None, None, rows)
    c = _mini_aggregate(tmp_path, "c", "m", "h2", rows)
    result = CliRunner().invoke(main, [
        "attribute", "--baseline", a, "--model-swap", b, "--harness-swap", c,
    ])
    assert result.exit_code == 3
    flat = " ".join(result.output.split())
    assert "cannot be verified" in flat
    assert "SAME" not in flat


def test_n2_null_and_mistyped_rows_are_clean_errors(tmp_path) -> None:
    """[fixver N2] null / string / non-integral row values must be clean
    ValueErrors — a null previously tracebacked with TypeError at exit 1."""
    from keen_touchstone.online import load_aggregate_tasks

    def write(row):
        path = tmp_path / "agg.json"
        path.write_text(json.dumps({"suite": {}, "tasks": [row]}))
        return path

    for bad_n in (None, "4", 4.7, True):
        with pytest.raises(ValueError, match="must be an integer"):
            load_aggregate_tasks(write({"task_key": "t", "n_rollouts": bad_n, "pass_rate": 0.5}))
    for bad_rate in (None, "0.5", True):
        with pytest.raises(ValueError, match="must be a number"):
            load_aggregate_tasks(write({"task_key": "t", "n_rollouts": 4, "pass_rate": bad_rate}))

    # and at the CLI it is a domain error (3), never "gate fired" (1)
    bad = write({"task_key": "t", "n_rollouts": None, "pass_rate": 0.5})
    result = CliRunner().invoke(main, ["slo-gate", str(bad), "--slo", "0.5@2"])
    assert result.exit_code == 3, result.output


def test_r2_1_malformed_curve_entries_are_clean_errors(tmp_path) -> None:
    """[fixver R2-1] decay-curve entries get the same validation as task rows."""
    rows = [{"task_key": "t", "n_rollouts": 4, "pass_rate": 0.5}]
    runner = CliRunner()

    no_k = _mini_aggregate(tmp_path, "no-k", "m", "h", rows, curve=[{"pass_hat_k": 0.5}])
    result = runner.invoke(main, ["slo-gate", no_k, "--slo", "0.5@1"])
    assert result.exit_code == 3, result.output

    null_val = _mini_aggregate(
        tmp_path, "null-val", "m", "h", rows,
        curve=[{"k": 1, "pass_hat_k": None, "ci_low": None, "ci_high": None}],
    )
    result = runner.invoke(main, ["slo-gate", null_val, "--slo", "0.5@1", "--strict"])
    assert result.exit_code == 3, result.output


def test_r2_3_window_labels_follow_the_parsed_timeline() -> None:
    """[fixver R2-3] a window's displayed start/end come from the PARSED
    timeline; 10:00+02:00 (=08:00Z) precedes 09:00Z despite string order."""
    spans = [
        _watch_run("A", "failure", "2026-07-01T09:00:00Z", "r-later"),
        _watch_run("A", "failure", "2026-07-01T10:00:00+02:00", "r-earlier"),
    ]
    report = watch_stream(spans, window_size=2, slo="0.5@1")
    window = report.windows[0]
    assert window.start_time == "2026-07-01T10:00:00+02:00", window
    assert window.end_time == "2026-07-01T09:00:00Z"


def test_r2_4_shred_warning_reaches_terminal_and_warnings(tmp_path) -> None:
    """[fixver R2-4] a fully-shredding stream must show the shred warning in
    the CLI output and in aggregate.json warnings — not only in grouping."""
    traces = tmp_path / "shred.jsonl"
    with open(traces, "w") as fh:
        for i in range(12):
            fh.write(json.dumps({
                "trace_id": f"t{i}", "span_id": "r", "parent_span_id": None,
                "gen_ai.operation.name": "invoke_agent", "harness.step_id": 0,
                "gen_ai.request.model": "m", "harness.agent_config_hash": "cfg",
                "input.messages": f"handle case {'x' * (i + 1)}",
                "harness.outcome": "success",
            }) + "\n")
    out = tmp_path / "out"
    result = CliRunner().invoke(main, [
        "ingest", str(traces), "--signature-strategy", "template",
        "--out", str(out), "--resamples", "100",
    ])
    assert result.exit_code == 0, result.output
    assert "SHREDDING" in result.output  # the human sees it next to purity
    payload = json.loads((out / "aggregate.json").read_text())
    assert any("SHREDDING" in w for w in payload["warnings"])


def test_n3_underpowered_annotation_survives_the_terminal_and_json(tmp_path) -> None:
    """[fixver N3] rich ate '[too few tasks for a CI]' as a style tag; the
    underpowered CLI showed a bare point estimate and the JSON kept a
    width-zero confidence_band."""
    rows_fail = [{"task_key": "t1", "n_rollouts": 2, "pass_rate": 0.0}]
    rows_pass = [{"task_key": "t1", "n_rollouts": 2, "pass_rate": 1.0}]
    baseline = _mini_aggregate(tmp_path, "base", "m1", "h1", rows_fail)
    mswap = _mini_aggregate(tmp_path, "mswap", "m2", "h1", rows_pass)
    hswap = _mini_aggregate(tmp_path, "hswap", "m1", "h2", rows_fail)
    out = tmp_path / "out"
    result = CliRunner().invoke(main, [
        "attribute", "--baseline", baseline, "--model-swap", mswap,
        "--harness-swap", hswap, "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert "[too few tasks for a CI]" in flat  # rendered, not eaten as markup
    payload = json.loads((out / "attribution.json").read_text())
    band = payload["attribution"]["confidence_band"]
    assert "too few shared tasks" in band  # no width-zero fake band
    assert ".." not in band


def test_n4_benign_test_summaries_do_not_read_as_error_markers(tmp_path) -> None:
    """[fixver N4] '0 failed, 12 passed' is a tally, not a marker; a real
    failure phrase still fires (pinned by test_r5_f5)."""
    from keen_touchstone.attribution import diagnose_cassette
    from keen_touchstone.cassette import RecordingIO

    def agent(io):
        io.tool_call("test_runner", {"suite": "all"},
                     lambda x: {"summary": "0 failed, 12 passed"})
        raise RuntimeError("crashed after a perfectly happy tool call")

    rec = RecordingIO(tmp_path, task_input={})
    with pytest.raises(RuntimeError), rec as io:
        agent(io)
    report = diagnose_cassette(tmp_path / "cassettes" / f"{rec.run_id}.cassette.jsonl")
    reasons = [h.reason for h in report.hypotheses]
    assert not any("error markers" in r for r in reasons), reasons
