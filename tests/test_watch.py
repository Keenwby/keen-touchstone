"""watch: tumbling windows, two-level alerts, honest edge states."""

import random

import pytest

from keen_touchstone.demo.tracegen import gen_spans_labeled, write_spans_jsonl
from keen_touchstone.online import parse_slo, watch_stream

FIVE_SURE_TASKS = {sig: 1.0 for sig in (
    "support/refund-flow", "support/address-change", "ops/log-triage",
    "ops/incident-summary", "research/competitor-scan",
)}


def _stream(drift=None, runs_per_task=16, seed=2):
    spans, _ = gen_spans_labeled(
        seed=seed, tasks=FIVE_SURE_TASKS, runs_per_task=runs_per_task,
        unsigned_runs=0, drift=drift,
    )
    return spans


def test_parse_slo() -> None:
    assert parse_slo("0.6@4") == (0.6, 4)
    assert parse_slo("1.0@1") == (1.0, 1)
    with pytest.raises(ValueError, match="must look like"):
        parse_slo("60%@4")
    with pytest.raises(ValueError, match="must look like"):
        parse_slo("0.6")


def test_clean_stream_is_all_ok() -> None:
    report = watch_stream(_stream(), window_size=20, slo="0.5@2")
    assert [w.status for w in report.windows] == ["ok", "ok", "ok", "ok"]
    assert report.latest_full_status == "ok"
    assert not report.breached


def test_all_task_drift_breaches_at_the_right_window_and_not_before() -> None:
    """Every task collapses to p=0 from its 8th run: windows 0–1 (runs 1–40,
    local idx 0–7) are clean; windows 2–3 hold only failing runs → per-task
    rates all zero → the whole CI sits below the SLO → confident breach."""
    drift = {"task": "*", "after": 8, "p": 0.0}
    report = watch_stream(_stream(drift=drift), window_size=20, slo="0.5@2")
    assert [w.status for w in report.windows] == ["ok", "ok", "breach", "breach"]
    assert report.breached
    first_breach = next(w for w in report.windows if w.status == "breach")
    assert first_breach.index == 2
    assert "CONFIDENT BREACH" in first_breach.detail


def test_single_task_drift_is_a_warning_not_a_breach() -> None:
    """One task in five collapses: the point estimate drops below the SLO but
    with only 5 tasks the CI straddles it — warning, honestly not a breach."""
    drift = {"task": "ops/log-triage", "after": 8, "p": 0.0}
    report = watch_stream(_stream(drift=drift), window_size=20, slo="0.9@1")
    assert report.windows[0].status == "ok"
    assert report.windows[2].status == "warning"
    assert "could be noise" in report.windows[2].detail
    assert not report.breached  # warnings never fail the exit code


def test_slo_deeper_than_window_supports_is_insufficient_n() -> None:
    # window of 20 runs / 5 tasks = 4 trials per task; SLO at pass^8 is unanswerable
    report = watch_stream(_stream(), window_size=20, slo="0.5@8")
    assert all(w.status == "insufficient_n" for w in report.windows)
    assert "no claim at that depth" in report.windows[0].detail


def test_trailing_partial_window_is_pending() -> None:
    report = watch_stream(_stream(runs_per_task=13), window_size=20, slo="0.5@2")
    assert report.windows[-1].status == "pending"
    assert report.windows[-1].n_runs == 5  # 65 runs -> 3 full windows + 5
    assert report.latest_full_status == "ok"  # pending never drives the verdict


def test_windows_sort_by_start_time_not_file_order() -> None:
    spans = _stream(drift={"task": "*", "after": 8, "p": 0.0})
    shuffled = spans.copy()
    random.Random(99).shuffle(shuffled)
    a = watch_stream(spans, window_size=20, slo="0.5@2")
    b = watch_stream(shuffled, window_size=20, slo="0.5@2")
    assert [w.status for w in a.windows] == [w.status for w in b.windows]
    assert [w.n_runs for w in a.windows] == [w.n_runs for w in b.windows]


def test_repeated_look_note_present() -> None:
    report = watch_stream(_stream(), window_size=20, slo="0.5@2")
    assert any("repeated looks" in w for w in report.warnings)


def test_cli_watch_exit_codes_and_json(tmp_path) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    clean = write_spans_jsonl(tmp_path / "clean.jsonl", _stream())
    breached = write_spans_jsonl(
        tmp_path / "bad.jsonl", _stream(drift={"task": "*", "after": 8, "p": 0.0})
    )
    runner = CliRunner()
    ok = runner.invoke(main, ["watch", str(clean), "--window", "20", "--slo", "0.5@2",
                              "--out", str(tmp_path / "series.json")])
    assert ok.exit_code == 0, ok.output
    assert "✓ ok" in ok.output
    assert (tmp_path / "series.json").exists()

    bad = runner.invoke(main, ["watch", str(breached), "--window", "20", "--slo", "0.5@2"])
    assert bad.exit_code == 1
    assert "BREACH" in bad.output
    assert "CONFIDENT BREACH" in bad.output
