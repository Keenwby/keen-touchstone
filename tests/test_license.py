"""License issuance rules + the gate + judge report emission."""

import json

import jsonschema
import numpy as np
import pytest
from rich.console import Console

from keen_touchstone.artifacts import CalibrationThresholds, load_schema
from keen_touchstone.judge import alt_test, exam
from keen_touchstone.judge.license import check_license, issue_license
from keen_touchstone.report.judge_emit import emit_license, render_judge_html


def _exam(n=60, agree=0.92, p_pass=0.55, seed=5, abstain_every=0):
    rng = np.random.default_rng(seed)
    human = list(rng.random(n) < p_pass)
    judge = [h if rng.random() < agree else (not h) for h in human]
    if abstain_every:
        judge = [None if i % abstain_every == 0 else j for i, j in enumerate(judge)]
    return exam(human, judge, seed=seed)


def test_good_judge_gets_licensed() -> None:
    cal = issue_license(_exam(), judge_id="good-judge")
    assert cal.status == "JUDGE_LICENSED"
    assert cal.kappa is not None and cal.kappa > 0.6
    ok, msg = check_license(cal)
    assert ok and "JUDGE_LICENSED" in msg
    jsonschema.validate(cal.to_schema_dict(), load_schema("judge-calibration"))


def test_sloppy_judge_blocked_on_kappa() -> None:
    cal = issue_license(_exam(agree=0.65), judge_id="sloppy")
    assert cal.status == "NEEDS_HUMAN"
    assert any("below the licensing threshold" in r for r in cal.reasons)
    ok, msg = check_license(cal)
    assert not ok and "not evidence" in msg


def test_too_few_items_withholds_kappa_entirely() -> None:
    cal = issue_license(_exam(n=20), judge_id="tiny-anchor")
    assert cal.status == "NEEDS_HUMAN"
    assert cal.kappa is None and cal.kappa_ci_low is None
    assert any("withheld" in r for r in cal.reasons)
    jsonschema.validate(cal.to_schema_dict(), load_schema("judge-calibration"))


def test_high_abstention_blocks() -> None:
    cal = issue_license(_exam(abstain_every=3), judge_id="shrugger")
    assert cal.status == "NEEDS_HUMAN"
    assert any("abstention" in r.lower() for r in cal.reasons)


def test_strict_mode_gates_on_ci_low() -> None:
    e = _exam(n=40, agree=0.85)  # decent point, wide-ish CI
    lenient = issue_license(e, judge_id="j", thresholds=CalibrationThresholds())
    strict = issue_license(
        e, judge_id="j", thresholds=CalibrationThresholds(gate_on="ci_low")
    )
    assert lenient.status == "JUDGE_LICENSED"
    # point passed but CI-low is below 0.6 for this construction
    assert strict.status == "NEEDS_HUMAN"
    assert any("licensed on the κ point estimate" in r for r in lenient.reasons)


def test_failed_alt_test_outranks_kappa() -> None:
    rng = np.random.default_rng(9)
    n = 80
    truth = list(rng.random(n) < 0.5)
    human = {
        f"h{j}": [t if rng.random() < 0.95 else (not t) for t in truth] for j in range(3)
    }
    # judge agrees with h0's consensus view often enough for a decent kappa,
    # but is strictly worse than every annotator
    judge = [t if rng.random() < 0.75 else (not t) for t in truth]
    e = exam(truth, judge, seed=9)
    a = alt_test(human, judge, epsilon=0.0)  # no cost discount -> hard to win
    if a.passed:  # construction guard — the design intends a failing alt-test
        pytest.skip("alt-test unexpectedly passed for this seed")
    cal = issue_license(e, judge_id="j", alt=a, n_human_annotators=3)
    assert cal.status == "NEEDS_HUMAN"
    assert any("alt-test FAILED" in r for r in cal.reasons)


def test_inapplicable_alt_test_does_not_block() -> None:
    a = alt_test({"h1": [True, False], "h2": [True, False]}, [True, False])
    cal = issue_license(_exam(), judge_id="j", alt=a)
    assert cal.status == "JUDGE_LICENSED"
    assert cal.alt_test is not None and cal.alt_test.applicable is False


def test_emit_license_writes_files_and_html(tmp_path) -> None:
    e = _exam()
    cal = issue_license(e, judge_id="good-judge", anchor_set_ref="anchors.jsonl")
    path = emit_license(cal, e, None, tmp_path, console=Console(quiet=True))
    assert path.exists()
    data = json.loads(path.read_text())
    jsonschema.validate(data, load_schema("judge-calibration"))
    html = (tmp_path / "judge-report.html").read_text()
    assert "JUDGE_LICENSED" in html
    assert "anti-circularity" in html
    assert "judge exam" in html


def test_gate_refuses_tampered_status() -> None:
    """Round-2 regression (A2/A3): the gate re-derives the decision from the
    recorded numbers — a hand-flipped status is refused, not trusted."""
    cal = issue_license(_exam(agree=0.65), judge_id="sloppy")
    assert cal.status == "NEEDS_HUMAN"
    tampered = cal.model_copy(update={"status": "JUDGE_LICENSED"})
    ok, msg = check_license(tampered)
    assert not ok
    assert "self-contradictory" in msg
    # and the reverse direction: a good license demoted by hand is also flagged
    good = issue_license(_exam(), judge_id="good")
    demoted = good.model_copy(update={"status": "NEEDS_HUMAN"})
    ok2, msg2 = check_license(demoted)
    assert not ok2 and "self-contradictory" in msg2


def test_gate_names_lenient_thresholds_out_loud() -> None:
    """Round-2 regression: loosened thresholds pass (policy is the user's)
    but the gate says so — never silent leniency."""
    e = _exam(n=40, agree=0.8)
    cal = issue_license(
        e, judge_id="lenient",
        thresholds=CalibrationThresholds(kappa_licensed=0.3, min_items=10),
    )
    assert cal.status == "JUDGE_LICENSED"
    ok, msg = check_license(cal)
    assert ok
    assert "more lenient than the defaults" in msg


def test_thresholds_are_bounded() -> None:
    """Round-2 regression: kappa_licensed=-1 (an always-pass 'gate') is not a
    legal license."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalibrationThresholds(kappa_licensed=-1)
    with pytest.raises(ValidationError):
        CalibrationThresholds(min_items=0)


def test_judge_html_escapes_and_renders_needs_human() -> None:
    e = _exam(agree=0.6)
    cal = issue_license(e, judge_id="<script>x</script>")
    html = render_judge_html(cal, e, None)
    assert "<script>x</script>" not in html
    assert "NEEDS_HUMAN" in html
