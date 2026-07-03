"""The two-phase loop: verdict gate enforcement in the data path + demo E2E."""

import json

import jsonschema
import pytest
from rich.console import Console

from keen_touchstone.artifacts import EvalVerdict, load_schema
from keen_touchstone.demo.judge_demo import run_judge_demo


def _verdict(trace_id, value, kind="model_graded", ref="cal-good", vid=None):
    return EvalVerdict(
        verdict_id=vid or f"v-{trace_id}",
        trace_id=trace_id,
        scorer_id="j",
        scorer_version="1",
        scorer_kind=kind,
        tier="T2_ungrounded" if kind == "model_graded" else "T0_deterministic",
        value=value,
        judge_calibration_ref=ref if kind == "model_graded" else None,
    )


@pytest.fixture(scope="module")
def demo_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("judge-demo")
    licenses = run_judge_demo(out, n_items=60, seed=2026, console=Console(quiet=True))
    return out, licenses


def test_demo_licenses_split_as_designed(demo_out) -> None:
    _, licenses = demo_out
    assert licenses["good-judge"].status == "JUDGE_LICENSED"
    assert licenses["sloppy-judge"].status == "NEEDS_HUMAN"
    # the rubber-stamp signature: sloppy judge passes real failures at a high rate
    assert licenses["sloppy-judge"].fpr > 0.3
    assert licenses["sloppy-judge"].alt_test.passed is False
    assert licenses["good-judge"].alt_test.passed is True


def test_demo_writes_the_full_loop(demo_out) -> None:
    out, _ = demo_out
    for rel in (
        "anchors.jsonl", "good-judge/license.json", "sloppy-judge/license.json",
        "traces.unlabeled.jsonl", "verdicts.good-judge.jsonl",
        "loop/aggregate.json", "loop/report.html",
    ):
        assert (out / rel).exists(), rel
    payload = json.loads((out / "loop" / "aggregate.json").read_text())
    jsonschema.validate(payload["suite"], load_schema("reliability-aggregate"))
    assert payload["suite"]["context"] == "online"
    assert "licensed judge" in payload["scorer"]
    assert any("licensed judge" in w for w in payload["warnings"])


def test_gate_in_the_data_path(demo_out) -> None:
    from keen_touchstone.judge.verdicts import outcomes_from_verdicts, read_verdicts

    out, licenses = demo_out
    verdicts = read_verdicts(out / "verdicts.good-judge.jsonl")

    ok = outcomes_from_verdicts(verdicts, licenses["good-judge"])
    assert len(ok) == len(verdicts)

    with pytest.raises(ValueError, match="NEEDS_HUMAN"):
        outcomes_from_verdicts(verdicts, licenses["sloppy-judge"])
    with pytest.raises(ValueError, match="no --license"):
        outcomes_from_verdicts(verdicts, None)


def test_license_not_transferable(demo_out) -> None:
    from keen_touchstone.judge.verdicts import outcomes_from_verdicts

    _, licenses = demo_out
    good = licenses["good-judge"]
    foreign = [_verdict("t1", True, ref="cal-someone-else")]
    with pytest.raises(ValueError, match=r"not\s+transferable"):
        outcomes_from_verdicts(foreign, good)


def test_laundering_rejected_at_every_layer(demo_out, tmp_path) -> None:
    """Round-2 regression (A1, Critical): relabeling a model-graded verdict as
    'programmatic' while it still carries judge tells must fail — at pydantic
    construction, at the schema, and at file read."""
    from pydantic import ValidationError

    from keen_touchstone.judge.verdicts import read_verdicts

    with pytest.raises(ValidationError, match="laundered"):
        EvalVerdict(
            verdict_id="v", trace_id="t", scorer_id="j", scorer_version="1",
            scorer_kind="programmatic", tier="T0_deterministic", value=True,
            judge_model="gpt-4-turbo",
        )
    schema = load_schema("eval-verdict")
    laundered = {
        "verdict_id": "v", "trace_id": "t", "scorer_id": "j", "scorer_version": "1",
        "scorer_kind": "trajectory", "tier": "T0_deterministic", "value": True,
        "judge_calibration_ref": "cal-x",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(laundered, schema)
    path = tmp_path / "laundered.jsonl"
    path.write_text(json.dumps(laundered) + "\n")
    with pytest.raises(ValueError, match="invalid EvalVerdict"):
        read_verdicts(path)


def test_identity_binding(demo_out) -> None:
    """Round-2 regression (A4): citing a real license id is not enough — the
    verdicts must come from the judge the license was issued to."""
    from keen_touchstone.judge.verdicts import outcomes_from_verdicts

    _, licenses = demo_out
    good = licenses["good-judge"]
    imposter = [EvalVerdict(
        verdict_id="v-1", trace_id="t1", scorer_id="some-other-judge",
        scorer_version="1", scorer_kind="model_graded", tier="T2_ungrounded",
        value=True, judge_model="(simulated)", judge_calibration_ref=good.calibration_id,
    )]
    with pytest.raises(ValueError, match="issued to"):
        outcomes_from_verdicts(imposter, good)

    wrong_model = [EvalVerdict(
        verdict_id="v-1", trace_id="t1", scorer_id=good.judge_id,
        scorer_version="1", scorer_kind="model_graded", tier="T2_ungrounded",
        value=True, judge_model="some-random-7b", judge_calibration_ref=good.calibration_id,
    )]
    with pytest.raises(ValueError, match="different model"):
        outcomes_from_verdicts(wrong_model, good)


def test_dangling_verdicts_warned(demo_out) -> None:
    """Round-2 regression (A6): verdicts matching no trace must not vanish
    silently."""
    from keen_touchstone.adapters.otel_traces import trials_from_traces
    from keen_touchstone.demo.tracegen import gen_spans_with_truth

    spans, truth = gen_spans_with_truth(seed=5, runs_per_task=4, unsigned_runs=0, include_outcomes=False)
    overrides = dict(truth)
    overrides["no-such-trace-1"] = True
    overrides["no-such-trace-2"] = False
    ingest = trials_from_traces(spans, outcome_overrides=overrides)
    assert any("matched no run" in w for w in ingest.warnings)


def test_programmatic_verdicts_need_no_license() -> None:
    from keen_touchstone.judge.verdicts import outcomes_from_verdicts

    verdicts = [_verdict("t1", True, kind="programmatic"), _verdict("t2", "fail", kind="programmatic")]
    outcomes = outcomes_from_verdicts(verdicts, None)
    assert outcomes == {"t1": True, "t2": False}


def test_verdict_value_coercion_refuses_to_guess() -> None:
    from keen_touchstone.judge.verdicts import outcomes_from_verdicts

    with pytest.raises(ValueError, match="refuse to guess"):
        outcomes_from_verdicts([_verdict("t1", "maybe-ish", kind="programmatic")], None)
    with pytest.raises(ValueError, match="duplicate"):
        outcomes_from_verdicts(
            [_verdict("t1", True, kind="programmatic", vid="a"),
             _verdict("t1", False, kind="programmatic", vid="b")], None
        )


def test_override_ingest_excludes_runs_without_verdicts() -> None:
    from keen_touchstone.adapters.otel_traces import trials_from_traces
    from keen_touchstone.demo.tracegen import gen_spans_with_truth

    spans, truth = gen_spans_with_truth(seed=3, runs_per_task=4, unsigned_runs=0, include_outcomes=False)
    some = dict(list(truth.items())[: len(truth) // 2])
    ingest = trials_from_traces(spans, outcome_overrides=some)
    assert ingest.excluded.get("no_verdict") == len(truth) - len(some)
    assert sum(t.n for t in ingest.tasks) == len(some)


def test_cli_gate_refuses_schema_valid_tamper(demo_out) -> None:
    """Round-2 regression (A2): flipping NEEDS_HUMAN→JUDGE_LICENSED keeps the
    file schema-valid — the gate must still refuse it via re-derivation."""
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    out, _ = demo_out
    original = json.loads((out / "sloppy-judge" / "license.json").read_text())
    original["status"] = "JUDGE_LICENSED"  # schema-valid value, contradicts κ=0.4
    tampered_path = out / "tampered-license.json"
    tampered_path.write_text(json.dumps(original))
    result = CliRunner().invoke(main, ["judge", "gate", str(tampered_path)])
    assert result.exit_code == 1
    assert "self-contradictory" in result.output


def test_cli_ingest_with_license_end_to_end(demo_out) -> None:
    from click.testing import CliRunner

    from keen_touchstone.cli import main

    out, _ = demo_out
    runner = CliRunner()
    good = runner.invoke(main, [
        "ingest", str(out / "traces.unlabeled.jsonl"),
        "--outcomes-from", str(out / "verdicts.good-judge.jsonl"),
        "--license", str(out / "good-judge" / "license.json"),
        "--out", str(out / "cli-loop"), "--resamples", "300",
    ])
    assert good.exit_code == 0, good.output
    assert (out / "cli-loop" / "aggregate.json").exists()

    blocked = runner.invoke(main, [
        "ingest", str(out / "traces.unlabeled.jsonl"),
        "--outcomes-from", str(out / "verdicts.good-judge.jsonl"),
        "--license", str(out / "sloppy-judge" / "license.json"),
        "--out", str(out / "cli-blocked"),
    ])
    assert blocked.exit_code == 1
    assert "NEEDS_HUMAN" in blocked.output
    assert not (out / "cli-blocked" / "aggregate.json").exists()  # nothing emitted

    no_license = runner.invoke(main, [
        "ingest", str(out / "traces.unlabeled.jsonl"),
        "--outcomes-from", str(out / "verdicts.good-judge.jsonl"),
        "--out", str(out / "cli-nolicense"),
    ])
    assert no_license.exit_code == 1
    assert "not evidence" in no_license.output
