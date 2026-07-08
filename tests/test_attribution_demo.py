"""attribute-demo truth recovery + attribute/diagnose CLI end to end."""

import json

import jsonschema
import pytest
from click.testing import CliRunner
from rich.console import Console

from keen_touchstone.artifacts import load_schema
from keen_touchstone.cli import main
from keen_touchstone.demo.attribution_demo import run_attribution_demo


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("attr-demo")
    summary = run_attribution_demo(out, seed=2026, console=Console(quiet=True))
    return out, summary


def test_designed_truth_recovered_exactly(demo) -> None:
    _, summary = demo
    result = summary["attribution"]
    assert result.baseline_failure == pytest.approx(13 / 24)
    assert result.model_share.delta == pytest.approx(6 / 24)
    assert result.harness_share.delta == pytest.approx(6 / 24)
    assert result.interaction.delta == pytest.approx(1 / 24)
    assert result.irreducible_failure == pytest.approx(0.0)
    assert result.model_share.significant and result.harness_share.significant
    assert "BOTH fixes" in result.sentence()
    # truth lands inside its own CIs
    assert result.model_share.ci_low <= 6 / 24 <= result.model_share.ci_high


def test_demo_diagnosis_is_caged(demo) -> None:
    _, summary = demo
    diagnosis = summary["diagnosis"]
    assert diagnosis.failed and diagnosis.error_type == "ValueError"
    assert diagnosis.hypotheses[0].layer == "Execution"
    assert "NOT A VERDICT" in diagnosis.disclaimer


def test_cli_attribute_on_demo_cells(demo) -> None:
    out, _ = demo
    args = ["attribute"]
    for flag, cell in (("--baseline", "baseline"), ("--model-swap", "model_swap"),
                       ("--harness-swap", "harness_swap"), ("--both-swap", "both_swap")):
        args += [flag, str(out / cell / "report" / "aggregate.json")]
    args += ["--out", str(out / "attr-out")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "baseline failure rate 54.2%" in result.output
    assert "significant" in result.output

    payload = json.loads((out / "attr-out" / "attribution.json").read_text())
    # the Phase 0 attribution block, filled and schema-valid at last
    schema = load_schema("reliability-aggregate")["properties"]["attribution"]
    jsonschema.validate(payload["attribution"], schema)
    assert payload["attribution"]["method"] == "measured_ab"
    assert payload["detail"]["interaction"]["delta"] == pytest.approx(1 / 24)
    assert payload["sentence"].startswith("baseline failure rate")


def test_cli_attribute_three_cells_honest(demo) -> None:
    out, _ = demo
    args = ["attribute",
            "--baseline", str(out / "baseline" / "report" / "aggregate.json"),
            "--model-swap", str(out / "model_swap" / "report" / "aggregate.json"),
            "--harness-swap", str(out / "harness_swap" / "report" / "aggregate.json"),
            "--out", str(out / "attr-3cell")]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "not estimable" in result.output


def test_cli_diagnose(demo) -> None:
    out, _summary = demo
    # find the tape the demo diagnosed (a ValueError crash in the baseline cell)
    crashed = None
    from keen_touchstone.attribution import diagnose_cassette

    for cassette in (out / "baseline" / "cassettes").glob("*.cassette.jsonl"):
        if diagnose_cassette(cassette).failed:
            crashed = cassette
            break
    assert crashed is not None
    result = CliRunner().invoke(main, ["diagnose", str(crashed)])
    assert result.exit_code == 0, result.output
    assert "NOT A VERDICT" in result.output
    assert "Execution" in result.output
    assert "MEASURE instead of guess" in result.output

    # a successful tape refuses politely
    ok_tape = None
    for cassette in (out / "both_swap" / "cassettes").glob("*.cassette.jsonl"):
        if not diagnose_cassette(cassette).failed:
            ok_tape = cassette
            break
    assert ok_tape is not None
    result = CliRunner().invoke(main, ["diagnose", str(ok_tape)])
    assert result.exit_code == 0
    assert "nothing to diagnose" in result.output


def test_cli_attribute_demo_e2e(tmp_path) -> None:
    result = CliRunner().invoke(main, ["attribute-demo", "--out", str(tmp_path / "ad")])
    assert result.exit_code == 0, result.output
    assert "the sentence no other tool can produce" in result.output
    assert "swapping the model removes +25.0pp" in result.output
    assert "NOT A VERDICT" in result.output
