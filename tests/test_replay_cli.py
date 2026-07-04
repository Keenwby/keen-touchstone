"""replay / replay-demo CLI end-to-end."""

import json

import jsonschema
import pytest
from click.testing import CliRunner

from keen_touchstone.artifacts import load_schema
from keen_touchstone.cli import main


@pytest.fixture(scope="module")
def demo_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("replay-demo")
    result = CliRunner().invoke(main, [
        "replay-demo", "--invoices", "5", "--runs-per-invoice", "2", "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    return out, result.output


def test_replay_demo_tells_the_whole_story(demo_out) -> None:
    out, output = demo_out
    assert "failed" in output
    assert "REPLAY FAITHFUL" in output
    assert "same failure" in output
    assert "zero network" in output
    assert "replay it yourself" in output
    payload = json.loads((out / "report" / "aggregate.json").read_text())
    jsonschema.validate(payload["suite"], load_schema("reliability-aggregate"))
    assert (out / "spans.jsonl").exists()
    assert list((out / "cassettes").glob("*.cassette.jsonl"))


def test_cli_replay_faithful_and_divergent(demo_out) -> None:
    out, _ = demo_out
    cassette = str(next((out / "cassettes").glob("*.cassette.jsonl")))
    runner = CliRunner()
    ok = runner.invoke(main, [
        "replay", cassette, "--entry", "keen_touchstone.demo.replay_agent:demo_agent",
    ])
    assert ok.exit_code == 0, ok.output
    assert "REPLAY FAITHFUL" in ok.output

    diverged = runner.invoke(main, [
        "replay", cassette, "--entry", "keen_touchstone.demo.replay_agent:demo_agent_v2",
    ])
    assert diverged.exit_code == 1
    assert "DIVERGED" in diverged.output


def test_cli_replay_entry_validation(demo_out) -> None:
    out, _ = demo_out
    cassette = str(next((out / "cassettes").glob("*.cassette.jsonl")))
    runner = CliRunner()
    bad_format = runner.invoke(main, ["replay", cassette, "--entry", "no-colon"])
    assert bad_format.exit_code == 2
    missing = runner.invoke(main, [
        "replay", cassette, "--entry", "keen_touchstone.demo.replay_agent:nope",
    ])
    assert missing.exit_code == 2
    assert "no attribute" in missing.output
    # [battery C] wrong arity is a usage error, not a fake DIVERGED verdict
    wrong_sig = runner.invoke(main, [
        "replay", cassette, "--entry", "keen_touchstone.demo.replay_agent:scripted_llm",
    ])
    assert wrong_sig.exit_code == 2
    assert "must accept (io, task_input)" in wrong_sig.output
    not_callable = runner.invoke(main, [
        "replay", cassette, "--entry", "keen_touchstone.demo.replay_agent:DEMO_CONFIG_HASH",
    ])
    assert not_callable.exit_code == 2
    assert "not callable" in not_callable.output


def test_join_across_three_artifacts(demo_out) -> None:
    """The Phase 0 promise: cassette, span, and aggregate rows share keys."""
    out, _ = demo_out
    cassette_ids = {p.name.split(".")[0] for p in (out / "cassettes").glob("*.cassette.jsonl")}
    span_lines = [json.loads(x) for x in (out / "spans.jsonl").read_text().splitlines()]
    span_trace_ids = {s["trace_id"] for s in span_lines}
    assert cassette_ids == span_trace_ids
    roots = [s for s in span_lines if s["parent_span_id"] is None]
    assert all(s["harness.replay.cassette_ref"].endswith(".cassette.jsonl") for s in roots)
    payload = json.loads((out / "report" / "aggregate.json").read_text())
    task_keys = {t["task_key"] for t in payload["tasks"]}
    root_signatures = {s["harness.task_signature"] for s in roots}
    assert task_keys == root_signatures
