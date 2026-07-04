"""Cassette IO + recording + the seven replay primitives."""

import json

import jsonschema
import pytest

from keen_touchstone.artifacts import load_schema
from keen_touchstone.cassette import (
    CassetteWriter,
    RecordingIO,
    TraceEvent,
    read_cassette,
    replay_run,
)
from keen_touchstone.demo.replay_agent import demo_agent, demo_agent_v2

# ------------------------------------------------------------------- P0: file IO


def _event(step, kind="tool_call", **over):
    base = dict(
        run_id="r1", step_id=step, timestamp="2026-07-03T00:00:00+00:00",
        kind=kind, input={"x": step}, output={"y": step},
        metadata={"tool_id": "t"} if kind == "tool_call" else {"model_id": "m"},
    )
    base.update(over)
    return TraceEvent(**base)


def test_cassette_round_trip(tmp_path) -> None:
    path = tmp_path / "r1.cassette.jsonl"
    with CassetteWriter(path, "r1") as writer:
        for step in range(3):
            writer.append(_event(step))
    events = read_cassette(path)
    assert [e.step_id for e in events] == [0, 1, 2]
    assert events[1].input == {"x": 1}
    schema = load_schema("cassette")
    for line in path.read_text().splitlines():
        jsonschema.validate(json.loads(line), schema)


def test_cassette_refuses_corruption(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"run_id": "r1"}\n')
    with pytest.raises(ValueError, match="not a valid cassette"):
        read_cassette(path)
    path.write_text("not-json\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_cassette(path)
    with pytest.raises(ValueError, match="empty"):
        (tmp_path / "empty.jsonl").write_text("\n")
        read_cassette(tmp_path / "empty.jsonl")


def test_cassette_refuses_spliced_tapes(tmp_path) -> None:
    path = tmp_path / "spliced.jsonl"
    lines = [json.dumps(_event(s).to_dict()) for s in (0, 2, 1)]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_cassette(path)
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        json.dumps(_event(0).to_dict()) + "\n"
        + json.dumps(_event(1, run_id="OTHER").to_dict()) + "\n"
    )
    with pytest.raises(ValueError, match="one cassette holds exactly one run"):
        read_cassette(mixed)


def test_writer_rejects_foreign_run_id(tmp_path) -> None:
    with CassetteWriter(tmp_path / "c.jsonl", "r1") as writer, pytest.raises(ValueError):
        writer.append(_event(0, run_id="r2"))


# ---------------------------------------------------------------- P1: recording


def test_recording_success_run(tmp_path) -> None:
    task = {"invoice_id": "INV-001"}  # not poisoned
    with RecordingIO(tmp_path, task_input=task, task_signature="demo/x") as io:
        result = demo_agent(io, task)
        io.finish(result)
    cassette = tmp_path / "cassettes" / f"{io.run_id}.cassette.jsonl"
    events = read_cassette(cassette)
    kinds = [e.kind for e in events]
    assert kinds.count("llm_call") == 2
    assert kinds.count("tool_call") == 2
    finals = [e for e in events if e.decision_name == "__final__"]
    assert len(finals) == 1 and finals[0].output["ok"] is True
    # companion spans validate and carry the join keys
    span_schema = load_schema("span")
    spans = [json.loads(line) for line in (tmp_path / "spans.jsonl").read_text().splitlines()]
    for span in spans:
        jsonschema.validate(span, span_schema)
    root = next(s for s in spans if s["parent_span_id"] is None)
    assert root["trace_id"] == io.run_id
    assert root["harness.outcome"] == "success"
    assert root["harness.replay.cassette_ref"].endswith(f"{io.run_id}.cassette.jsonl")


def test_recording_tapes_the_crash_and_reraises(tmp_path) -> None:
    task = {"invoice_id": "INV-004"}  # poisoned: float("1,1xx.xx") raises
    rec = RecordingIO(tmp_path, task_input=task, task_signature="demo/x")
    with pytest.raises(ValueError, match="could not convert"), rec as io:
        demo_agent(io, task)
    events = read_cassette(tmp_path / "cassettes" / f"{rec.run_id}.cassette.jsonl")
    final = next(e for e in events if e.decision_name == "__final__")
    assert final.output["ok"] is False
    assert final.output["error"]["type"] == "ValueError"
    root = json.loads((tmp_path / "spans.jsonl").read_text().splitlines()[0])
    assert root["harness.outcome"] == "failure"


def test_recorded_spans_feed_ingest(tmp_path) -> None:
    from keen_touchstone.adapters.otel_traces import read_spans_jsonl, trials_from_traces
    from keen_touchstone.demo.replay_agent import record_demo_runs

    record_demo_runs(tmp_path, invoices=4, runs_per_invoice=2)
    ingest = trials_from_traces(read_spans_jsonl(tmp_path / "spans.jsonl"))
    assert len(ingest.tasks) == 4
    assert all(t.n == 2 for t in ingest.tasks)
    # INV-004 is poisoned -> always fails; others always pass
    by_key = {t.task_key: t.c for t in ingest.tasks}
    assert by_key["demo/reconcile/INV-004"] == 0
    assert by_key["demo/reconcile/INV-001"] == 2


# ----------------------------------------------------------------- P2: replay


def _record_one(tmp_path, invoice="INV-001"):
    task = {"invoice_id": invoice}
    rec = RecordingIO(tmp_path, task_input=task, task_signature="demo/x")
    try:
        with rec as io:
            result = demo_agent(io, task)
            io.finish(result)
    except ValueError:
        pass
    return tmp_path / "cassettes" / f"{rec.run_id}.cassette.jsonl"


def test_replay_faithful_success(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    report = replay_run(cassette, demo_agent)
    assert report.faithful, report.verdict
    assert "REPLAY FAITHFUL" in report.verdict
    assert report.replayed_final == report.recorded_final
    assert report.unconsumed == {}


def test_replay_reproduces_the_same_crash(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-004")
    report = replay_run(cassette, demo_agent)
    assert report.faithful, report.verdict
    assert "same failure" in report.verdict
    assert report.replayed_final["error"]["type"] == "ValueError"


def test_replay_serves_recorded_clock(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    events = read_cassette(cassette)
    recorded_stamp = next(e for e in events if e.decision_name == "__clock__").output
    report = replay_run(cassette, demo_agent)
    assert report.replayed_final["result"]["checked_at"] == str(recorded_stamp)


def test_changed_harness_diverges_at_the_right_step(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    report = replay_run(cassette, demo_agent_v2)
    assert not report.faithful
    assert "DIVERGED" in report.verdict
    assert "step 1" in report.verdict  # step 0 = task input; step 1 = first llm_call
    assert "first difference at char" in report.verdict


def test_tampered_output_breaks_faithfulness(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    lines = cassette.read_text().splitlines()
    doctored = []
    for line in lines:
        data = json.loads(line)
        if data["kind"] == "tool_call" and data["metadata"].get("tool_id") == "ledger_lookup":
            data["output"]["amount"] = 999999.0  # rewrite history
        doctored.append(json.dumps(data))
    cassette.write_text("\n".join(doctored) + "\n")
    report = replay_run(cassette, demo_agent)
    assert not report.faithful
    # downstream inputs no longer match: caught as divergence, not silently accepted
    assert "DIVERGED" in report.verdict


def test_truncated_tape_exhausts_loudly(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    lines = cassette.read_text().splitlines()
    # drop the second llm_call but keep __final__ (schema-valid, step-monotonic)
    kept = [
        line for line in lines
        if not (json.loads(line)["kind"] == "llm_call" and "Summarize" in json.dumps(json.loads(line)["input"]))
    ]
    cassette.write_text("\n".join(kept) + "\n")
    report = replay_run(cassette, demo_agent)
    assert not report.faithful
    assert "CassetteExhausted" in (report.divergence or "")
    assert "Refusing to fall back" in report.verdict


def test_short_circuit_harness_leaves_unconsumed_events(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")

    def lazy_agent(io, task_input):
        io.llm_call(
            f"Plan how to reconcile invoice {task_input['invoice_id']}.",
            "demo/planner-v1", None,
        )
        return None  # walks away mid-run

    report = replay_run(cassette, lazy_agent)
    assert not report.faithful
    assert report.unconsumed.get("tool_call") == 2
    assert "unconsumed" in report.verdict or "outcome differs" in report.verdict


def test_replay_never_calls_live_functions(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")
    calls = {"n": 0}

    def spy_llm(prompt):
        calls["n"] += 1
        return "LIVE"

    def spying_agent(io, task_input):
        return demo_agent(io, task_input)

    # demo_agent passes its own call_fns, but ReplayIO must never invoke them
    import keen_touchstone.demo.replay_agent as mod
    original = mod.scripted_llm
    mod.scripted_llm = spy_llm
    try:
        replay_run(cassette, spying_agent)
    finally:
        mod.scripted_llm = original
    assert calls["n"] == 0  # zero network, zero live calls


def test_wrong_identity_diverges(tmp_path) -> None:
    cassette = _record_one(tmp_path, "INV-001")

    def wrong_model_agent(io, task_input):
        io.llm_call(
            f"Plan how to reconcile invoice {task_input['invoice_id']}.",
            "demo/planner-v2",  # different model id, same prompt
            None,
        )

    report = replay_run(cassette, wrong_model_agent)
    assert not report.faithful
    assert "different model/tool" in (report.divergence or "")


# -------------------------------------------- round-3 attack-battery regressions


def test_mixed_key_dict_input_does_not_crash_the_recorder(tmp_path) -> None:
    """[battery A1] The flight recorder must never crash the plane: mixed-type
    dict keys used to raise TypeError inside the writer and blame the agent."""

    def agent(io, task_input):
        return io.tool_call("t", {1: "a", "b": 2}, lambda x: "ok")

    with RecordingIO(tmp_path, task_input={}) as io:
        io.finish(agent(io, {}))
    report = replay_run(
        next((tmp_path / "cassettes").glob("*.jsonl")), agent
    )
    assert report.faithful, report.verdict


def test_set_inputs_canonicalize_deterministically(tmp_path) -> None:
    """[battery A2] Sets are taped as sorted lists — str(set) ordering is
    PYTHONHASHSEED-dependent across processes and caused false divergence."""
    from keen_touchstone.cassette.io import jsonable

    assert jsonable({"items": {"gamma", "alpha", "beta"}}) == {
        "items": ["alpha", "beta", "gamma"]
    }
    assert jsonable(frozenset([3, 1, 2])) == [1, 2, 3]

    def agent(io, task_input):
        return io.tool_call("t", {"tags": {"b", "a"}}, lambda x: sorted(x["tags"]))

    with RecordingIO(tmp_path, task_input={}) as io:
        io.finish(agent(io, {}))
    report = replay_run(next((tmp_path / "cassettes").glob("*.jsonl")), agent)
    assert report.faithful, report.verdict


def test_object_reprs_in_result_compare_modulo_addresses(tmp_path) -> None:
    """[battery A3] A faithful replay whose result holds a non-serializable
    object must not falsely diverge on the memory address in its repr —
    and the verdict says the comparison was masked."""

    class Opaque:
        pass

    def agent(io, task_input):
        io.tool_call("t", {"k": 1}, lambda x: "v")
        return {"handle": Opaque()}

    with RecordingIO(tmp_path, task_input={}) as io:
        io.finish(agent(io, {}))
    report = replay_run(next((tmp_path / "cassettes").glob("*.jsonl")), agent)
    assert report.faithful, report.verdict
    assert "modulo memory addresses" in report.verdict


def test_corrupted_clock_reports_tape_problem_not_agent_crash(tmp_path) -> None:
    """[battery B] A garbage __clock__ value is a tape problem — Divergence,
    never misclassified as the agent crashing."""

    def agent(io, task_input):
        return {"t": io.now().isoformat()}

    with RecordingIO(tmp_path, task_input={}) as io:
        io.finish(agent(io, {}))
    cassette = next((tmp_path / "cassettes").glob("*.jsonl"))
    doctored = []
    for line in cassette.read_text().splitlines():
        data = json.loads(line)
        if data["metadata"].get("decision") == "__clock__":
            data["output"] = "not-a-timestamp"
        doctored.append(json.dumps(data))
    cassette.write_text("\n".join(doctored) + "\n")
    report = replay_run(cassette, agent)
    assert not report.faithful
    assert report.divergence is not None
    assert "tape problem" in report.divergence


def test_recording_after_finish_is_a_clear_error(tmp_path) -> None:
    """[battery D] io.* after finish() gets a plain-language refusal, not
    'I/O operation on closed file'."""
    with RecordingIO(tmp_path, task_input={}) as io:
        io.finish("done")
    with pytest.raises(ValueError, match="already finished"):
        io.llm_call("late", "m", lambda p: "x")


def test_reserved_decision_names_rejected(tmp_path) -> None:
    task = {"invoice_id": "INV-001"}
    with RecordingIO(tmp_path, task_input=task) as io:
        with pytest.raises(ValueError, match="reserved"):
            io.decision("__final__", 1)
        io.finish(None)
