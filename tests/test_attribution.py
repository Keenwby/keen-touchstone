"""Attribution math: hand goldens, the 2×2 identity, honesty paths, diagnose rules."""

import pytest

from keen_touchstone.attribution import DIAGNOSE_DISCLAIMER, decompose, diagnose_cassette
from keen_touchstone.cassette import RecordingIO

# ------------------------------------------------------------- decompose math


def _cells(n_tasks=24, n=2, model_fail=(), harness_fail=(), overlap=()):
    """Deterministic cells: tasks fail (all n trials) per the designed sets."""
    model_set, harness_set, both_set = set(model_fail), set(harness_fail), set(overlap)

    def cell(model_fixed: bool, harness_fixed: bool):
        tasks = {}
        for i in range(1, n_tasks + 1):
            fails = False
            if i in both_set:
                fails = not (model_fixed and harness_fixed)  # needs BOTH fixes
            elif i in model_set:
                fails = not model_fixed
            elif i in harness_set:
                fails = not harness_fixed
            tasks[f"t{i:02d}"] = (n, 0 if fails else n)
        return tasks

    return {
        "baseline": cell(False, False),
        "model_swap": cell(True, False),
        "harness_swap": cell(False, True),
        "both_swap": cell(True, True),
    }


def test_hand_golden_with_overlap() -> None:
    """24 tasks: 6 model-only, 6 harness-only, 1 needs-both. Exact shares:
    model 6/24=25pp (the overlap task does NOT recover under model swap alone),
    harness 6/24=25pp, interaction +1/24 (the needs-both task), irreducible 0."""
    cells = _cells(model_fail=range(1, 12, 2), harness_fail=range(2, 13, 2), overlap=(15,))
    result = decompose(**cells, seed=1)
    assert result.baseline_failure == pytest.approx(13 / 24)
    assert result.model_share.delta == pytest.approx(6 / 24)
    assert result.harness_share.delta == pytest.approx(6 / 24)
    assert result.interaction.delta == pytest.approx(+1 / 24)
    assert result.irreducible_failure == pytest.approx(0.0)
    # 2x2 identity (also asserted inside decompose)
    total = (result.model_share.delta + result.harness_share.delta
             + result.interaction.delta + result.irreducible_failure)
    assert total == pytest.approx(result.baseline_failure)
    # 6 nonzero paired deltas per factor -> p ≈ 2/64 (sampled path: 24 deltas > exact limit)
    assert result.model_share.significant
    assert result.model_share.p_value == pytest.approx(2 / 64, abs=0.02)
    assert result.harness_share.significant
    assert "BOTH fixes" in result.sentence()


def test_three_cells_is_honest_about_interaction() -> None:
    cells = _cells(model_fail=(1, 3, 5, 7, 9, 11), harness_fail=(2, 4, 6, 8, 10, 12))
    result = decompose(cells["baseline"], cells["model_swap"], cells["harness_swap"])
    assert result.interaction is None
    assert result.irreducible_failure is None
    assert not result.has_both_cell
    assert "not estimable" in result.sentence()


def test_negative_share_is_a_finding() -> None:
    """The 'better' model actually fails MORE tasks — share goes negative and
    the notes say it's a finding, not an error."""
    baseline = {f"t{i}": (4, 4) for i in range(1, 11)}  # nothing fails
    worse_model = dict(baseline)
    for i in (1, 2, 3, 4, 5, 6):
        worse_model[f"t{i}"] = (4, 0)
    harness_swap = dict(baseline)
    result = decompose(baseline, worse_model, harness_swap)
    assert result.model_share.delta == pytest.approx(-0.6)
    assert result.model_share.significant  # 6 nonzero deltas, all negative
    assert any("WORSE" in n for n in result.notes)


def test_mismatched_task_sets_drop_honestly() -> None:
    cells = _cells(model_fail=(1, 3, 5, 7, 9, 11))
    cells["model_swap"].pop("t01")
    cells["baseline"]["extra"] = (2, 2)
    result = decompose(**cells, seed=1)
    assert result.n_shared_tasks == 23
    assert result.dropped_per_cell["baseline"] == 2  # its own "extra" + the shared drop of t01
    assert any("dropped" in n for n in result.notes)
    with pytest.raises(ValueError, match="no tasks shared"):
        decompose({"a": (2, 2)}, {"b": (2, 2)}, {"c": (2, 2)})


def test_underpowered_flag() -> None:
    tiny = {f"t{i}": (2, 0 if i == 1 else 2) for i in range(1, 4)}
    fixed = {key: (2, 2) for key in tiny}
    result = decompose(tiny, fixed, dict(tiny))
    assert result.underpowered
    assert any("direction-only" in n for n in result.notes)


# ------------------------------------------------------------------- diagnose


def _tape_with_crash(tmp_path, agent):
    rec = RecordingIO(tmp_path, task_input={"x": 1})
    try:
        with rec as io:
            agent(io)
            io.finish("ok")
    except Exception:
        pass
    return tmp_path / "cassettes" / f"{rec.run_id}.cassette.jsonl"


def test_diagnose_tool_parse_crash_ranks_execution_then_tooling(tmp_path) -> None:
    def agent(io):
        entry = io.tool_call("ledger", {"id": 1}, lambda x: {"amount": "1,130.00"})
        float(entry["amount"])  # the classic harness bug

    report = diagnose_cassette(_tape_with_crash(tmp_path, agent))
    assert report.failed and report.error_type == "ValueError"
    assert report.died_after == "tool_call"
    layers = [h.layer for h in report.hypotheses]
    assert layers[0] == "Execution" and "Tooling" in layers
    assert report.disclaimer == DIAGNOSE_DISCLAIMER
    assert "14%" in report.disclaimer  # the ceiling is in the header, always


def test_diagnose_timeout_ranks_lifecycle(tmp_path) -> None:
    def agent(io):
        io.llm_call("plan", "m", lambda p: "reply")
        raise RuntimeError("request timed out after 30s")

    report = diagnose_cassette(_tape_with_crash(tmp_path, agent))
    assert report.hypotheses[0].layer == "Lifecycle"


def test_diagnose_context_marker(tmp_path) -> None:
    def agent(io):
        io.llm_call("plan", "m", lambda p: "reply")
        raise ValueError("prompt exceeds the model's context window; truncated")

    report = diagnose_cassette(_tape_with_crash(tmp_path, agent))
    assert report.hypotheses[0].layer == "Context"


def test_diagnose_pre_seam_crash(tmp_path) -> None:
    def agent(io):
        raise KeyError("config")

    report = diagnose_cassette(_tape_with_crash(tmp_path, agent))
    layers = [h.layer for h in report.hypotheses]
    assert layers[0] == "Execution" and "Lifecycle" in layers


def test_diagnose_refuses_successful_tapes(tmp_path) -> None:
    def agent(io):
        io.tool_call("t", {"a": 1}, lambda x: "fine")

    with RecordingIO(tmp_path, task_input={}) as io:
        agent(io)
        io.finish("done")
    cassette = next((tmp_path / "cassettes").glob("*.jsonl"))
    report = diagnose_cassette(cassette)
    assert not report.failed
    assert report.hypotheses == []
