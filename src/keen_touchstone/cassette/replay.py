"""ReplayIO: re-execute the harness against the tape. Zero network, loud edges.

The seven primitives, enforced:

1. Per-kind cursors (llm_call / tool_call / named decisions / clock reads).
2. Input matching — every replayed call's input must equal the recorded one
   (canonical JSON); a mismatch raises Divergence naming the step, with a
   compact diff. A changed harness diverges BY DESIGN — the error tells you
   exactly where the old and new logic part ways.
3. Identity validation — model_id / tool_id must match the recording.
4. Exhaustion detection — calling past the end raises CassetteExhausted.
   NEVER a silent fallback to live systems (that would invalidate the debug
   session and possibly cost money or cause side effects).
5. Clock virtualization — ``now()`` serves the recorded instants.
6. Self-contained tape — task input and final outcome ride the cassette, so
   replay needs the file and the entry function, nothing else.
7. Unconsumed-events report — events the replay never asked for are the
   other half of control-flow drift, and they are reported, not ignored.

What replay proves: your orchestration logic, given the frozen model/tool
outputs, behaves identically — same result or same crash. What it cannot
prove: that the model would answer the same today. Stated here, in the
README, and in the thesis; pretending otherwise is how replay tools lie.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import (
    DECISION_CLOCK,
    DECISION_FINAL,
    DECISION_TASK_INPUT,
    TraceEvent,
    jsonable,
    read_cassette,
)
from .record import error_payload


class Divergence(Exception):
    """The replayed harness did something the recording did not."""


class CassetteExhausted(Exception):
    """The replayed harness asked for more than the recording holds."""


def canonical(value: Any) -> str:
    """Same normalization as the writer (jsonable) — record and replay must
    serialize identically or matching is meaningless."""
    return json.dumps(jsonable(value), sort_keys=True, default=str)


_ADDRESS_RE = re.compile(r"0x[0-9a-f]{6,}")


def _mask_addresses(text: str) -> str:
    return _ADDRESS_RE.sub("0xADDR", text)


def _diff_snippet(expected: str, got: str, context: int = 40) -> str:
    if expected == got:
        return "(identical serializations)"
    i = next(
        (k for k, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b),
        min(len(expected), len(got)),
    )
    lo = max(0, i - context)
    return (
        f"first difference at char {i}: "
        f"recorded …{expected[lo:i + context]}… vs replayed …{got[lo:i + context]}…"
    )


@dataclass(frozen=True)
class ReplayReport:
    faithful: bool
    run_id: str
    verdict: str  # human-readable one-liner
    recorded_final: dict[str, Any]
    replayed_final: dict[str, Any]
    divergence: str | None
    unconsumed: dict[str, int]  # kind/queue -> leftover count
    steps_replayed: int


class ReplayIO:
    """Drop-in for RecordingIO on the replay side: same seam, zero network."""

    def __init__(self, cassette_path: str | Path):
        self.path = Path(cassette_path)
        events = read_cassette(self.path)
        self.run_id = events[0].run_id

        self._llm: list[TraceEvent] = []
        self._tool: list[TraceEvent] = []
        self._clock: list[TraceEvent] = []
        self._decisions: list[TraceEvent] = []
        self.task_input: Any = None
        self._has_task_input = False
        self.recorded_final: dict[str, Any] | None = None
        for ev in events:
            name = ev.decision_name
            if ev.kind == "llm_call":
                self._llm.append(ev)
            elif ev.kind == "tool_call":
                self._tool.append(ev)
            elif name == DECISION_CLOCK:
                self._clock.append(ev)
            elif name == DECISION_TASK_INPUT:
                self.task_input = ev.output
                self._has_task_input = True
            elif name == DECISION_FINAL:
                self.recorded_final = ev.output
            else:
                self._decisions.append(ev)
        if not self._has_task_input:
            raise ValueError(f"{self.path}: no {DECISION_TASK_INPUT} event — not a complete tape")
        self._cursors = {"llm_call": 0, "tool_call": 0, "clock": 0, "decision": 0}
        self.steps_replayed = 0

    # ------------------------------------------------------------- the seam

    def llm_call(self, prompt: Any, model_id: str, call_fn=None) -> Any:
        ev = self._next(self._llm, "llm_call")
        self._match_identity(ev, "model_id", model_id)
        self._match_input(ev, prompt)
        return ev.output

    def tool_call(self, tool_id: str, tool_input: Any, call_fn=None) -> Any:
        ev = self._next(self._tool, "tool_call")
        self._match_identity(ev, "tool_id", tool_id)
        self._match_input(ev, tool_input)
        return ev.output

    def decision(self, name: str, value: Any = None) -> Any:
        """On replay the RECORDED value wins — that is what makes harness-side
        nondeterminism deterministic again. The live ``value`` is ignored."""
        if name.startswith("__"):
            raise ValueError(f"decision name {name!r} is reserved")
        ev = self._next(self._decisions, "decision")
        recorded_name = ev.decision_name
        if recorded_name != name:
            raise Divergence(
                f"step {ev.step_id}: recording expected decision {recorded_name!r}, "
                f"replay asked for {name!r} — control flow has drifted"
            )
        return ev.output

    def now(self) -> datetime:
        ev = self._next(self._clock, "clock read (io.now())")
        try:
            return datetime.fromisoformat(str(ev.output))
        except ValueError as err:
            raise Divergence(
                f"step {ev.step_id}: taped clock value {ev.output!r} is not a timestamp — "
                "corrupted or hand-edited tape (this is a tape problem, not an agent crash)"
            ) from err

    # ------------------------------------------------------------- plumbing

    def _next(self, queue: list[TraceEvent], label: str) -> TraceEvent:
        key = "clock" if label.startswith("clock") else (
            "decision" if label == "decision" else label
        )
        idx = self._cursors[key]
        if idx >= len(queue):
            raise CassetteExhausted(
                f"replay asked for {label} #{idx + 1} but the recording holds only "
                f"{len(queue)} — the harness now makes calls the original run never made. "
                "Refusing to fall back to live systems."
            )
        self._cursors[key] = idx + 1
        self.steps_replayed += 1
        return queue[idx]

    def _match_identity(self, ev: TraceEvent, key: str, got: str) -> None:
        recorded = ev.metadata.get(key)
        if recorded != got:
            raise Divergence(
                f"step {ev.step_id}: recorded {key}={recorded!r} but replay used {got!r} — "
                "a different model/tool is a different run"
            )

    def _match_input(self, ev: TraceEvent, got: Any) -> None:
        rec_c, got_c = canonical(ev.input), canonical(got)
        if rec_c != got_c:
            raise Divergence(
                f"step {ev.step_id} ({ev.kind}): replay input differs from the recording — "
                f"{_diff_snippet(rec_c, got_c)}. Either the harness logic changed, or it "
                "depends on nondeterminism not routed through io.now()/io.decision()."
            )

    def unconsumed(self) -> dict[str, int]:
        return {
            key: len(queue) - self._cursors[key]
            for key, queue in (
                ("llm_call", self._llm),
                ("tool_call", self._tool),
                ("clock", self._clock),
                ("decision", self._decisions),
            )
            if len(queue) - self._cursors[key] > 0
        }


def replay_run(cassette_path: str | Path, entry_fn) -> ReplayReport:
    """Re-execute ``entry_fn(io, task_input)`` against the tape and reconcile
    the outcome with the recorded one."""
    io = ReplayIO(cassette_path)
    if io.recorded_final is None:
        raise ValueError(
            f"{cassette_path}: no {DECISION_FINAL} event — the recording never finished "
            "(crashed before RecordingIO could tape the outcome?)"
        )

    divergence: str | None = None
    try:
        result = entry_fn(io, io.task_input)
        replayed_final: dict[str, Any] = {"ok": True, "result": result}
    except (Divergence, CassetteExhausted) as err:
        divergence = f"{type(err).__name__}: {err}"
        replayed_final = {"ok": False, "error": error_payload(err)}
    except Exception as err:  # the agent crashed — possibly exactly as recorded
        replayed_final = {"ok": False, "error": error_payload(err)}

    unconsumed = io.unconsumed()
    rec_c, got_c = canonical(io.recorded_final), canonical(replayed_final)
    outcome_matches = rec_c == got_c
    masked_match = False
    if not outcome_matches and _mask_addresses(rec_c) == _mask_addresses(got_c):
        # Non-serializable objects carry process-local memory addresses in
        # their reprs; identical-modulo-address outcomes are the same value
        # as far as any tape can tell. Said out loud in the verdict.
        outcome_matches = True
        masked_match = True
    faithful = divergence is None and outcome_matches and not unconsumed

    if divergence is not None:
        verdict = f"DIVERGED — {divergence}"
    elif not outcome_matches:
        verdict = (
            "DIVERGED — replay completed but the outcome differs from the recording "
            f"(recorded {canonical(io.recorded_final)[:120]} vs replayed "
            f"{canonical(replayed_final)[:120]})"
        )
    elif unconsumed:
        verdict = (
            f"DIVERGED — outcome matches but the recording holds unconsumed events "
            f"{unconsumed}: the replayed harness took a shorter path than the original"
        )
    else:
        outcome = "same result" if replayed_final.get("ok") else (
            f"same failure ({replayed_final['error']['type']}: "
            f"{replayed_final['error']['message']})"
        )
        verdict = f"REPLAY FAITHFUL — {outcome}, {io.steps_replayed} step(s), zero network"
        if masked_match:
            verdict += (
                " (non-serializable object reprs in the result compared modulo memory "
                "addresses — return JSON-able results for exact reconciliation)"
            )

    return ReplayReport(
        faithful=faithful,
        run_id=io.run_id,
        verdict=verdict,
        recorded_final=io.recorded_final,
        replayed_final=replayed_final,
        divergence=divergence,
        unconsumed=unconsumed,
        steps_replayed=io.steps_replayed,
    )
