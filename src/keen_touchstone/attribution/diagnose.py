"""ETCLOVG layer hypotheses from a failed cassette — a HINT, never a verdict.

Reads a failed run's tape and applies a small table of deterministic rules
("where did the run die, on what kind of event, with what error") to rank
which HARNESS layer is most suspect: Execution, Tooling, Context, Lifecycle,
Observability, Verification, Governance.

Why the disclaimer is part of the output header and not a footnote: the
published state of the art for step-level failure attribution is ~14%
accurate (agent-level ~53%, the Who&When benchmark) — an authoritative-
looking guess at these accuracy levels would be malpractice. This module's
output is a ranked reading list for a human, and the trustworthy instrument
is the measured A/B next door (`touchstone attribute`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from keen_touchstone.cassette.io import DECISION_FINAL, TraceEvent, read_cassette

DIAGNOSE_DISCLAIMER = (
    "RANKED HYPOTHESES, NOT A VERDICT — published step-level failure attribution tops out "
    "near 14% accuracy (agent-level ~53%, Who&When). Treat this as a reading list; the "
    "trustworthy instrument is a measured A/B (`touchstone attribute`)."
)


@dataclass(frozen=True)
class Hypothesis:
    layer: str  # Execution | Tooling | Context | Lifecycle | Observability | Verification | Governance
    reason: str


@dataclass(frozen=True)
class DiagnosisReport:
    run_id: str
    failed: bool
    error_type: str | None
    error_message: str | None
    died_after: str | None  # kind of the last seam event before the crash
    hypotheses: list[Hypothesis] = field(default_factory=list)
    disclaimer: str = DIAGNOSE_DISCLAIMER


_PARSE_ERRORS = {"ValueError", "TypeError", "KeyError", "AttributeError", "JSONDecodeError",
                 "IndexError"}
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline")
_CONTEXT_MARKERS = ("context", "token limit", "max tokens", "too long", "truncat")


def diagnose_cassette(cassette_path: str | Path) -> DiagnosisReport:
    events = read_cassette(cassette_path)
    run_id = events[0].run_id
    final = next((e for e in events if e.decision_name == DECISION_FINAL), None)
    if final is None:
        raise ValueError(f"{cassette_path}: no __final__ event — incomplete tape")
    outcome = final.output or {}
    if outcome.get("ok"):
        # r5-F4: "ok" on the tape means NO HARNESS CRASH — the task itself may
        # still have failed its grading (invisible from the tape). Never claim
        # the run "succeeded"; the CLI phrasing follows this distinction.
        return DiagnosisReport(
            run_id=run_id, failed=False, error_type=None, error_message=None,
            died_after=None,
            hypotheses=[],
        )

    error = outcome.get("error") or {}
    err_type = str(error.get("type") or "")
    err_message = str(error.get("message") or "")
    seam = [e for e in events if e.kind in ("llm_call", "tool_call")]
    last: TraceEvent | None = seam[-1] if seam else None
    died_after = last.kind if last else None

    hypotheses = _rules(err_type, err_message.lower(), last)
    return DiagnosisReport(
        run_id=run_id,
        failed=True,
        error_type=err_type or None,
        error_message=err_message or None,
        died_after=died_after,
        hypotheses=hypotheses,
    )


def _rules(err_type: str, message: str, last: TraceEvent | None) -> list[Hypothesis]:
    ranked: list[Hypothesis] = []

    def add(layer: str, reason: str) -> None:
        if all(h.layer != layer for h in ranked):
            ranked.append(Hypothesis(layer, reason))

    # r5-F5: the tool-output error-marker check must run BEFORE the generic
    # tool_call arms — both arms add Tooling, and add() dedupes by layer, so
    # this rule was dead code (its most useful signal could never surface)
    if last is not None and last.kind == "tool_call":
        tool_output = json.dumps(last.output, default=str).lower()
        # [fixver N4] bare "failed" false-positived on benign test-runner
        # summaries ("0 failed, 12 passed") — a count immediately before the
        # word reads as a tally, not a marker; keep the quoted-"error" check
        failed_marker = re.search(r"(?<![0-9])(?<![0-9]\s)failed", tool_output)
        if '"error"' in tool_output or failed_marker:
            add("Tooling", "the last tool output itself carries error markers")

    if any(marker in message for marker in _TIMEOUT_MARKERS) or "Timeout" in err_type:
        add("Lifecycle", "the failure mentions a timeout/deadline — run lifecycle management")
    if any(marker in message for marker in _CONTEXT_MARKERS):
        add("Context", "the failure mentions context/token limits or truncation")

    if last is not None and last.kind == "tool_call":
        if err_type in _PARSE_ERRORS:
            add("Execution",
                f"the run died in harness code right after a tool call ({err_type} — likely "
                "the harness parsing/handling the tool's output)")
            add("Tooling",
                "…or the tool returned an unexpected shape the harness never promised to handle")
        else:
            add("Tooling", "the run died right after a tool call")
            add("Execution", "harness-side handling of that tool result is the other suspect")
    elif last is not None and last.kind == "llm_call":
        if err_type in _PARSE_ERRORS:
            add("Execution",
                f"the run died parsing/handling a model reply ({err_type})")
            add("Verification",
                "no output validation caught the malformed reply before the harness consumed it")
        else:
            add("Execution", "the run died in harness code right after a model call")
    elif last is None:
        add("Execution", "the run died before reaching any model or tool call — pure harness code")
        add("Lifecycle", "…or setup/initialization failed")

    if not ranked:
        add("Execution", "insufficient signal — defaulting to the harness's own code path")
    if len(ranked) < 3:
        add("Verification",
            "a final-state check would have converted this crash into a graded failure")
    return ranked[:3]
