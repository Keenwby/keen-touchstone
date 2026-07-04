"""Phase 3: the cassette — record an agent run, replay it deterministically.

The flight recorder for agent runs (Artifact B). Recording captures every
model call, tool call, clock read, and named decision in order; replay
re-executes YOUR orchestration logic against the frozen outputs — zero
network, zero API cost, same failure.

Honest scope (stated everywhere, verbatim from the design thesis): replay
reproduces the harness orchestration logic given frozen model/tool outputs —
NOT model reproducibility. Hosted-API nondeterminism is unfixable from
outside; a changed harness diverges by design, loudly, at the exact step.
"""

from .io import (
    DECISION_CLOCK,
    DECISION_FINAL,
    DECISION_TASK_INPUT,
    CassetteWriter,
    TraceEvent,
    read_cassette,
)
from .record import RecordingIO
from .replay import CassetteExhausted, Divergence, ReplayIO, ReplayReport, replay_run

__all__ = [
    "DECISION_CLOCK",
    "DECISION_FINAL",
    "DECISION_TASK_INPUT",
    "CassetteExhausted",
    "CassetteWriter",
    "Divergence",
    "RecordingIO",
    "ReplayIO",
    "ReplayReport",
    "TraceEvent",
    "read_cassette",
    "replay_run",
]
