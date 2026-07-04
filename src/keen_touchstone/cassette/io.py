"""Cassette file I/O: the append-only ground truth of one agent run.

One JSONL line per TraceEvent, validated against the canonical packaged
``cassette.schema.json`` on write AND on read (a tampered or hand-edited
line fails loudly with its line number — never silently skipped).

Reserved decision names (stored in ``metadata.decision``):

- ``__task_input__`` — the task the run was asked to do (makes replay
  self-contained: the cassette alone is enough to re-run).
- ``__clock__``      — a wall-clock read via ``io.now()`` (clock
  virtualization: replay serves the recorded time).
- ``__final__``      — how the run ended: ``{"ok": bool, "result": ...}`` or
  ``{"ok": false, "error": {"type": ..., "message": ...}}`` (crashes are
  recorded, not lost — reproducing the failure is the point).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from keen_touchstone.artifacts import load_schema

DECISION_TASK_INPUT = "__task_input__"
DECISION_CLOCK = "__clock__"
DECISION_FINAL = "__final__"


def jsonable(value: Any) -> Any:
    """Normalize a value for taping/matching — deterministically.

    Found by the round-3 attack battery: raw ``json.dumps(sort_keys=True,
    default=str)`` (a) crashes on mixed-type dict keys (the RECORDER killing
    the run and blaming the agent), and (b) serializes sets via ``str()``,
    whose ordering is PYTHONHASHSEED-dependent — record in one process,
    replay in another, false divergence. So: dict keys become strings before
    sorting; sets/frozensets become lists sorted by their canonical form;
    tuples become lists (JSON does that anyway); everything else falls back
    to ``str()``.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(
            (jsonable(v) for v in value), key=lambda x: json.dumps(x, sort_keys=True, default=str)
        )
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    step_id: int
    timestamp: str
    kind: str  # llm_call | tool_call | decision
    input: Any
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }

    @property
    def decision_name(self) -> str | None:
        return self.metadata.get("decision") if self.kind == "decision" else None


def _validate_event_dict(data: dict[str, Any], where: str) -> None:
    try:
        jsonschema.validate(data, load_schema("cassette"), cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError as err:
        raise ValueError(f"{where}: not a valid cassette TraceEvent — {err.message}") from err


class CassetteWriter:
    """Append-only writer; every line schema-validated and flushed as written
    (a crash mid-run must still leave a replayable prefix on disk)."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")  # noqa: SIM115 — held across appends; class is the context manager
        self._n_written = 0

    def append(self, event: TraceEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError(
                f"event run_id {event.run_id!r} does not match cassette run_id {self.run_id!r}"
            )
        data = jsonable(event.to_dict())
        _validate_event_dict(data, f"{self.path} (while recording step {event.step_id})")
        self._fh.write(json.dumps(data, sort_keys=True, default=str) + "\n")
        self._fh.flush()
        self._n_written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> CassetteWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_cassette(path: str | Path) -> list[TraceEvent]:
    """Read and validate a whole cassette. Errors name the line; step_ids must
    be strictly increasing and all events must share one run_id — a spliced or
    reordered tape is refused, not guessed at."""
    path = Path(path)
    events: list[TraceEvent] = []
    run_id: str | None = None
    last_step = -1
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({err.msg})") from err
            _validate_event_dict(data, f"{path}:{lineno}")
            if run_id is None:
                run_id = data["run_id"]
            elif data["run_id"] != run_id:
                raise ValueError(
                    f"{path}:{lineno}: run_id {data['run_id']!r} differs from {run_id!r} — "
                    "one cassette holds exactly one run"
                )
            if data["step_id"] <= last_step:
                raise ValueError(
                    f"{path}:{lineno}: step_id {data['step_id']} is not strictly increasing "
                    f"(previous {last_step}) — reordered or spliced tape refused"
                )
            last_step = data["step_id"]
            events.append(
                TraceEvent(
                    run_id=data["run_id"],
                    step_id=data["step_id"],
                    timestamp=data["timestamp"],
                    kind=data["kind"],
                    input=data["input"],
                    output=data["output"],
                    metadata=data.get("metadata", {}),
                )
            )
    if not events:
        raise ValueError(f"{path}: empty cassette")
    return events
