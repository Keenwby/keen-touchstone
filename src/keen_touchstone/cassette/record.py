"""RecordingIO: run the agent for real, and tape everything it touches.

The seam is explicit — the agent function accepts an ``io`` object and routes
every side of nondeterminism through it (``llm_call``, ``tool_call``,
``now()``, ``decision``). No SDK monkey-patching magic: what goes through the
seam replays; what bypasses it is invisible (and the divergence detector will
say so at replay time).

Recording also co-emits Spans (same trace_id, same step_id, span.schema.json
vocabulary) into a shared ``spans.jsonl`` — so recorded runs feed straight
into ``touchstone ingest`` and the reliability stats. One run, three joined
artifacts: Cassette + Span + (downstream) ReliabilityAggregate.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import (
    DECISION_CLOCK,
    DECISION_FINAL,
    DECISION_TASK_INPUT,
    CassetteWriter,
    TraceEvent,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def error_payload(exc: BaseException) -> dict[str, str]:
    """Stable, comparable representation of a crash (no memory addresses)."""
    return {"type": type(exc).__name__, "message": str(exc)}


class RecordingIO:
    """Record one agent run. Use as a context manager so crashes are taped:

        with RecordingIO(out_dir, task_input={...}, task_signature="ops/x") as io:
            result = my_agent(io, io.task_input)
            io.finish(result)

    If the agent raises, ``__exit__`` records ``__final__`` with the error and
    re-raises — the cassette then reproduces the failure, which is the point.
    """

    def __init__(
        self,
        out_dir: str | Path,
        task_input: Any,
        task_signature: str | None = None,
        run_id: str | None = None,
        agent_config_hash: str = "recorded-agent",
    ):
        self.out_dir = Path(out_dir)
        self.run_id = run_id or uuid.uuid4().hex
        self.task_input = task_input
        self.task_signature = task_signature
        self.agent_config_hash = agent_config_hash
        self._step = 0
        self._spans: list[dict[str, Any]] = []
        self._models_seen: list[str] = []
        self._finished = False
        self._writer = CassetteWriter(
            self.out_dir / "cassettes" / f"{self.run_id}.cassette.jsonl", self.run_id
        )
        self._record("decision", DECISION_TASK_INPUT, task_input, {"decision": DECISION_TASK_INPUT})

    # ------------------------------------------------------------- the seam

    def _ensure_recording(self) -> None:
        """Guard BEFORE any live call: a seam call after finish() must not
        execute the real side effect and then fail to tape it (independent
        round-3 finding — the off-tape side effect is the dangerous half)."""
        if self._finished:
            raise ValueError(
                "recording already finished — io.* calls after finish() cannot be taped "
                "(and the live call was NOT executed)"
            )

    def llm_call(self, prompt: Any, model_id: str, call_fn) -> Any:
        self._ensure_recording()
        output = call_fn(prompt)
        step = self._record("llm_call", prompt, output, {"model_id": model_id})
        self._models_seen.append(model_id)
        self._spans.append(
            {
                "trace_id": self.run_id,
                "span_id": f"s{step:04d}",
                "parent_span_id": "root",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model_id,
                "harness.step_id": step,
                "harness.agent_config_hash": self.agent_config_hash,
            }
        )
        return output

    def tool_call(self, tool_id: str, tool_input: Any, call_fn) -> Any:
        self._ensure_recording()
        output = call_fn(tool_input)
        step = self._record("tool_call", tool_input, output, {"tool_id": tool_id})
        self._spans.append(
            {
                "trace_id": self.run_id,
                "span_id": f"s{step:04d}",
                "parent_span_id": "root",
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_id,
                "harness.step_id": step,
                "harness.agent_config_hash": self.agent_config_hash,
            }
        )
        return output

    def decision(self, name: str, value: Any) -> Any:
        """Tape harness-side nondeterminism (an RNG draw, a sampled choice) so
        replay is deterministic. Returns the value unchanged when recording."""
        if name.startswith("__"):
            raise ValueError(f"decision name {name!r} is reserved")
        self._ensure_recording()
        self._record("decision", name, value, {"decision": name})
        return value

    def now(self) -> datetime:
        """Clock virtualization, the explicit way: route wall-clock reads here
        and replay serves the recorded instant."""
        stamp = _now_iso()
        self._record("decision", DECISION_CLOCK, stamp, {"decision": DECISION_CLOCK})
        return datetime.fromisoformat(stamp)

    # ------------------------------------------------------------ lifecycle

    def finish(self, result: Any = None) -> None:
        self._close_out({"ok": True, "result": result})

    def _close_out(self, final: dict[str, Any]) -> None:
        if self._finished:
            return
        self._record("decision", DECISION_FINAL, final, {"decision": DECISION_FINAL})
        self._finished = True  # after taping __final__ — the guard blocks only post-finish calls
        self._writer.close()
        self._emit_spans(ok=bool(final.get("ok")))

    def __enter__(self) -> RecordingIO:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._close_out({"ok": False, "error": error_payload(exc)})
            return None  # re-raise
        if not self._finished:
            self.finish(None)
        return None

    # ------------------------------------------------------------ internals

    def _record(self, kind: str, input_: Any, output: Any, metadata: dict[str, Any]) -> int:
        if self._finished:
            raise ValueError(
                "recording already finished — io.* calls after finish() cannot be taped "
                "(and would make the cassette lie about the run)"
            )
        step = self._step
        self._step += 1
        self._writer.append(
            TraceEvent(
                run_id=self.run_id,
                step_id=step,
                timestamp=_now_iso(),
                kind=kind,
                input=input_,
                output=output,
                metadata=metadata,
            )
        )
        return step

    def _emit_spans(self, ok: bool) -> None:
        import json

        model = self._models_seen[0] if self._models_seen else "unknown"
        root: dict[str, Any] = {
            "trace_id": self.run_id,
            "span_id": "root",
            "parent_span_id": None,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.request.model": model,
            "harness.step_id": 0,
            "harness.agent_config_hash": self.agent_config_hash,
            "harness.outcome": "success" if ok else "failure",
            "harness.replay.cassette_ref": str(
                Path("cassettes") / f"{self.run_id}.cassette.jsonl"
            ),
        }
        if self.task_signature:
            root["harness.task_signature"] = self.task_signature
        spans_path = self.out_dir / "spans.jsonl"
        spans_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spans_path, "a") as fh:
            for span in [root, *self._spans]:
                fh.write(json.dumps(span, sort_keys=True, default=str) + "\n")
