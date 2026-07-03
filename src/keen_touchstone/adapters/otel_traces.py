"""OTel GenAI trace ingestion: gen_ai.* spans (JSONL) → TaskTrials.

The "no golden dataset" lane: production traces become the eval set. Input is
one span per line, flat attributes in the Phase 0 ``span.schema.json``
vocabulary (OTel ``gen_ai.*`` + net-new ``harness.*`` fields). Spans sharing a
``trace_id`` form one run.

Task identity (SPEC §5): v0.1 ships only the *declared-tag* strategy — the app
annotates ``harness.task_signature``. The ``TaskSignatureStrategy`` interface
is the extension point where derived strategies (template extraction,
tool-sequence signature, embedding clustering) land in Phase 4.

Outcome resolution, in priority order (a run with no resolvable outcome is
EXCLUDED and counted — never silently treated as a failure):

1. ``harness.outcome`` attribute: "success"/"failure" (or boolean).
2. ``gen_ai.evaluation.score.value`` (>= threshold) or
   ``gen_ai.evaluation.score.label`` ("pass"/"success" vs "fail"/"failure") —
   the OTel evaluation-event vocabulary, flattened onto a span.
3. ``--outcome-regex`` matched against the root span's ``output.messages``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from keen_touchstone.stats import TaskTrials

Span = dict[str, Any]


@dataclass(frozen=True)
class TraceRun:
    """All spans sharing one trace_id — one agent run."""

    trace_id: str
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        return next((s for s in self.spans if s.get("parent_span_id") in (None, "")), None)

    def attr(self, key: str) -> Any:
        """Value of ``key`` from the root span if present there, else the first
        span that carries it (documents-before-guesses ordering)."""
        root = self.root
        if root is not None and root.get(key) is not None:
            return root[key]
        return next((s[key] for s in self.spans if s.get(key) is not None), None)

    @property
    def total_tokens(self) -> int:
        return sum(
            _as_int(s.get("gen_ai.usage.input_tokens")) + _as_int(s.get("gen_ai.usage.output_tokens"))
            for s in self.spans
        )


def _as_int(value: Any) -> int:
    """Tolerant token-count coercion: ints, floats, and numeric strings
    (including "300.5") all count; garbage counts as 0 rather than aborting
    an entire ingest over one malformed usage attribute."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


class TaskSignatureStrategy(Protocol):
    """SPEC §5: derive the 'same task' grouping key from an organic run."""

    name: str

    def signature(self, run: TraceRun) -> str | None: ...


class DeclaredTagStrategy:
    """Read the app-annotated ``harness.task_signature`` — highest fidelity,
    lowest coverage. The only strategy shipped in v0.1."""

    name = "declared_tag"

    def signature(self, run: TraceRun) -> str | None:
        value = run.attr("harness.task_signature")
        return str(value) if value else None


@dataclass(frozen=True)
class TraceIngest:
    tasks: list[TaskTrials]
    model: str
    agent_config_hash: str
    n_runs: int
    excluded: dict[str, int]  # reason -> count
    warnings: list[str] = field(default_factory=list)


def read_spans_jsonl(path: str | Path) -> list[Span]:
    spans: list[Span] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({err.msg})") from err
            if not isinstance(span, dict) or "trace_id" not in span:
                raise ValueError(f"{path}:{lineno}: expected a span object with trace_id")
            spans.append(span)
    if not spans:
        raise ValueError(f"{path}: no spans found")
    return spans


def _resolve_outcome(
    run: TraceRun, threshold: float, outcome_regex: str | None
) -> bool | None:
    declared = run.attr("harness.outcome")
    if declared is not None:
        if isinstance(declared, bool):
            return declared
        text = str(declared).lower()
        if text in ("success", "true", "pass", "passed"):
            return True
        if text in ("failure", "false", "fail", "failed"):
            return False
        return None  # unparseable declaration -> excluded, not guessed

    score = run.attr("gen_ai.evaluation.score.value")
    if score is not None:
        try:
            return float(score) >= threshold
        except (TypeError, ValueError):
            return None
    label = run.attr("gen_ai.evaluation.score.label")
    if label is not None:
        text = str(label).lower()
        if text in ("pass", "passed", "success"):
            return True
        if text in ("fail", "failed", "failure", "error"):
            return False
        return None

    if outcome_regex:
        root = run.root
        output = (root or {}).get("output.messages")
        if output is not None:
            return re.search(outcome_regex, json.dumps(output) if not isinstance(output, str) else output) is not None
    return None


def trials_from_traces(
    spans: list[Span],
    strategy: TaskSignatureStrategy | None = None,
    threshold: float = 1.0,
    outcome_regex: str | None = None,
    outcome_overrides: dict[str, bool] | None = None,
) -> TraceIngest:
    """``outcome_overrides`` (trace_id → outcome, e.g. from a licensed judge's
    verdicts) REPLACES span-attribute outcome resolution entirely: the verdict
    layer is authoritative, and runs without a verdict are excluded
    (reason: no_verdict) — never resolved from a mix of sources."""
    strategy = strategy or DeclaredTagStrategy()

    by_trace: dict[str, list[Span]] = {}
    for span in spans:
        by_trace.setdefault(str(span["trace_id"]), []).append(span)
    runs = [TraceRun(trace_id=tid, spans=s) for tid, s in sorted(by_trace.items())]

    excluded: Counter[str] = Counter()
    warnings: list[str] = []
    counts: dict[str, dict[str, Any]] = {}
    models: Counter[str] = Counter()
    config_hashes: set[str] = set()

    for run in runs:
        signature = strategy.signature(run)
        if signature is None:
            excluded["no_task_signature"] += 1
            continue
        if outcome_overrides is not None:
            outcome = outcome_overrides.get(run.trace_id)
            if outcome is None:
                excluded["no_verdict"] += 1
                continue
        else:
            outcome = _resolve_outcome(run, threshold, outcome_regex)
            if outcome is None:
                excluded["no_resolvable_outcome"] += 1
                continue
        model = run.attr("gen_ai.request.model") or run.attr("gen_ai.response.model")
        if model:
            models[str(model)] += 1
        declared_hash = run.attr("harness.agent_config_hash")
        if declared_hash:
            config_hashes.add(str(declared_hash))

        entry = counts.setdefault(signature, {"n": 0, "c": 0, "tokens": []})
        entry["n"] += 1
        entry["c"] += int(outcome)
        entry["tokens"].append(run.total_tokens)

    if not counts:
        raise ValueError(
            f"no usable runs: {dict(excluded)} — runs need harness.task_signature plus a "
            "resolvable outcome (harness.outcome, gen_ai.evaluation.score.*, or --outcome-regex)"
        )
    if len(models) > 1:
        raise ValueError(
            f"traces span multiple models {sorted(models)} — analyze one (model, config) at a "
            "time so pass^k groups honestly (filter the JSONL first)"
        )
    if len(config_hashes) > 1:
        raise ValueError(
            f"traces span multiple harness.agent_config_hash values {sorted(config_hashes)} — "
            "analyze one configuration at a time"
        )

    model = next(iter(models), "unknown")
    if config_hashes:
        agent_config_hash = next(iter(config_hashes))
    else:
        agent_config_hash = hashlib.sha256(f"model:{model}".encode()).hexdigest()[:12]
        warnings.append(
            "traces carry no harness.agent_config_hash — defaulted from model only; stamp it "
            "at emission time so config changes are distinguishable"
        )
    for reason, count in excluded.items():
        warnings.append(f"excluded {count} run(s): {reason} (excluded, not counted as failures)")
    if outcome_overrides is not None:
        run_ids = {run.trace_id for run in runs}
        dangling = sum(1 for tid in outcome_overrides if tid not in run_ids)
        if dangling:
            warnings.append(
                f"{dangling} verdict(s)/override(s) matched no run in these traces — stale "
                "export or wrong trace file?"
            )

    tasks = [
        TaskTrials(
            task_key=key,
            n=entry["n"],
            c=entry["c"],
            tokens=tuple(entry["tokens"]) if any(entry["tokens"]) else None,
        )
        for key, entry in sorted(counts.items())
    ]
    return TraceIngest(
        tasks=tasks,
        model=model,
        agent_config_hash=agent_config_hash,
        n_runs=len(runs),
        excluded=dict(excluded),
        warnings=warnings,
    )
