"""Seeded synthetic OTel GenAI trace generator.

Produces gen_ai.* span JSONL shaped like production agent traffic — root
invoke_agent span + chat/execute_tool children with token usage — with known
true per-task success probabilities. Two jobs:

1. the trace-lane demo (``touchstone ingest --demo``);
2. a truth harness for tests: the estimator must recover p_i^k from traces it
   did not author.

Every span validates against the packaged span.schema.json (tested).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DEMO_TRACE_TASKS: dict[str, float] = {
    "support/refund-flow": 0.9,
    "support/address-change": 0.8,
    "ops/log-triage": 0.7,
    "ops/incident-summary": 0.6,
    "research/competitor-scan": 0.5,
}

DEMO_MODEL = "demo/agent-model"

# realistic per-task input templates — material for derived task identity
# (ids vary per run; derived DETERMINISTICALLY from the run counter so adding
# them consumes NO shared-rng draws — seeded outcome tests stay stable)
DEMO_TASK_INPUTS: dict[str, str] = {
    "support/refund-flow": "Refund order #{n} for customer C{c} and email confirmation to user{c}@example.com",
    "support/address-change": "Update the shipping address for customer C{c} to {n} Main Street",
    "ops/log-triage": "Triage the error spike in service svc-{c} around 2026-07-{d:02d}",
    "ops/incident-summary": "Write a postmortem summary for incident INC-{n}",
    "research/competitor-scan": "Find the top competitor mentions for product P{c} since 2026-06-{d:02d}",
}


def gen_spans(
    seed: int = 2026,
    tasks: dict[str, float] | None = None,
    runs_per_task: int = 12,
    model: str = DEMO_MODEL,
    unsigned_runs: int = 2,
    config_hash: str | None = None,
    include_outcomes: bool = True,
) -> list[dict[str, Any]]:
    """Deterministic synthetic spans. ``unsigned_runs`` adds runs without a
    task signature so exclusion accounting is visible in the demo.
    ``include_outcomes=False`` omits harness.outcome — the unlabeled-traffic
    shape that the judge-verdict loop exists for."""
    spans, _ = gen_spans_with_truth(
        seed=seed, tasks=tasks, runs_per_task=runs_per_task, model=model,
        unsigned_runs=unsigned_runs, config_hash=config_hash,
        include_outcomes=include_outcomes,
    )
    return spans


def gen_spans_with_truth(
    seed: int = 2026,
    tasks: dict[str, float] | None = None,
    runs_per_task: int = 12,
    model: str = DEMO_MODEL,
    unsigned_runs: int = 2,
    config_hash: str | None = None,
    include_outcomes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Like gen_spans, but also returns the latent truth (trace_id → really
    succeeded) — the test/demo oracle that production traffic never has."""
    spans, labels = gen_spans_labeled(
        seed=seed, tasks=tasks, runs_per_task=runs_per_task, model=model,
        unsigned_runs=unsigned_runs, config_hash=config_hash,
        include_outcomes=include_outcomes,
    )
    return spans, {tid: lab["ok"] for tid, lab in labels.items()}


def gen_spans_labeled(
    seed: int = 2026,
    tasks: dict[str, float] | None = None,
    runs_per_task: int = 12,
    model: str = DEMO_MODEL,
    unsigned_runs: int = 2,
    config_hash: str | None = None,
    include_outcomes: bool = True,
    include_signatures: bool = True,
    drift: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Full-truth variant: trace_id → {"ok", "task", "start_time"}.

    New Phase 4 material, all derived WITHOUT extra shared-rng draws:
    - root spans carry a realistic ``input.messages`` text (per-task template,
      ids varied by the run counter) — food for derived task identity;
    - root spans carry ``start_time`` interleaving tasks round-robin in time,
      so the stream reads like production traffic (watch sorts by it);
    - ``include_signatures=False`` omits harness.task_signature (the unlabeled
      traffic that derived identity exists for);
    - ``drift={"task": sig, "after": k, "p": new_p}`` degrades one task's
      success probability from its k-th run onward ("*" degrades every task) —
      trigger-designed, not probabilistic (the Phase 2 demo lesson).
    """
    tasks = tasks if tasks is not None else DEMO_TRACE_TASKS
    rng = random.Random(seed)
    config = config_hash or f"democfg{seed:05d}"
    spans: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    task_order = list(tasks)
    run_no = 0
    local_index: dict[str, int] = {sig: 0 for sig in tasks}

    def add_run(signature: str | None, p: float) -> None:
        nonlocal run_no
        run_no += 1
        idx = local_index.get(signature, 0)
        if signature is not None:
            local_index[signature] = idx + 1
            if (
                drift is not None
                and drift["task"] in (signature, "*")
                and idx >= drift["after"]
            ):
                p = drift["p"]
        trace_id = f"{rng.getrandbits(64):016x}{run_no:016x}"
        succeeded = rng.random() < p
        # interleave tasks round-robin in TIME without touching rng order:
        # minute offset = local_run_index * n_tasks + task_position
        task_pos = task_order.index(signature) if signature in task_order else len(task_order)
        minute = idx * (len(task_order) + 1) + task_pos
        start_time = _minute_to_iso(minute)
        labels[trace_id] = {"ok": succeeded, "task": signature, "start_time": start_time}
        step = 0

        def span(span_id: str, parent: str | None, op: str, extra: dict[str, Any]) -> None:
            nonlocal step
            base: dict[str, Any] = {
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent,
                "gen_ai.operation.name": op,
                "harness.step_id": step,
                "harness.agent_config_hash": config,
            }
            step += 1
            base.update(extra)
            spans.append(base)

        root_extra: dict[str, Any] = {"gen_ai.request.model": model, "start_time": start_time}
        if include_outcomes:
            root_extra["harness.outcome"] = "success" if succeeded else "failure"
        if signature is not None and include_signatures:
            root_extra["harness.task_signature"] = signature
        template = DEMO_TASK_INPUTS.get(signature) if signature else None
        if template:
            root_extra["input.messages"] = template.format(
                n=7000 + run_no, c=800 + run_no, d=(run_no % 28) + 1
            )
        span("0001", None, "invoke_agent", root_extra)

        for turn in range(rng.randint(2, 4)):
            span(
                f"c{turn:03d}",
                "0001",
                "chat",
                {
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": rng.randint(300, 1200),
                    "gen_ai.usage.output_tokens": rng.randint(80, 400),
                },
            )
            span(
                f"t{turn:03d}",
                "0001",
                "execute_tool",
                {"gen_ai.tool.name": rng.choice(["search", "read_file", "http_get", "sql"])},
            )

    for signature, p in tasks.items():
        for _ in range(runs_per_task):
            add_run(signature, p)
    for _ in range(unsigned_runs):
        add_run(None, 0.5)

    return spans, labels


def _minute_to_iso(minute: int) -> str:
    hour, minute_of_hour = 8 + minute // 60, minute % 60
    return f"2026-07-01T{hour:02d}:{minute_of_hour:02d}:00+00:00"


def write_spans_jsonl(path: str | Path, spans: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for span in spans:
            fh.write(json.dumps(span, sort_keys=True) + "\n")
    return path
