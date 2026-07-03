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
    tasks = tasks if tasks is not None else DEMO_TRACE_TASKS
    rng = random.Random(seed)
    config = config_hash or f"democfg{seed:05d}"
    spans: list[dict[str, Any]] = []
    truth: dict[str, bool] = {}
    run_no = 0

    def add_run(signature: str | None, p: float) -> None:
        nonlocal run_no
        run_no += 1
        trace_id = f"{rng.getrandbits(64):016x}{run_no:016x}"
        succeeded = rng.random() < p
        truth[trace_id] = succeeded
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

        root_extra: dict[str, Any] = {"gen_ai.request.model": model}
        if include_outcomes:
            root_extra["harness.outcome"] = "success" if succeeded else "failure"
        if signature is not None:
            root_extra["harness.task_signature"] = signature
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

    return spans, truth


def write_spans_jsonl(path: str | Path, spans: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for span in spans:
            fh.write(json.dumps(span, sort_keys=True) + "\n")
    return path
