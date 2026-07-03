"""Inspect AI eval-log ingestion: .eval/.json logs → TaskTrials.

This is the only module (plus the CLI ``run`` command) that touches
inspect-ai — the isolation boundary the CLAUDE.md rules require. It uses the
public log API exclusively (``inspect_ai.log``), never raw file parsing.

Success semantics match Inspect's own reducers: value_to_float(score.value)
>= threshold (default 1.0), so "one metric definition" holds end to end.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.scorer import value_to_float

from keen_touchstone.stats import TaskTrials


@dataclass(frozen=True)
class LogIngest:
    """TaskTrials plus the identity metadata the aggregate builder needs."""

    tasks: list[TaskTrials]
    task_name: str
    model: str
    scorer_name: str
    agent_config_hash: str
    warnings: list[str] = field(default_factory=list)


def _canonical(location: str) -> str:
    """Normalize a log location (plain path or file:// URI) for de-duplication."""
    if location.startswith("file://"):
        location = unquote(urlparse(location).path)
    return os.path.realpath(location)


def resolve_log_paths(sources: list[str]) -> list[str]:
    """Expand files/directories into a flat, de-duplicated list of log locations.

    De-duplication is by canonical real path: passing a directory plus a file
    inside it (or the same path twice) must not double every task's n and c.
    """
    out: list[str] = []
    seen: set[str] = set()
    for src in sources:
        p = Path(src)
        candidates = (
            [info.name for info in list_eval_logs(str(p))] if p.is_dir() else [str(p)]
        )
        for loc in candidates:
            key = _canonical(loc)
            if key not in seen:
                seen.add(key)
                out.append(loc)
    if not out:
        raise ValueError(f"no eval logs found under {sources!r}")
    return out


def config_hash(task: str, task_version: Any, task_args: Any, model: str) -> str:
    """Stable 12-hex identity for (task, args, model) — the harness identity
    for reliability grouping. Epoch count deliberately excluded: rerunning
    with more epochs is the same configuration measured harder."""
    payload = json.dumps(
        {"task": task, "task_version": task_version, "task_args": task_args, "model": model},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _log_config_hash(log: EvalLog) -> str:
    return config_hash(
        log.eval.task,
        getattr(log.eval, "task_version", None),
        getattr(log.eval, "task_args", None),
        str(log.eval.model),
    )


def trials_from_logs(
    logs: list[EvalLog | str],
    scorer: str | None = None,
    success_threshold: float = 1.0,
    score_key: str | None = None,
) -> LogIngest:
    resolved: list[EvalLog] = [
        log if isinstance(log, EvalLog) else read_eval_log(log) for log in logs
    ]
    resolved = [log for log in resolved if log.samples]
    if not resolved:
        raise ValueError("no logs with samples (were these header-only or failed runs?)")

    task_names = {log.eval.task for log in resolved}
    models = {str(log.eval.model) for log in resolved}
    if len(task_names) > 1 or len(models) > 1:
        raise ValueError(
            f"logs span multiple tasks/models (tasks={sorted(task_names)}, models={sorted(models)}); "
            "analyze one (task, model) configuration at a time so pass^k groups honestly"
        )
    by_config: dict[str, EvalLog] = {_log_config_hash(log): log for log in resolved}
    if len(by_config) > 1:
        details = sorted(
            f"{h}: task_args={getattr(log.eval, 'task_args', None)!r}, "
            f"task_version={getattr(log.eval, 'task_version', None)!r}"
            for h, log in by_config.items()
        )
        raise ValueError(
            "logs share a task name and model but differ in configuration "
            f"(task_args/task_version) — merging them would pool different agents into one "
            f"pass^k. Configurations found: {details}"
        )

    scorer_names = sorted({name for log in resolved for s in log.samples for name in (s.scores or {})})
    if not scorer_names:
        raise ValueError("no scores found in these logs — was the eval scored?")
    if scorer is None:
        if len(scorer_names) > 1:
            raise ValueError(
                f"multiple scorers present {scorer_names}; pass --scorer to choose one"
            )
        scorer = scorer_names[0]
    elif scorer not in scorer_names:
        raise ValueError(f"scorer {scorer!r} not in logs (available: {scorer_names})")

    to_float = value_to_float()
    counts: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    skipped_unscored = 0

    for log in resolved:
        for sample in log.samples:
            score = (sample.scores or {}).get(scorer)
            if score is None:
                skipped_unscored += 1
                continue
            raw = score.value
            # Container values must never silently flatten: value_to_float()
            # maps an un-indexed dict/list to 0.0, which would report a
            # perfectly reliable agent as 0% reliable (adversarial-review
            # finding). Inspect's own reducers decompose per key.
            if isinstance(raw, dict):
                if score_key is None:
                    raise ValueError(
                        f"scorer {scorer!r} returns dict values with keys "
                        f"{sorted(map(str, raw))}; pass --score-key to choose the metric "
                        "that defines success"
                    )
                if score_key not in raw:
                    raise ValueError(
                        f"--score-key {score_key!r} not in score dict "
                        f"(available: {sorted(map(str, raw))})"
                    )
                raw = raw[score_key]
            elif isinstance(raw, (list, tuple)):
                raise ValueError(
                    f"scorer {scorer!r} returns list values; KeenTouchstone needs a scalar "
                    "success signal — use a scalar scorer or a dict scorer with --score-key"
                )
            value = to_float(raw)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                skipped_unscored += 1
                continue
            key = str(sample.id)
            entry = counts.setdefault(key, {"n": 0, "c": 0, "tokens": []})
            entry["n"] += 1
            entry["c"] += int(value >= success_threshold)
            usage_total = sum(
                (u.input_tokens or 0) + (u.output_tokens or 0)
                for u in (sample.model_usage or {}).values()
            )
            entry["tokens"].append(usage_total)

    if skipped_unscored:
        warnings.append(
            f"skipped {skipped_unscored} unscored/NaN sample(s) — they are excluded from n, not counted as failures"
        )
    if not counts:
        raise ValueError(f"no scored samples for scorer {scorer!r}")

    tasks = [
        TaskTrials(
            task_key=key,
            n=entry["n"],
            c=entry["c"],
            tokens=tuple(entry["tokens"]) if any(entry["tokens"]) else None,
        )
        for key, entry in sorted(counts.items())
    ]
    first = resolved[0]
    return LogIngest(
        tasks=tasks,
        task_name=first.eval.task,
        model=str(first.eval.model),
        scorer_name=scorer if score_key is None else f"{scorer}[{score_key}]",
        agent_config_hash=_log_config_hash(first),
        warnings=warnings,
    )
