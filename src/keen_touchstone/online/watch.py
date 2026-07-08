"""Continuous reliability: the SAME pass^k definition, computed over a stream.

Design choices, stated (they are statistical honesty decisions):

- **Tumbling windows, never sliding.** Overlapping windows re-consume the same
  runs on every look — alerts flap and the effective number of "tests" run
  against the SLO explodes silently. Tumbling windows consume each run exactly
  once. The repeated-look problem is thereby bounded, not solved: the more
  windows you evaluate, the more chances for a false alarm. Documented, not
  hidden; sequential-testing corrections are future work.
- **Two alert levels.** BREACH only when the window's entire confidence
  interval sits below the SLO (a confident claim); WARNING when the point
  estimate is below but the interval straddles (could be noise). With few
  tasks per window, confident breaches are rare — that is honest, not timid.
- **Insufficient depth is named, not fudged.** An SLO at pass^4 needs ≥4
  trials of each task inside the window; a window that can't estimate at that
  depth reports ``insufficient_n`` and makes no claim.
- **The trailing partial window is 'pending'** — never alerted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from keen_touchstone.adapters.otel_traces import Span, TraceRun, trials_from_traces
from keen_touchstone.adapters.signatures import runs_from_spans
from keen_touchstone.aggregate import build_suite_result

_SLO_RE = re.compile(r"^(0?\.\d+|1(?:\.0+)?)@(\d+)$")


def parse_slo(text: str) -> tuple[float, int]:
    """'0.6@4' → (0.6, 4): pass^4 must stay at or above 60%."""
    match = _SLO_RE.match(text.strip())
    if not match:
        raise ValueError(
            f"SLO {text!r} must look like '0.6@4' (minimum pass^k value @ k)"
        )
    value, k = float(match.group(1)), int(match.group(2))
    if k < 1:
        raise ValueError("SLO k must be >= 1")
    return value, k


@dataclass(frozen=True)
class WindowResult:
    index: int
    start_time: str | None
    end_time: str | None
    n_runs: int
    n_tasks: int
    k: int | None  # depth actually estimated (None when no data)
    pass_rate: float | None
    pass_hat_k: float | None
    ci_low: float | None
    ci_high: float | None
    status: str  # ok | warning | breach | insufficient_n | pending | no_data
    detail: str


@dataclass(frozen=True)
class WatchReport:
    windows: list[WindowResult]
    slo_value: float
    slo_k: int
    latest_full_status: str  # status of the last non-pending window (display)
    standing_breach: bool
    """r4-F2: only an `ok` window may CLEAR a confident breach. `warning`,
    `insufficient_n` and `no_data` are abstentions/ambiguity, not evidence of
    recovery — an abstention must never retract a standing alert."""
    warnings: list[str] = field(default_factory=list)

    @property
    def breached(self) -> bool:
        return self.standing_breach


def parse_stamp(value: Any) -> datetime | None:
    """Parse start_time into a real timeline (review r4-F4: string comparison
    misordered mixed UTC offsets and sorted numeric epochs by digit count).
    Accepts ISO-8601 (Z normalized) and numeric epochs (s/ms/ns heuristic)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        if v > 1e14:  # nanoseconds (OTLP startTimeUnixNano)
            v /= 1e9
        elif v > 1e11:  # milliseconds
            v /= 1e3
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _order_runs(runs: list[TraceRun]) -> tuple[list[TraceRun], int, int]:
    """Timestamped runs in true chronological order first; runs without a
    parseable stamp keep FILE order after them (r4-F1: file order is now real
    — runs_from_spans preserves insertion order). Returns (ordered,
    n_untimestamped, n_unparseable)."""
    keyed: list[tuple[int, datetime | None, int, TraceRun]] = []
    n_missing = n_unparseable = 0
    for position, run in enumerate(runs):
        raw = run.attr("start_time")
        stamp = parse_stamp(raw)
        if stamp is None:
            if raw is None:
                n_missing += 1
            else:
                n_unparseable += 1
            keyed.append((1, None, position, run))
        else:
            keyed.append((0, stamp, position, run))
    keyed.sort(key=lambda item: (item[0], item[1] or datetime.min.replace(tzinfo=timezone.utc), item[2]))
    return [item[3] for item in keyed], n_missing, n_unparseable


def _evaluate_window(
    index: int,
    runs: list[TraceRun],
    slo_value: float,
    slo_k: int,
    pending: bool,
    strategy: Any,
    threshold: float,
    outcome_regex: str | None,
    seed: int,
) -> WindowResult:
    spans = [span for run in runs for span in run.spans]
    stamps = sorted(str(r.attr("start_time")) for r in runs if r.attr("start_time"))
    start, end = (stamps[0], stamps[-1]) if stamps else (None, None)
    base = dict(index=index, start_time=start, end_time=end, n_runs=len(runs))

    try:
        ingested = trials_from_traces(
            spans, strategy=strategy, threshold=threshold, outcome_regex=outcome_regex
        )
    except ValueError as err:
        return WindowResult(**base, n_tasks=0, k=None, pass_rate=None, pass_hat_k=None,
                            ci_low=None, ci_high=None, status="no_data", detail=str(err)[:140])

    common = min(t.n for t in ingested.tasks)
    k = min(slo_k, common)
    result = build_suite_result(
        ingested.tasks, context="online", model=ingested.model,
        agent_config_hash=ingested.agent_config_hash, k_max=k, headline_k=k, seed=seed,
    )
    suite = result.suite
    base.update(
        n_tasks=len(ingested.tasks), k=k, pass_rate=suite.pass_rate,
        pass_hat_k=suite.pass_hat_k, ci_low=suite.pass_hat_k_ci_low,
        ci_high=suite.pass_hat_k_ci_high,
    )

    if pending:
        return WindowResult(**base, status="pending",
                            detail="trailing partial window — no claim made")
    if common < slo_k:
        return WindowResult(
            **base, status="insufficient_n",
            detail=f"SLO is at pass^{slo_k} but some task has only {common} trial(s) in this "
                   f"window — no claim at that depth (showing pass^{k})",
        )
    if suite.pass_hat_k_ci_high is not None and suite.pass_hat_k_ci_high < slo_value:
        return WindowResult(
            **base, status="breach",
            detail=f"CONFIDENT BREACH: the whole 95% CI [{suite.pass_hat_k_ci_low:.2f}, "
                   f"{suite.pass_hat_k_ci_high:.2f}] sits below the SLO {slo_value}",
        )
    if suite.pass_hat_k is not None and suite.pass_hat_k < slo_value:
        return WindowResult(
            **base, status="warning",
            detail=f"point estimate {suite.pass_hat_k:.2f} below SLO {slo_value} but the CI "
                   "straddles it — could be noise; keep watching",
        )
    return WindowResult(**base, status="ok", detail="")


def watch_stream(
    spans: list[Span],
    window_size: int,
    slo: str,
    strategy: Any = None,
    threshold: float = 1.0,
    outcome_regex: str | None = None,
    seed: int = 2026,
) -> WatchReport:
    slo_value, slo_k = parse_slo(slo)
    if window_size < 2:
        raise ValueError("window must hold at least 2 runs")
    runs, n_missing, n_unparseable = _order_runs(runs_from_spans(spans))
    if not runs:
        raise ValueError("no runs in the stream")

    windows: list[WindowResult] = []
    warnings: list[str] = []
    for i in range(0, len(runs), window_size):
        chunk = runs[i : i + window_size]
        pending = len(chunk) < window_size
        windows.append(
            _evaluate_window(
                len(windows), chunk, slo_value, slo_k, pending,
                strategy, threshold, outcome_regex, seed,
            )
        )
    full = [w for w in windows if w.status != "pending"]
    latest = full[-1].status if full else "pending"

    standing = False
    breach_window: int | None = None
    for w in full:
        if w.status == "breach":
            standing = True
            breach_window = w.index
        elif w.status == "ok":
            standing = False
    if standing and latest != "breach":
        warnings.append(
            f"a confident breach at window {breach_window} is STILL STANDING — later windows "
            f"read {latest!r}, which is not evidence of recovery (only an ok window clears)"
        )

    if n_missing:
        warnings.append(
            f"{n_missing} run(s) carry no start_time — kept in file order after timestamped "
            "runs; a mixed stream may window imprecisely"
        )
    if n_unparseable:
        warnings.append(
            f"{n_unparseable} run(s) carry an UNPARSEABLE start_time — treated as "
            "untimestamped (expected ISO-8601 or epoch seconds/ms/ns)"
        )
    n_evaluated = sum(1 for w in full if w.status in ("ok", "warning", "breach"))
    if n_evaluated > 1:
        warnings.append(
            f"{n_evaluated} windows evaluated against the SLO — repeated looks inflate the "
            "false-alarm rate (bounded by tumbling windows, not eliminated)"
        )
    return WatchReport(
        windows=windows, slo_value=slo_value, slo_k=slo_k,
        latest_full_status=latest, standing_breach=standing, warnings=warnings,
    )
