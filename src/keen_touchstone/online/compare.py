"""Paired comparison of two agent configurations + the release gate.

The last unshipped item on the thesis's statistics list (§6.4): **paired
difference testing** — comparing two configs on the SAME tasks cancels
task-difficulty noise and buys "free" statistical power vs comparing two
independent suite averages.

Method, stated exactly:

- Per shared task, delta = pass^k(candidate) − pass^k(baseline) (the same
  UMVUE as everywhere else; per-task n and c are reconstructed exactly from
  aggregate.json's ``n_rollouts`` × ``pass_rate`` — locked by tests).
- Significance: sign-flip permutation test on the deltas (exact enumeration
  up to 2^12 flips, seeded sampling beyond), two-sided, add-one smoothed.
  Cross-checked in tests against scipy.stats.permutation_test.
- Magnitude: bootstrap CI on the mean delta (resampling tasks).
- Verdict: SIGNIFICANT_REGRESSION / SIGNIFICANT_IMPROVEMENT / NOISE, plus an
  UNDERPOWERED flag when the shared-task count is too small to mean much —
  a non-significant result on 3 tasks is absence of evidence, and says so.

The SLO gate reads the suite's decay curve at the SLO's k and fails only on
a **confident breach** (whole CI below the line) by default; ``--strict``
fails on the point estimate. A release gate should be conservative about
blocking on noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from keen_touchstone.stats import pass_hat_k

from .watch import parse_slo

MIN_SHARED_FOR_POWER = 5
EXACT_ENUMERATION_LIMIT = 12  # 2^12 sign patterns


@dataclass(frozen=True)
class TaskDelta:
    task_key: str
    n_a: int
    c_a: int
    n_b: int
    c_b: int
    pass_hat_a: float
    pass_hat_b: float
    delta: float


@dataclass(frozen=True)
class ComparisonResult:
    k: int
    n_shared: int
    n_only_baseline: int
    n_only_candidate: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str  # "exact_sign_flip" | "sampled_sign_flip"
    verdict: str  # SIGNIFICANT_REGRESSION | SIGNIFICANT_IMPROVEMENT | NOISE
    underpowered: bool
    deltas: list[TaskDelta] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_aggregate_tasks(path: str | Path) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    """aggregate.json → {task_key: (n, c)} + the raw payload.

    c is reconstructed as round(pass_rate × n) — exact, because pass_rate was
    computed as c/n; the reconstruction is verified, not assumed."""
    payload = json.loads(Path(path).read_text())
    if "tasks" not in payload or "suite" not in payload:
        raise ValueError(f"{path}: not a KeenTouchstone aggregate.json (missing suite/tasks)")
    tasks: dict[str, tuple[int, int]] = {}
    for row in payload["tasks"]:
        n = int(row["n_rollouts"])
        c = round(row["pass_rate"] * n)
        if abs(c - row["pass_rate"] * n) > 1e-6:
            raise ValueError(
                f"{path}: task {row['task_key']!r} pass_rate {row['pass_rate']} × n {n} is not "
                "a whole count — file edited or not produced by this tool"
            )
        tasks[str(row["task_key"])] = (n, c)
    return tasks, payload


def _sign_flip_p_value(
    deltas: np.ndarray, seed: int, n_resamples: int
) -> tuple[float, str]:
    observed = abs(deltas.mean())
    n = len(deltas)
    if n <= EXACT_ENUMERATION_LIMIT:
        hits = total = 0
        for signs in product((1.0, -1.0), repeat=n):
            total += 1
            if abs((deltas * np.array(signs)).mean()) >= observed - 1e-12:
                hits += 1
        return hits / total, "exact_sign_flip"
    rng = np.random.default_rng(seed)
    signs = rng.choice((1.0, -1.0), size=(n_resamples, n))
    perm_means = np.abs((signs * deltas).mean(axis=1))
    # add-one smoothing: the observed arrangement is one of the permutations
    p = (int(np.sum(perm_means >= observed - 1e-12)) + 1) / (n_resamples + 1)
    return float(p), "sampled_sign_flip"


def compare(
    baseline: dict[str, tuple[int, int]],
    candidate: dict[str, tuple[int, int]],
    at_k: int | None = None,
    alpha: float = 0.05,
    seed: int = 2026,
    n_resamples: int = 10000,
) -> ComparisonResult:
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise ValueError(
            "no shared tasks between the two aggregates — a paired comparison needs the same "
            "tasks on both sides (check task signatures / signature strategy)"
        )
    notes: list[str] = []
    common = min(min(baseline[key][0], candidate[key][0]) for key in shared)
    k = at_k if at_k is not None else max(1, (common + 1) // 2)
    if k > common:
        notes.append(f"requested k={k} exceeds the shared trial floor ({common}); clamped")
        k = common

    deltas: list[TaskDelta] = []
    for key in shared:
        n_a, c_a = baseline[key]
        n_b, c_b = candidate[key]
        a = pass_hat_k(n_a, c_a, k)
        b = pass_hat_k(n_b, c_b, k)
        deltas.append(TaskDelta(key, n_a, c_a, n_b, c_b, a, b, b - a))

    d = np.array([t.delta for t in deltas])
    p_value, method = _sign_flip_p_value(d, seed=seed, n_resamples=n_resamples)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    boot_means = d[idx].mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

    mean_delta = float(d.mean())
    if p_value < alpha:
        verdict = "SIGNIFICANT_REGRESSION" if mean_delta < 0 else "SIGNIFICANT_IMPROVEMENT"
    else:
        verdict = "NOISE"
    underpowered = len(shared) < MIN_SHARED_FOR_POWER
    if underpowered:
        notes.append(
            f"only {len(shared)} shared task(s) (< {MIN_SHARED_FOR_POWER}): a non-significant "
            "result here is absence of evidence, not evidence of absence"
        )
    return ComparisonResult(
        k=k,
        n_shared=len(shared),
        n_only_baseline=len(set(baseline) - set(candidate)),
        n_only_candidate=len(set(candidate) - set(baseline)),
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        method=method,
        verdict=verdict,
        underpowered=underpowered,
        deltas=deltas,
        notes=notes,
    )


@dataclass(frozen=True)
class GateResult:
    ok: bool
    level: str  # pass | warning | breach
    message: str


def slo_gate(aggregate_path: str | Path, slo: str, strict: bool = False) -> GateResult:
    slo_value, slo_k = parse_slo(slo)
    _, payload = load_aggregate_tasks(aggregate_path)
    curve = payload["suite"].get("reliability_decay_curve") or []
    point = next((p for p in curve if p["k"] == slo_k), None)
    if point is None:
        raise ValueError(
            f"the aggregate's decay curve has no k={slo_k} (it runs to k={len(curve)}) — "
            "regenerate with enough trials per task to evaluate this SLO"
        )
    value, ci_high = point["pass_hat_k"], point.get("ci_high")
    where = f"pass^{slo_k} = {value:.3f}" + (
        f" (95% CI [{point.get('ci_low'):.3f}, {ci_high:.3f}])" if ci_high is not None else ""
    )
    if ci_high is not None and ci_high < slo_value:
        return GateResult(False, "breach", f"CONFIDENT BREACH: {where} — the whole CI is below the SLO {slo_value}")
    if value < slo_value:
        message = f"point estimate below SLO {slo_value} but the CI straddles it: {where}"
        if strict:
            return GateResult(False, "breach", "STRICT MODE: " + message)
        return GateResult(True, "warning", message + " — passing (default gates only on confident breach)")
    return GateResult(True, "pass", f"SLO {slo_value}@{slo_k} met: {where}")
