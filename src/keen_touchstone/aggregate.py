"""Build ReliabilityAggregates from TaskTrials — the glue every adapter feeds.

One suite-level aggregate (task_key "__suite__", carrying the decay curve with
its bootstrap CI band) plus one aggregate per task (Beta-Binomial CI at the
same headline k, so tasks are comparable). The headline k is the deepest k of
the suite curve — min(n_i) unless the caller narrows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

import numpy as np

from .artifacts import SUITE_TASK_KEY, DecayCurvePoint, ReliabilityAggregate
from .stats import (
    SMALL_SAMPLE_TASKS,
    TaskTrials,
    beta_binomial_ci,
    bootstrap_ci_curve,
    decay_curve,
    max_common_k,
    pass_at_k,
    pass_hat_k,
    power_status,
    variance_decomposition,
)

TaskKeySource = str  # "dataset_id" | "task_signature" | "declared_tag"


@dataclass(frozen=True)
class SuiteResult:
    suite: ReliabilityAggregate
    tasks: list[ReliabilityAggregate]
    warnings: list[str] = field(default_factory=list)
    lever: str | None = None  # variance-decomposition readout, for the report


def _cost_columns(tasks: list[TaskTrials]) -> tuple[float | None, float | None, float | None]:
    costs = [c for t in tasks if t.costs for c in t.costs]
    tokens = [tok for t in tasks if t.tokens for tok in t.tokens]
    cost_mean = fmean(costs) if costs else None
    cost_p95 = float(np.percentile(costs, 95)) if costs else None
    token_mean = fmean(tokens) if tokens else None
    return cost_mean, cost_p95, token_mean


def default_headline_k(curve_k_max: int) -> int:
    """Headline k defaults to ceil(k_max/2): at k close to n the per-task
    estimator C(c,k)/C(n,k) degenerates toward {0,1} (one failure zeroes it),
    so the headline number lives where the estimator is still informative.
    The full curve is still computed to k_max — nothing is hidden."""
    return max(1, (curve_k_max + 1) // 2)


def build_suite_result(
    tasks: list[TaskTrials],
    *,
    context: str,
    model: str,
    agent_config_hash: str,
    task_key_source: TaskKeySource | None = None,
    k_max: int | None = None,
    headline_k: int | None = None,
    n_resamples: int = 2000,
    seed: int | None = 2026,
    level: float = 0.95,
) -> SuiteResult:
    if not tasks:
        raise ValueError("no tasks supplied")
    warnings: list[str] = []

    common = max_common_k(tasks)
    curve_k_max = min(k_max, common) if k_max is not None else common
    if headline_k is None:
        headline_k = default_headline_k(curve_k_max)
    elif headline_k > curve_k_max:
        warnings.append(
            f"requested headline k={headline_k} exceeds the curve range (k_max={curve_k_max}); "
            f"clamped to {curve_k_max}"
        )
        headline_k = curve_k_max
    ks = list(range(1, curve_k_max + 1))

    points = decay_curve(tasks, k_max=curve_k_max)
    band = bootstrap_ci_curve(
        tasks, ks=ks, statistic="pass_hat_k", n_resamples=n_resamples, seed=seed, level=level
    )
    if band.small_sample_warning:
        warnings.append(
            f"only {len(tasks)} task(s) (< {SMALL_SAMPLE_TASKS}): the bootstrap cannot see the "
            "between-task tail — treat the suite CI as optimistic"
        )
    if curve_k_max == 1:
        warnings.append(
            "only 1 trial per task on at least one task: pass^k beyond k=1 is not estimable — "
            "rerun with more epochs/trials to get a decay curve"
        )
    if band.widened_ks:
        warnings.append(
            f"bootstrap CI degenerated to zero width at k={list(band.widened_ks)} (per-task "
            "estimator saturates near k=n); those intervals were widened with Beta-posterior "
            "draws rather than reported as false certainty"
        )

    decomp = variance_decomposition(tasks)
    curve_models = [
        DecayCurvePoint(
            k=pt.k,
            pass_hat_k=pt.pass_hat_k,
            ci_low=band.intervals[pt.k].low,
            ci_high=band.intervals[pt.k].high,
        )
        for pt in points
    ]
    headline_ci = band.intervals[headline_k]
    headline_point = points[headline_k - 1]
    cost_mean, cost_p95, token_mean = _cost_columns(tasks)

    suite = ReliabilityAggregate(
        task_key=SUITE_TASK_KEY,
        task_key_source=task_key_source,
        agent_config_hash=agent_config_hash,
        model=model,
        n_rollouts=sum(t.n for t in tasks),
        context=context,
        pass_rate=fmean(t.pass_rate for t in tasks),
        pass_at_k=headline_point.pass_at_k,
        pass_hat_k=headline_point.pass_hat_k,
        headline_k=headline_k,
        pass_hat_k_ci_low=headline_ci.low,
        pass_hat_k_ci_high=headline_ci.high,
        ci_method="bootstrap",
        reliability_decay_curve=curve_models,
        variance=decomp.observed_between_task_variance,
        skew=decomp.skew,
        cost_mean=cost_mean,
        cost_p95=cost_p95,
        token_mean=token_mean,
        power_status=power_status(headline_ci),
    )

    per_task: list[ReliabilityAggregate] = []
    for t in sorted(tasks, key=lambda t: t.task_key):
        ci = beta_binomial_ci(t.n, t.c, k=headline_k, level=level)
        t_cost_mean, t_cost_p95, t_token_mean = _cost_columns([t])
        per_task.append(
            ReliabilityAggregate(
                task_key=t.task_key,
                task_key_source=task_key_source,
                agent_config_hash=agent_config_hash,
                model=model,
                n_rollouts=t.n,
                context=context,
                pass_rate=t.pass_rate,
                pass_at_k=pass_at_k(t.n, t.c, headline_k),
                pass_hat_k=pass_hat_k(t.n, t.c, headline_k),
                headline_k=headline_k,
                pass_hat_k_ci_low=ci.low,
                pass_hat_k_ci_high=ci.high,
                ci_method="beta_binomial",
                cost_mean=t_cost_mean,
                cost_p95=t_cost_p95,
                token_mean=t_token_mean,
            )
        )

    return SuiteResult(suite=suite, tasks=per_task, warnings=warnings, lever=decomp.lever)
