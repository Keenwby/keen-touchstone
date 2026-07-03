"""Variance decomposition: tell the user which lever to pull.

Observed spread in per-task pass rates mixes two sources (law of total
variance, the Miller-2024 framing):

- **within-task sampling noise** — you only ran n_i trials per task; shrinks
  with more rollouts per task. Unbiased per-task estimate of Var(p_hat_i):
  p_hat(1-p_hat)/(n-1).
- **between-task heterogeneity** — tasks genuinely differ in difficulty;
  shrinks only with more tasks. Estimated as observed variance of p_hat_i
  minus mean within-task noise (floored at 0).

The lever readout is the actionable output: spending budget on the wrong axis
is the classic way eval programs waste their rerun budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from .estimators import TaskTrials
from .intervals import Interval

CI_WIDTH_UNDERPOWERED = 0.2
"""Suite-CI width at the headline k above which we flag UNDERPOWERED."""


@dataclass(frozen=True)
class VarianceDecomposition:
    observed_between_task_variance: float | None  # var of per-task pass rates (needs >=2 tasks)
    mean_within_task_variance: float | None  # sampling noise (needs n_i >= 2)
    est_true_between_task_variance: float | None  # observed minus within, floored at 0
    skew: float | None  # skew of per-task pass rates (needs >=3 tasks)
    lever: str | None  # "more_rollouts_per_task" | "more_tasks" | None if undecidable
    n_tasks_excluded_within: int  # tasks with n_i == 1 (within-variance undefined)


def variance_decomposition(tasks: list[TaskTrials]) -> VarianceDecomposition:
    if not tasks:
        raise ValueError("no tasks supplied")
    rates = np.array([t.pass_rate for t in tasks], dtype=float)

    observed_between = float(np.var(rates, ddof=1)) if len(tasks) >= 2 else None

    within_terms = [t.pass_rate * (1 - t.pass_rate) / (t.n - 1) for t in tasks if t.n >= 2]
    excluded = sum(1 for t in tasks if t.n == 1)
    mean_within = float(np.mean(within_terms)) if within_terms else None

    est_between: float | None = None
    if observed_between is not None and mean_within is not None:
        est_between = max(0.0, observed_between - mean_within)

    lever: str | None = None
    if est_between is not None and mean_within is not None:
        lever = "more_rollouts_per_task" if mean_within > est_between else "more_tasks"

    skew: float | None = None
    if len(tasks) >= 3:
        # identical rates make scipy's moment ratio 0/0; define skew as 0 there
        skew = 0.0 if np.ptp(rates) == 0 else float(scipy_stats.skew(rates))

    return VarianceDecomposition(
        observed_between_task_variance=observed_between,
        mean_within_task_variance=mean_within,
        est_true_between_task_variance=est_between,
        skew=skew,
        lever=lever,
        n_tasks_excluded_within=excluded,
    )


def power_status(headline_ci: Interval | None) -> str | None:
    """v0.1 heuristic for the schema's power_status field.

    A full power analysis (minimum detectable difference at target power) is
    Phase 2+ scope; for now the only claim we can make honestly from a single
    configuration is "this interval is too wide to act on".
    """
    if headline_ci is None:
        return None
    if headline_ci.high - headline_ci.low > CI_WIDTH_UNDERPOWERED:
        return "UNDERPOWERED_NEED_MORE_N"
    return None
