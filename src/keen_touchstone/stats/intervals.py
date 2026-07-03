"""Confidence intervals for pass^k / pass@k aggregates.

What the suite-level CI means (written down so nobody has to guess):

The cluster bootstrap resamples **tasks**, never pooled attempts. Pooling
attempts would treat n_tasks x n_trials Bernoulli draws as independent and
understate variance wherever tasks differ in difficulty (they always do — the
clustered-SE lesson from Miller 2024). Resampling tasks targets the
*superpopulation* estimand — expected reliability on "tasks like these" —
which is the deployment question, and is why the interval stays honest even
when a fixed task list is all you have.

Below ``SMALL_SAMPLE_TASKS`` tasks the bootstrap cannot see the tail of the
between-task distribution; results carry a loud warning rather than a quietly
narrow interval. A wide interval is the honest product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from .estimators import TaskTrials, pass_at_k, pass_hat_k

SMALL_SAMPLE_TASKS = 5
"""Below this many tasks, bootstrap CIs carry a small-sample warning."""

JEFFREYS_PRIOR = (0.5, 0.5)


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    method: str  # matches the schema's ci_method enum
    level: float = 0.95


def wilson_ci(n: int, c: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a single task's pass rate (k=1)."""
    _validate(n, c)
    z = float(scipy_stats.norm.ppf(0.5 + level / 2))
    p = c / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half), "wilson", level)


def beta_binomial_ci(
    n: int, c: int, k: int = 1, level: float = 0.95, prior: tuple[float, float] = JEFFREYS_PRIOR
) -> Interval:
    """Bayesian credible interval for a single task's p^k.

    Posterior p ~ Beta(c + a, n - c + b) (Jeffreys prior by default). Because
    x -> x^k is monotone increasing, the interval for p^k is exactly the
    p-interval with endpoints raised to the k-th power — no resampling needed.
    """
    _validate(n, c)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    a, b = prior
    lo_q, hi_q = 0.5 - level / 2, 0.5 + level / 2
    dist = scipy_stats.beta(c + a, n - c + b)
    return Interval(float(dist.ppf(lo_q)) ** k, float(dist.ppf(hi_q)) ** k, "beta_binomial", level)


@dataclass(frozen=True)
class CurveCI:
    """CI band for the aggregate decay curve, one entry per requested k."""

    intervals: dict[int, Interval]
    n_tasks: int
    small_sample_warning: bool
    seed: int | None
    n_resamples: int
    widened_ks: tuple[int, ...] = ()
    """k values where the bootstrap component alone was zero-width (every
    per-task estimate identical — typical at k near n, where the UMVUE
    collapses to {0,1}); there the envelope's width comes entirely from the
    posterior component. Surfaced so reports can flag the deep tail."""


def bootstrap_ci_curve(
    tasks: list[TaskTrials],
    ks: list[int],
    statistic: str = "pass_hat_k",
    n_resamples: int = 2000,
    seed: int | None = None,
    level: float = 0.95,
) -> CurveCI:
    """CI band for the suite aggregate at each k: the element-wise envelope of
    a percentile cluster bootstrap and a posterior band on the same resamples.

    Construction (all on ONE set of task-resample indices, shared across k):

    1. **Cluster bootstrap** — resample tasks, mean of per-task point
       estimates per replicate, percentile interval per k.
    2. **Posterior band** — same task resamples, but each drawn task
       contributes p_i ~ Beta(c+1/2, n-c+1/2) (Jeffreys) and the replicate
       mean of p_i^k (or 1-(1-p_i)^k for pass@k).
    3. **Envelope** — low = min of the two lows, high = max of the two highs.

    Why the envelope instead of the bootstrap alone: near k = n the per-task
    estimator collapses to {0, 1}; when every estimate is equal the bootstrap
    band has zero width — a false claim of certainty — and widening only the
    degenerate k's would mix bound types along the curve, letting the upper
    band *rise* with k (asserting the impossible P(all k+1) > P(all k); found
    in adversarial review). Both component bands are monotone along k (every
    replicate curve is), so their envelope is monotone, never narrower than
    either component, and slightly conservative — the honest direction.

    Every task must have n >= max(ks); build per-k task subsets upstream
    (``decay_curve`` records them) and call once per subset if extending past
    min(n_i).
    """
    if not tasks:
        raise ValueError("no tasks supplied")
    if not ks:
        raise ValueError("no k values supplied")
    est_fn = {"pass_hat_k": pass_hat_k, "pass_at_k": pass_at_k}[statistic]
    n_common = min(t.n for t in tasks)
    bad = [k for k in ks if not 1 <= k <= n_common]
    if bad:
        raise ValueError(f"k values {bad} outside [1, min(n_i)={n_common}] for this task set")

    n_tasks = len(tasks)
    # (K, T) matrix of per-task point estimates
    est = np.array([[est_fn(t.n, t.c, k) for t in tasks] for k in ks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_tasks, size=(n_resamples, n_tasks))
    replicate_means = est[:, idx].mean(axis=2)  # (K, B)
    lo_q, hi_q = (0.5 - level / 2) * 100, (0.5 + level / 2) * 100
    boot_lows, boot_highs = np.percentile(replicate_means, [lo_q, hi_q], axis=1)

    # posterior component on the same resamples
    alpha_post = np.array([t.c + JEFFREYS_PRIOR[0] for t in tasks])
    beta_post = np.array([t.n - t.c + JEFFREYS_PRIOR[1] for t in tasks])
    p_draws = rng.beta(alpha_post[idx], beta_post[idx])  # (B, T)
    if statistic == "pass_hat_k":
        post_means = np.stack([(p_draws**k).mean(axis=1) for k in ks])  # (K, B)
    else:
        post_means = np.stack([(1 - (1 - p_draws) ** k).mean(axis=1) for k in ks])
    post_lows, post_highs = np.percentile(post_means, [lo_q, hi_q], axis=1)

    degenerate = tuple(
        k for k, lo, hi in zip(ks, boot_lows, boot_highs, strict=True) if lo == hi
    )
    lows = np.minimum(boot_lows, post_lows)
    highs = np.maximum(boot_highs, post_highs)

    intervals = {
        k: Interval(float(lo), float(hi), "bootstrap_posterior_envelope", level)
        for k, lo, hi in zip(ks, lows, highs, strict=True)
    }
    return CurveCI(
        intervals=intervals,
        n_tasks=n_tasks,
        small_sample_warning=n_tasks < SMALL_SAMPLE_TASKS,
        seed=seed,
        n_resamples=n_resamples,
        widened_ks=degenerate,
    )


def _validate(n: int, c: int) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n={n}], got {c}")
