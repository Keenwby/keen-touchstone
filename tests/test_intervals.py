"""Interval correctness: golden properties, reproducibility, and empirical coverage."""

import numpy as np
import pytest

from keen_touchstone.stats import (
    TaskTrials,
    aggregate_pass_hat_k,
    beta_binomial_ci,
    bootstrap_ci_curve,
    wilson_ci,
)

# ---------------------------------------------------------------------- wilson


def test_wilson_basic_properties() -> None:
    ci = wilson_ci(10, 7)
    assert 0.0 <= ci.low <= 0.7 <= ci.high <= 1.0
    assert ci.method == "wilson"


def test_wilson_edge_counts() -> None:
    lo_all_fail = wilson_ci(10, 0)
    assert lo_all_fail.low == pytest.approx(0.0)
    assert lo_all_fail.high > 0.0  # never a zero-width interval at the edge
    hi_all_pass = wilson_ci(10, 10)
    assert hi_all_pass.high == pytest.approx(1.0)
    assert hi_all_pass.low < 1.0


# --------------------------------------------------------------- beta-binomial


def test_beta_binomial_contains_point_estimate() -> None:
    ci = beta_binomial_ci(20, 14, k=1)
    assert ci.low < 14 / 20 < ci.high
    assert ci.method == "beta_binomial"


def test_beta_binomial_k_transform_is_exact_power() -> None:
    base = beta_binomial_ci(12, 9, k=1)
    powered = beta_binomial_ci(12, 9, k=4)
    assert powered.low == pytest.approx(base.low**4)
    assert powered.high == pytest.approx(base.high**4)


def test_beta_binomial_shifts_down_with_k() -> None:
    k1, k5 = beta_binomial_ci(10, 8, k=1), beta_binomial_ci(10, 8, k=5)
    assert k5.high < k1.high
    assert k5.low < k1.low


# ------------------------------------------------------------------- bootstrap


def _suite(seed: int, n_tasks: int = 12, n: int = 10) -> list[TaskTrials]:
    rng = np.random.default_rng(seed)
    ps = rng.uniform(0.3, 0.95, size=n_tasks)
    return [
        TaskTrials(task_key=f"t{i}", n=n, c=int(rng.binomial(n, p))) for i, p in enumerate(ps)
    ]


def test_bootstrap_reproducible_and_ordered() -> None:
    tasks = _suite(1)
    a = bootstrap_ci_curve(tasks, ks=[1, 2, 3], seed=42)
    b = bootstrap_ci_curve(tasks, ks=[1, 2, 3], seed=42)
    assert a.intervals == b.intervals
    for k, ci in a.intervals.items():
        point = aggregate_pass_hat_k(tasks, k)
        assert ci.low <= point <= ci.high
        assert ci.method == "bootstrap"
    # the band is coherent along k: bounds decay like the point estimate does
    assert a.intervals[3].high <= a.intervals[1].high + 1e-12
    assert a.intervals[3].low <= a.intervals[1].low + 1e-12


def test_bootstrap_width_shrinks_with_more_tasks() -> None:
    small = _suite(7, n_tasks=8)
    big = small * 10  # same composition, 10x the tasks
    w_small = (lambda ci: ci.high - ci.low)(bootstrap_ci_curve(small, ks=[3], seed=0).intervals[3])
    w_big = (lambda ci: ci.high - ci.low)(bootstrap_ci_curve(big, ks=[3], seed=0).intervals[3])
    assert w_big < w_small / 2


def test_bootstrap_small_sample_warning() -> None:
    tasks = _suite(3, n_tasks=3)
    assert bootstrap_ci_curve(tasks, ks=[1], seed=0).small_sample_warning is True
    assert bootstrap_ci_curve(_suite(3, n_tasks=12), ks=[1], seed=0).small_sample_warning is False


def test_degenerate_bootstrap_is_widened_not_certain() -> None:
    """All per-task estimates identical (here: all zero) would give a [0, 0]
    'CI' — a false claim of certainty. The guard widens with posterior draws."""
    tasks = [TaskTrials(f"t{i}", 10, 0) for i in range(6)]
    band = bootstrap_ci_curve(tasks, ks=[1, 2], seed=1)
    assert band.widened_ks == (1, 2)
    for k in (1, 2):
        ci = band.intervals[k]
        assert ci.high > ci.low  # no longer zero width
        assert ci.high > 0.0  # admits the estimand may be positive
        assert ci.high < 0.35  # but stays anchored to the data (0/10 x 6 tasks)


def test_nondegenerate_bootstrap_untouched() -> None:
    band = bootstrap_ci_curve(_suite(1), ks=[1, 2, 3], seed=42)
    assert band.widened_ks == ()


def test_bootstrap_rejects_k_beyond_common_range() -> None:
    tasks = [TaskTrials("a", 4, 3), TaskTrials("b", 9, 9)]
    with pytest.raises(ValueError, match="min"):
        bootstrap_ci_curve(tasks, ks=[5], seed=0)


def test_bootstrap_coverage_close_to_nominal() -> None:
    """Empirical coverage of the 95% task-bootstrap CI vs the analytic
    superpopulation truth E[p^k] with p ~ U(a, b): (b^{k+1}-a^{k+1})/((k+1)(b-a)).

    Percentile bootstrap at T=40 typically sits slightly below nominal;
    the assertion bounds are deliberately loose to stay non-flaky while still
    catching gross errors (e.g. pooled-attempt resampling would collapse the
    interval and fail this hard).
    """
    a, b, k, n, n_tasks, reps = 0.4, 0.95, 3, 10, 40, 200
    true_value = (b ** (k + 1) - a ** (k + 1)) / ((k + 1) * (b - a))
    rng = np.random.default_rng(2026)
    hits = 0
    for _ in range(reps):
        ps = rng.uniform(a, b, size=n_tasks)
        tasks = [
            TaskTrials(task_key=f"t{i}", n=n, c=int(rng.binomial(n, p)))
            for i, p in enumerate(ps)
        ]
        ci = bootstrap_ci_curve(tasks, ks=[k], n_resamples=500, seed=int(rng.integers(2**31)))
        if ci.intervals[k].low <= true_value <= ci.intervals[k].high:
            hits += 1
    coverage = hits / reps
    assert 0.85 <= coverage <= 1.0, f"coverage {coverage:.3f} far from nominal 0.95"
