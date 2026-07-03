"""Which-lever variance decomposition and the power heuristic."""

import pytest

from keen_touchstone.stats import (
    Interval,
    TaskTrials,
    power_status,
    variance_decomposition,
)


def test_homogeneous_tasks_point_at_rollouts_lever() -> None:
    # identical observed rates -> zero between-task variance; the only noise
    # left is within-task sampling -> spend budget on more rollouts per task
    tasks = [TaskTrials(f"t{i}", 10, 5) for i in range(8)]
    d = variance_decomposition(tasks)
    assert d.observed_between_task_variance == pytest.approx(0.0)
    assert d.est_true_between_task_variance == pytest.approx(0.0)
    assert d.mean_within_task_variance > 0
    assert d.lever == "more_rollouts_per_task"


def test_heterogeneous_tasks_with_huge_n_point_at_tasks_lever() -> None:
    # rates spread 0.1..0.9 measured with n=1000 -> within-task noise is tiny;
    # the spread is real task heterogeneity -> more tasks is the lever
    rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    tasks = [TaskTrials(f"t{i}", 1000, int(1000 * r)) for i, r in enumerate(rates)]
    d = variance_decomposition(tasks)
    assert d.lever == "more_tasks"
    assert d.est_true_between_task_variance > d.mean_within_task_variance


def test_single_trial_tasks_are_flagged_not_crashed() -> None:
    tasks = [TaskTrials("a", 1, 1), TaskTrials("b", 10, 5)]
    d = variance_decomposition(tasks)
    assert d.n_tasks_excluded_within == 1
    assert d.mean_within_task_variance is not None  # from task b only


def test_degenerate_inputs() -> None:
    d = variance_decomposition([TaskTrials("only", 10, 6)])
    assert d.observed_between_task_variance is None
    assert d.lever is None
    assert d.skew is None
    with pytest.raises(ValueError):
        variance_decomposition([])


def test_power_status_heuristic() -> None:
    assert power_status(None) is None
    assert power_status(Interval(0.5, 0.6, "bootstrap")) is None
    assert power_status(Interval(0.3, 0.6, "bootstrap")) == "UNDERPOWERED_NEED_MORE_N"
