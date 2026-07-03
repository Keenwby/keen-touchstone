"""Estimator correctness: golden cases, invariants, unbiasedness, Inspect parity."""

from itertools import pairwise
from math import comb

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keen_touchstone.stats import (
    TaskTrials,
    aggregate_pass_hat_k,
    decay_curve,
    max_common_k,
    pass_at_k,
    pass_hat_k,
)

# ---------------------------------------------------------------- golden cases


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (5, 3, 2, 3 / 10),  # C(3,2)/C(5,2) = 3/10
        (5, 3, 3, 1 / 10),  # C(3,3)/C(5,3) = 1/10
        (4, 2, 2, 1 / 6),
        (10, 10, 7, 1.0),  # always succeeds
        (10, 0, 1, 0.0),  # never succeeds
        (10, 3, 5, 0.0),  # c < k -> impossible to draw k successes
        (8, 4, 1, 0.5),  # k=1 is the pass rate
    ],
)
def test_pass_hat_k_golden(n: int, c: int, k: int, expected: float) -> None:
    assert pass_hat_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (5, 3, 2, 9 / 10),  # 1 - C(2,2)/C(5,2)
        (4, 2, 2, 5 / 6),
        (10, 0, 4, 0.0),
        (10, 10, 1, 1.0),
        (8, 4, 1, 0.5),
    ],
)
def test_pass_at_k_golden(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize("fn", [pass_hat_k, pass_at_k])
def test_domain_errors(fn) -> None:
    with pytest.raises(ValueError):
        fn(5, 3, 6)  # k > n
    with pytest.raises(ValueError):
        fn(5, 3, 0)  # k < 1
    with pytest.raises(ValueError):
        fn(5, 6, 2)  # c > n
    with pytest.raises(ValueError):
        fn(0, 0, 1)  # n < 1


# ------------------------------------------------------------------ invariants

nck = st.integers(1, 60).flatmap(
    lambda n: st.tuples(st.just(n), st.integers(0, n), st.integers(1, n))
)


@given(nck)
def test_bounds_and_ordering(nck_tuple: tuple[int, int, int]) -> None:
    n, c, k = nck_tuple
    hat, at, rate = pass_hat_k(n, c, k), pass_at_k(n, c, k), c / n
    assert 0.0 <= hat <= 1.0
    assert 0.0 <= at <= 1.0
    # reliability <= capability, with the pass rate between them
    assert hat <= rate + 1e-12
    assert rate <= at + 1e-12


@given(nck)
def test_monotone_in_k(nck_tuple: tuple[int, int, int]) -> None:
    n, c, k = nck_tuple
    if k < n:
        assert pass_hat_k(n, c, k + 1) <= pass_hat_k(n, c, k) + 1e-12
        assert pass_at_k(n, c, k + 1) >= pass_at_k(n, c, k) - 1e-12


@given(nck)
def test_k_equals_1_is_pass_rate(nck_tuple: tuple[int, int, int]) -> None:
    n, c, _ = nck_tuple
    assert pass_hat_k(n, c, 1) == pytest.approx(c / n)
    assert pass_at_k(n, c, 1) == pytest.approx(c / n)


# ------------------------------------------------------------- unbiasedness MC


def test_unbiased_for_p_to_the_k() -> None:
    """E[C(c,k)/C(n,k)] over c~Binomial(n,p) equals p^k exactly (computed, not sampled)."""
    p, n, k = 0.7, 10, 3
    expectation = sum(
        comb(n, c) * p**c * (1 - p) ** (n - c) * pass_hat_k(n, c, k) for c in range(n + 1)
    )
    assert expectation == pytest.approx(p**k, abs=1e-12)


def test_pass_at_unbiased_complement() -> None:
    p, n, k = 0.6, 12, 4
    expectation = sum(
        comb(n, c) * p**c * (1 - p) ** (n - c) * pass_at_k(n, c, k) for c in range(n + 1)
    )
    assert expectation == pytest.approx(1 - (1 - p) ** k, abs=1e-12)


# -------------------------------------------------------------- Inspect parity


@given(nck)
@settings(deadline=None, max_examples=50)
def test_parity_with_inspect_reducers(nck_tuple: tuple[int, int, int]) -> None:
    """Same metric definition as Inspect AI's pass_k_{k}/pass_at_{k} — the
    'one metric definition' invariant, checked against the installed package."""
    from inspect_ai.scorer import Score, pass_at, pass_k

    n, c, k = nck_tuple
    scores = [Score(value=1.0)] * c + [Score(value=0.0)] * (n - c)
    assert pass_k(k)(scores).value == pytest.approx(pass_hat_k(n, c, k), abs=1e-12)
    assert pass_at(k)(scores).value == pytest.approx(pass_at_k(n, c, k), abs=1e-12)


# ----------------------------------------------------------------- decay curve


def _t(key: str, n: int, c: int) -> TaskTrials:
    return TaskTrials(task_key=key, n=n, c=c)


def test_decay_curve_default_k_max_is_min_n() -> None:
    tasks = [_t("a", 5, 4), _t("b", 8, 8)]
    points = decay_curve(tasks)
    assert max_common_k(tasks) == 5
    assert [pt.k for pt in points] == [1, 2, 3, 4, 5]
    assert all(pt.n_tasks == 2 and pt.dropped_task_keys == () for pt in points)
    # aggregate at k=1 is the mean pass rate
    assert points[0].pass_hat_k == pytest.approx((4 / 5 + 1.0) / 2)


def test_decay_curve_monotone() -> None:
    tasks = [_t("a", 10, 7), _t("b", 10, 9), _t("c", 10, 3)]
    points = decay_curve(tasks)
    hats = [pt.pass_hat_k for pt in points]
    ats = [pt.pass_at_k for pt in points]
    assert all(x >= y - 1e-12 for x, y in pairwise(hats))
    assert all(x <= y + 1e-12 for x, y in pairwise(ats))


def test_decay_curve_task_drop_is_optin_and_recorded() -> None:
    tasks = [_t("small", 3, 2), _t("big", 6, 5)]
    with pytest.raises(ValueError, match="allow_task_drop"):
        decay_curve(tasks, k_max=6)
    points = decay_curve(tasks, k_max=6, allow_task_drop=True)
    assert points[2].dropped_task_keys == ()  # k=3: both tasks
    assert points[3].dropped_task_keys == ("small",)  # k=4: small lacks trials
    assert points[3].n_tasks == 1
    with pytest.raises(ValueError, match="max"):
        decay_curve(tasks, k_max=7, allow_task_drop=True)  # beyond every task


def test_aggregate_requires_enough_trials() -> None:
    with pytest.raises(ValueError):
        aggregate_pass_hat_k([_t("a", 3, 2)], k=4)


def test_task_trials_validation() -> None:
    with pytest.raises(ValueError):
        _t("bad", 0, 0)
    with pytest.raises(ValueError):
        _t("bad", 3, 4)
    with pytest.raises(ValueError):
        TaskTrials(task_key="bad", n=3, c=2, tokens=(1, 2))  # wrong length
