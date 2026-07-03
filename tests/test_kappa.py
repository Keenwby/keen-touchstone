"""Calibration stats core: κ goldens, invariants, sklearn parity, error profile."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from keen_touchstone.judge import cohen_kappa, exam

# ---------------------------------------------------------------- κ golden cases


def test_kappa_hand_computed_golden() -> None:
    # human=[T,T,T,F,F], judge=[T,T,F,F,F]:
    # po=0.8; marginals 0.6/0.4 -> pe=0.6*0.4+0.4*0.6=0.48; κ=(0.8-0.48)/0.52
    h = np.array([True, True, True, False, False])
    j = np.array([True, True, False, False, False])
    assert cohen_kappa(h, j) == pytest.approx((0.8 - 0.48) / 0.52)


def test_kappa_perfect_and_inverse() -> None:
    h = np.array([True, False, True, False])
    assert cohen_kappa(h, h.copy()) == pytest.approx(1.0)
    assert cohen_kappa(h, ~h) == pytest.approx(-1.0)  # balanced complement


def test_kappa_constant_judge_is_zero() -> None:
    # a judge that always says pass discriminates nothing: κ = 0 exactly
    h = np.array([True] * 8 + [False] * 2)
    j = np.array([True] * 10)
    assert cohen_kappa(h, j) == pytest.approx(0.0)


def test_kappa_undefined_when_both_constant() -> None:
    assert cohen_kappa(np.array([True, True]), np.array([True, True])) is None


def test_kappa_independent_labels_near_zero() -> None:
    rng = np.random.default_rng(7)
    h = rng.random(20000) < 0.6
    j = rng.random(20000) < 0.4  # independent of h
    assert abs(cohen_kappa(h, j)) < 0.02


# ------------------------------------------------------------------- invariants

pairs = st.lists(
    st.tuples(st.booleans(), st.booleans()), min_size=2, max_size=200
)


@given(pairs)
@settings(max_examples=200)
def test_kappa_bounds_and_symmetry(items: list[tuple[bool, bool]]) -> None:
    h = np.array([a for a, _ in items])
    j = np.array([b for _, b in items])
    k = cohen_kappa(h, j)
    if k is not None:
        assert -1.0 - 1e-12 <= k <= 1.0 + 1e-12
        assert cohen_kappa(j, h) == pytest.approx(k)  # symmetric in raters


@given(pairs)
@settings(max_examples=100, deadline=None)
def test_kappa_sklearn_parity(items: list[tuple[bool, bool]]) -> None:
    """Dev-dependency cross-check, mirroring the Inspect-parity pattern."""
    from sklearn.metrics import cohen_kappa_score

    h = np.array([a for a, _ in items])
    j = np.array([b for _, b in items])
    ours = cohen_kappa(h, j)
    theirs = cohen_kappa_score(h, j)
    if ours is None:
        assert np.isnan(theirs) or theirs == pytest.approx(1.0)
    elif not np.isnan(theirs):
        assert ours == pytest.approx(theirs, abs=1e-12)


# ------------------------------------------------------------------- exam()


def _labels(n: int, seed: int, p_pass: float = 0.6, agree: float = 0.9):
    """Human labels + a judge that agrees with probability `agree`."""
    rng = np.random.default_rng(seed)
    human = list(rng.random(n) < p_pass)
    judge = [h if rng.random() < agree else (not h) for h in human]
    return human, judge


def test_exam_full_profile() -> None:
    human, judge = _labels(60, seed=3)
    result = exam(human, judge, seed=3)
    assert result.n_total == result.n_scored == 60
    assert result.n_abstained == 0
    assert result.kappa is not None and 0 < result.kappa <= 1
    assert result.kappa_ci is not None
    assert result.kappa_ci.low <= result.kappa <= result.kappa_ci.high
    assert result.tpr is not None and result.fpr is not None
    assert result.tpr_ci.method == "beta_binomial"
    assert sum(result.confusion.values()) == 60
    assert result.raw_agreement > 0.8


def test_exam_reproducible() -> None:
    human, judge = _labels(50, seed=5)
    a, b = exam(human, judge, seed=11), exam(human, judge, seed=11)
    assert a == b


def test_exam_abstentions_excluded_and_counted() -> None:
    human, judge = _labels(40, seed=9)
    judge = [None if i % 4 == 0 else j for i, j in enumerate(judge)]
    result = exam(human, judge)
    assert result.n_abstained == 10
    assert result.n_scored == 30
    assert result.abstention_rate == pytest.approx(0.25)
    assert any("abstained" in n for n in result.notes)


def test_exam_one_class_anchor_set_refuses_kappa() -> None:
    result = exam([True] * 40, [True] * 35 + [False] * 5)
    assert result.kappa is None
    assert result.raw_agreement is not None
    assert any("one outcome class" in n for n in result.notes)


def test_exam_kappa_paradox_note_on_imbalance() -> None:
    # 92% pass base rate -> paradox warning even with decent agreement
    human = [True] * 46 + [False] * 4
    judge = list(human)
    result = exam(human, judge)
    assert any("paradox" in n for n in result.notes)


def test_exam_input_validation() -> None:
    with pytest.raises(ValueError, match="differ"):
        exam([True], [True, False])
    with pytest.raises(ValueError, match="no items"):
        exam([], [])
    with pytest.raises(ValueError, match="abstained on every"):
        exam([True, False], [None, None])


def test_exam_ci_width_shrinks_with_n() -> None:
    h1, j1 = _labels(40, seed=13)
    h2, j2 = _labels(400, seed=13)
    small = exam(h1, j1, seed=1).kappa_ci
    big = exam(h2, j2, seed=1).kappa_ci
    assert (big.high - big.low) < (small.high - small.low)


def test_exam_truth_recovery_of_designed_kappa() -> None:
    """Judge built with a designed confusion structure: with agree=0.9 and
    p_pass=0.5, expected κ ≈ 2*po−1 = 0.8; the estimate must land in the CI
    and the CI must contain the design target."""
    rng = np.random.default_rng(42)
    n = 400
    human = list(rng.random(n) < 0.5)
    judge = [h if rng.random() < 0.9 else (not h) for h in human]
    result = exam(human, judge, seed=42)
    assert result.kappa_ci.low <= 0.8 <= result.kappa_ci.high
