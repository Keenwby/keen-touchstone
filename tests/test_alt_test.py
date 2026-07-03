"""alt-test: hand-enumerated goldens (paper formulas), invariants, applicability."""

import numpy as np
import pytest

from keen_touchstone.judge import alt_test


def test_hand_enumerated_golden() -> None:
    """Fully hand-computed case (4 items, 3 annotators, judge = h1 = h2).

    j=1 and j=2: judge and excluded human align identically with the others
    on every item -> all ties, d_i = 0 -> degenerate branch, mean(d)=0 < 0.2
    -> p = 0. j=3: S_f = [1,1,1,1] vs S_h3 = [1,0,1,0] -> rho_f=1.0,
    rho_h=0.5, d = [0,-1,0,-1]; Wilcoxon (n=4 < 30) one-sided min p = 0.0625.
    BY-adjusted: [0, 0, 0.1146] at q=0.05 -> 2 of 3 rejected -> omega = 2/3,
    passed. rho-bar = mean(1,1,1) = 1.0.
    """
    human = {
        "h1": [True, True, False, False],
        "h2": [True, True, False, False],
        "h3": [True, False, False, True],
    }
    judge = [True, True, False, False]
    out = alt_test(human, judge, epsilon=0.2)
    assert out.applicable and out.passed
    assert out.omega == pytest.approx(2 / 3)
    assert out.avg_advantage_probability == pytest.approx(1.0)

    by_name = {a.annotator: a for a in out.annotators}
    assert by_name["h1"].test == "degenerate" and by_name["h1"].p_value == 0.0
    assert by_name["h1"].rho_llm == by_name["h1"].rho_human == 1.0
    assert by_name["h3"].test == "wilcoxon"
    assert by_name["h3"].rho_llm == pytest.approx(1.0)
    assert by_name["h3"].rho_human == pytest.approx(0.5)
    assert by_name["h3"].p_value == pytest.approx(0.0625)
    assert by_name["h3"].rejected is False  # 0.1146 > q after BY
    assert by_name["h1"].rejected and by_name["h2"].rejected


def test_bad_judge_fails() -> None:
    """A judge that contradicts the panel consensus on every item loses to
    every annotator."""
    rng = np.random.default_rng(4)
    n = 60
    truth = rng.random(n) < 0.5
    human = {
        f"h{j}": [bool(t) if rng.random() < 0.9 else bool(not t) for t in truth]
        for j in range(3)
    }
    judge = [bool(not t) for t in truth]
    out = alt_test(human, judge, epsilon=0.2)
    assert out.applicable
    assert out.omega == 0.0
    assert out.passed is False
    # anti-consensus judge still scores on split-panel ties (paper rule:
    # ties credit both sides), so rho-bar sits above 0 — but clearly below
    # "as good as a randomly chosen annotator"
    assert out.avg_advantage_probability < 0.5
    assert all(a.p_value > 0.9 for a in out.annotators)


def test_good_judge_passes_with_enough_items() -> None:
    """A judge as accurate as the panel (90% vs truth) should pass with
    n=200 items and the paper's epsilon=0.2 cost margin."""
    rng = np.random.default_rng(11)
    n = 200
    truth = rng.random(n) < 0.55
    def noisy(acc: float) -> list[bool]:
        return [bool(t) if rng.random() < acc else bool(not t) for t in truth]
    human = {"h1": noisy(0.85), "h2": noisy(0.85), "h3": noisy(0.85)}
    judge = noisy(0.9)
    out = alt_test(human, judge, epsilon=0.2)
    assert out.applicable and out.passed
    assert out.omega == 1.0
    assert all(a.test == "t" for a in out.annotators)  # n >= 30 -> t-test


def test_fewer_than_three_annotators_is_inapplicable() -> None:
    out = alt_test({"h1": [True, False], "h2": [True, False]}, [True, False])
    assert out.applicable is False
    assert "3 human annotators" in out.reason
    assert out.omega is None and out.passed is None


def test_missing_labels_and_judge_abstentions_handled() -> None:
    human = {
        "h1": [True, True, False, None, True] * 12,
        "h2": [True, None, False, False, True] * 12,
        "h3": [True, True, None, False, False] * 12,
    }
    judge = [True, True, False, None, True] * 12
    out = alt_test(human, judge, epsilon=0.2)
    assert out.applicable
    assert any("abstained" in n for n in out.notes)
    for a in out.annotators:
        assert a.n_items < 60  # missing/abstained items dropped per annotator


def test_annotators_without_comparable_items_are_skipped() -> None:
    human = {
        "h1": [True, False, True, False] * 10,
        "h2": [True, False, False, False] * 10,
        "h3": [True, True, True, False] * 10,
        "h4": [None] * 40,  # never labeled anything usable
    }
    judge = [True, False, True, False] * 10
    out = alt_test(human, judge)
    assert out.applicable
    assert {a.annotator for a in out.annotators} == {"h1", "h2", "h3"}
    assert any("skipped" in n for n in out.notes)


def test_epsilon_zero_with_zero_differences_does_not_crash() -> None:
    # d - epsilon can be all zeros when epsilon=0 and every comparison ties
    human = {
        "h1": [True, False] * 20,
        "h2": [True, False] * 20,
        "h3": [True, False] * 20,
    }
    judge = [True, False] * 20
    out = alt_test(human, judge, epsilon=0.0)
    assert out.applicable
    # all ties -> d = 0 -> degenerate branch: mean(d)=0 is NOT < 0 -> never rejected
    assert out.omega == 0.0


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="expected"):
        alt_test({"h1": [True], "h2": [True, False], "h3": [True]}, [True])
    with pytest.raises(ValueError, match="test_method"):
        alt_test({"h1": [True], "h2": [True], "h3": [True]}, [True], test_method="bogus")


def test_deterministic() -> None:
    human = {
        "h1": [True, False, True] * 15,
        "h2": [True, True, False] * 15,
        "h3": [False, False, True] * 15,
    }
    judge = [True, False, False] * 15
    assert alt_test(human, judge) == alt_test(human, judge)
