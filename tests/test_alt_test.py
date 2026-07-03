"""alt-test: hand-enumerated goldens (paper formulas), invariants, applicability."""

import numpy as np
import pytest

from keen_touchstone.judge import alt_test


def test_hand_enumerated_golden() -> None:
    """Hand-computed case: the 4-item pattern tripled to 12 items (floor: 10).

    judge = h1 = h2; h3 disagrees on 2 of every 4 items.
    j=1/j=2 (excluded): judge and excluded human align identically with the
    others on every item -> all ties, d = 0 -> exact conservative bound
    p = (1-ε)^n = 0.8^12 ≈ 0.0687 (adversarial-review round 2: parity is weak
    evidence, NEVER a fabricated p=0).
    j=3: S_f = 1 on all items vs S_h3 ∈ {1,0} -> rho_f=1.0, rho_h=0.5,
    d = [0,-1,0,-1]×3 -> Wilcoxon, p ≪ 0.05, survives BY (rank 1).
    ω = 1/3 < 0.5 -> the alt-test now honestly FAILS this panel: beating one
    annotator while merely tying the other two is not "may replace humans".
    """
    human = {
        "h1": [True, True, False, False] * 3,
        "h2": [True, True, False, False] * 3,
        "h3": [True, False, False, True] * 3,
    }
    judge = [True, True, False, False] * 3
    out = alt_test(human, judge, epsilon=0.2)
    assert out.applicable
    assert out.passed is False
    assert out.omega == pytest.approx(1 / 3)
    assert out.avg_advantage_probability == pytest.approx(1.0)

    by_name = {a.annotator: a for a in out.annotators}
    assert by_name["h1"].test == "exact_bound"
    assert by_name["h1"].p_value == pytest.approx(0.8**12)
    assert by_name["h1"].rejected is False
    assert by_name["h1"].rho_llm == by_name["h1"].rho_human == 1.0
    assert by_name["h3"].test == "wilcoxon"
    assert by_name["h3"].rho_llm == pytest.approx(1.0)
    assert by_name["h3"].rho_human == pytest.approx(0.5)
    assert by_name["h3"].rejected is True
    assert any("Bernoulli bound" in n for n in out.notes)


def test_tiny_panels_are_underpowered_not_passed() -> None:
    """Round-2 regression: 3 items must NEVER yield 'may replace humans'."""
    human = {
        "h1": [True, True, False],
        "h2": [True, True, False],
        "h3": [True, True, False],
    }
    out = alt_test(human, [True, True, False], epsilon=0.2)
    assert out.applicable is False
    assert "underpowered" in out.reason
    assert out.passed is None and out.omega is None


def test_exact_bound_never_below_wilcoxon_floor_ordering() -> None:
    """Round-2 regression: the evidence ordering must not invert — an all-tie
    panel (zero advantage) can never be MORE significant than a panel where
    the judge strictly wins items. All-ties at n=12, ε=0.2 gives 0.8^12≈0.069
    (not significant); a clean sweep gives (0.4)^12 ≈ 1.7e-5 (significant)."""
    ties = {
        "h1": [True, False] * 6, "h2": [True, False] * 6, "h3": [True, False] * 6,
    }
    out_ties = alt_test(ties, [True, False] * 6, epsilon=0.2)
    for a in out_ties.annotators:
        assert a.p_value == pytest.approx(0.8**12)
        assert a.rejected is False
    assert out_ties.passed is False

    # judge == h1 == h2; h3 is the complement -> excluding h3, judge sweeps
    sweep = {
        "h1": [True, False] * 6,
        "h2": [True, False] * 6,
        "h3": [False, True] * 6,
    }
    out_sweep = alt_test(sweep, [True, False] * 6, epsilon=0.2)
    h3 = {a.annotator: a for a in out_sweep.annotators}["h3"]
    assert h3.test == "exact_bound"
    assert h3.p_value == pytest.approx(0.4**12)
    assert h3.rejected is True


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
