"""The alternative annotator test (alt-test): may this judge replace YOUR annotators?

Implemented from the paper's formulas alone — Calderon, Reichart & Dror,
"The Alternative Annotator Test for LLM-as-a-Judge" (ACL 2025,
arXiv 2501.10970). The authors' reference repository carries no license, so
neither its code nor its fixtures are used here; golden tests are
hand-enumerated.

Procedure (paper §3, binary-label instantiation):

1. Leave one annotator j out. Per item i, the alignment score S is the share
   of the REMAINING annotators an annotation agrees with — computed for the
   judge, S(f, x_i, j), and for the excluded human, S(h_j, x_i, j).
2. Indicators W^f (judge at least as aligned; ties score for both sides) and
   W^h; advantage probabilities ρ^f_j = mean(W^f), ρ^h_j = mean(W^h).
3. Per annotator, test H0: ρ^f_j ≤ ρ^h_j − ε (ε = cost-benefit margin
   favoring the cheaper judge; the paper uses 0.1–0.2, most datasets 0.2).
   Paired one-sided test on d_i = W^h_i − W^f_i against ε: t-test when
   n ≥ 30, Wilcoxon signed-rank below (paper's normality rule).
4. Benjamini–Yekutieli FDR correction across the m hypotheses (they are
   dependent — scores share annotators), target q = 0.05.
5. Winning rate ω = share of rejected H0. ω ≥ 0.5 ⇒ the judge may replace
   the annotators. Also reported: the average advantage probability
   ρ̄ = mean(ρ^f_j) — "how likely is the judge to be at least as good as a
   randomly chosen annotator".

Applicability: the paper requires **≥ 3 human annotators** (with fewer, the
"remaining annotators" panel is a single person and the comparison collapses).
With < 3 usable annotators the result is `applicable=False` — reported
honestly, never computed anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

MIN_ANNOTATORS = 3
T_TEST_MIN_N = 30  # paper: below this, normality is doubtful -> Wilcoxon


@dataclass(frozen=True)
class AnnotatorOutcome:
    annotator: str
    n_items: int  # comparable items for this annotator
    rho_llm: float  # ρ^f_j — P(judge at least as aligned as h_j)
    rho_human: float  # ρ^h_j
    p_value: float
    test: str  # "t" | "wilcoxon" | "degenerate"
    rejected: bool  # H0 rejected after BY correction


@dataclass(frozen=True)
class AltTestOutcome:
    applicable: bool
    reason: str | None
    epsilon: float
    q: float
    omega: float | None
    passed: bool | None  # ω >= 0.5 (the paper's criterion)
    avg_advantage_probability: float | None  # ρ̄
    annotators: list[AnnotatorOutcome]
    notes: list[str]


def _inapplicable(reason: str, epsilon: float, q: float, notes: list[str]) -> AltTestOutcome:
    return AltTestOutcome(
        applicable=False, reason=reason, epsilon=epsilon, q=q,
        omega=None, passed=None, avg_advantage_probability=None,
        annotators=[], notes=notes,
    )


def _one_sided_p(d: np.ndarray, epsilon: float, test_method: str) -> tuple[float, str]:
    """Left-tailed test of H0: mean(d) >= epsilon, per the paper."""
    if np.all(d == d[0]):
        # zero spread: the sample mean IS the observed statistic; the null is
        # contradicted outright or not at all
        return (0.0 if float(d[0]) < epsilon else 1.0), "degenerate"
    n = len(d)
    use_t = test_method == "t" or (test_method == "auto" and n >= T_TEST_MIN_N)
    if use_t:
        result = scipy_stats.ttest_1samp(d, popmean=epsilon, alternative="less")
        return float(result.pvalue), "t"
    result = scipy_stats.wilcoxon(d - epsilon, alternative="less")
    return float(result.pvalue), "wilcoxon"


def alt_test(
    human_labels: Mapping[str, Sequence[bool | None]],
    judge_labels: Sequence[bool | None],
    epsilon: float = 0.2,
    q: float = 0.05,
    test_method: str = "auto",  # "auto" (paper rule) | "t" | "wilcoxon"
) -> AltTestOutcome:
    """Run the alt-test on binary labels. ``None`` = not labeled/abstained.

    ``human_labels`` maps annotator name -> per-item labels (all sequences the
    same length as ``judge_labels``).
    """
    if test_method not in ("auto", "t", "wilcoxon"):
        raise ValueError(f"unknown test_method {test_method!r}")
    n_items_total = len(judge_labels)
    for name, labels in human_labels.items():
        if len(labels) != n_items_total:
            raise ValueError(
                f"annotator {name!r} has {len(labels)} labels, expected {n_items_total}"
            )

    notes: list[str] = []
    judge_abstained = [i for i, label in enumerate(judge_labels) if label is None]
    if judge_abstained:
        notes.append(
            f"judge abstained on {len(judge_abstained)}/{n_items_total} items — those items "
            "are excluded from the alt-test"
        )

    names = sorted(human_labels)
    if len(names) < MIN_ANNOTATORS:
        return _inapplicable(
            f"alt-test needs >= {MIN_ANNOTATORS} human annotators (leave-one-out requires a "
            f"panel); got {len(names)}",
            epsilon, q, notes,
        )

    per_annotator: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for j in names:
        w_f: list[int] = []
        w_h: list[int] = []
        for i in range(n_items_total):
            f_i = judge_labels[i]
            h_ji = human_labels[j][i]
            if f_i is None or h_ji is None:
                continue
            others = [
                human_labels[k][i]
                for k in names
                if k != j and human_labels[k][i] is not None
            ]
            if not others:
                continue  # nobody to align against on this item
            s_f = float(np.mean([f_i == o for o in others]))
            s_h = float(np.mean([h_ji == o for o in others]))
            w_f.append(int(s_f >= s_h))
            w_h.append(int(s_h >= s_f))
        if len(w_f) < 2:
            skipped.append(j)
            continue
        per_annotator.append((j, np.array(w_f), np.array(w_h), np.array(w_h) - np.array(w_f)))

    if skipped:
        notes.append(
            f"annotator(s) {skipped} skipped: fewer than 2 comparable items"
        )
    if len(per_annotator) < MIN_ANNOTATORS:
        return _inapplicable(
            f"only {len(per_annotator)} annotator(s) with >= 2 comparable items "
            f"(need {MIN_ANNOTATORS})",
            epsilon, q, notes,
        )

    p_values: list[float] = []
    tests: list[str] = []
    for _, _, _, d in per_annotator:
        p, test = _one_sided_p(d, epsilon, test_method)
        p_values.append(p)
        tests.append(test)

    adjusted = scipy_stats.false_discovery_control(p_values, method="by")
    rejected = [float(p_adj) <= q for p_adj in adjusted]

    outcomes = [
        AnnotatorOutcome(
            annotator=name,
            n_items=len(w_f),
            rho_llm=float(w_f.mean()),
            rho_human=float(w_h.mean()),
            p_value=p,
            test=test,
            rejected=rej,
        )
        for (name, w_f, w_h, _), p, test, rej in zip(
            per_annotator, p_values, tests, rejected, strict=True
        )
    ]
    omega = float(np.mean(rejected))
    rho_bar = float(np.mean([o.rho_llm for o in outcomes]))
    return AltTestOutcome(
        applicable=True,
        reason=None,
        epsilon=epsilon,
        q=q,
        omega=omega,
        passed=omega >= 0.5,
        avg_advantage_probability=rho_bar,
        annotators=outcomes,
        notes=notes,
    )
