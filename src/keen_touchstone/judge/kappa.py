"""The judge's exam: Cohen's κ + error profile against human-confirmed labels.

Why κ and not raw agreement: raw %-agreement doesn't correct for chance — a
judge that always says "pass" scores 80% agreement on an 80%-pass anchor set
while discriminating nothing. κ = (po − pe)/(1 − pe) subtracts the agreement
expected from the marginals alone (Cohen 1960; Landis-Koch bands: >0.6
acceptable, >0.8 strong).

The κ paradox, surfaced not hidden: at extreme base rates κ can be low despite
high raw agreement. That is why an ExamResult always carries prevalence,
TPR and FPR alongside κ — a single number is never the whole verdict.

Abstention: "unknown" is a legal judge answer. Abstained items are excluded
from every metric and counted separately — never coerced to failure (the same
honesty rule as Phase 1's unresolvable outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keen_touchstone.stats.intervals import Interval, beta_binomial_ci

MIN_CLASS_FOR_CI = 10
"""Adversarial-review round 2: the percentile bootstrap for κ undercovers
badly when the minority outcome class is tiny (empirical coverage as low as
~77% at 90% prevalence, n=30 — measured, not guessed). Below this many items
in the smaller class the κ CI is WITHHELD with a plain-language note instead
of shipping an interval that overstates confidence."""


@dataclass(frozen=True)
class ExamResult:
    """The judge's marks, always read together (κ + prevalence + TPR/FPR)."""

    n_total: int  # items presented
    n_scored: int  # items where the judge gave a definitive answer
    n_abstained: int
    abstention_rate: float
    prevalence: float | None  # base rate of human-pass among scored items
    raw_agreement: float | None
    kappa: float | None  # None: undefined (one-class anchor set / degenerate)
    kappa_ci: Interval | None
    tpr: float | None  # P(judge pass | human pass)
    tpr_ci: Interval | None
    fpr: float | None  # P(judge pass | human fail) — the rubber-stamp direction
    fpr_ci: Interval | None
    confusion: dict[str, int]  # tp / fp / tn / fn on scored items
    notes: list[str]


def cohen_kappa(human: np.ndarray, judge: np.ndarray) -> float | None:
    """Cohen's κ for paired binary labels. None when chance agreement pe = 1
    (both raters constant), where κ is mathematically undefined."""
    po = float(np.mean(human == judge))
    p_h, p_j = float(np.mean(human)), float(np.mean(judge))
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def _kappa_bootstrap_ci(
    human: np.ndarray,
    judge: np.ndarray,
    n_resamples: int,
    seed: int | None,
    level: float,
) -> tuple[Interval | None, int]:
    """Percentile bootstrap over ITEMS (paired resampling). Replicates where κ
    is undefined (single-class resample on both sides) are dropped and counted."""
    rng = np.random.default_rng(seed)
    n = len(human)
    idx = rng.integers(0, n, size=(n_resamples, n))
    h, j = human[idx], judge[idx]  # (B, n)
    po = (h == j).mean(axis=1)
    p_h, p_j = h.mean(axis=1), j.mean(axis=1)
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    valid = pe < 1.0
    n_dropped = int(n_resamples - valid.sum())
    if valid.sum() < n_resamples * 0.5:
        return None, n_dropped  # too degenerate to trust a percentile interval
    kappas = (po[valid] - pe[valid]) / (1 - pe[valid])
    lo_q, hi_q = (0.5 - level / 2) * 100, (0.5 + level / 2) * 100
    lo, hi = np.percentile(kappas, [lo_q, hi_q])
    return Interval(float(lo), float(hi), "bootstrap", level), n_dropped


def exam(
    human: list[bool],
    judge: list[bool | None],
    n_resamples: int = 2000,
    seed: int | None = 2026,
    level: float = 0.95,
) -> ExamResult:
    """Grade the judge against human-confirmed labels (one consensus label per
    item; consensus derivation and human-provenance enforcement live in the
    anchors layer)."""
    if len(human) != len(judge):
        raise ValueError(f"human ({len(human)}) and judge ({len(judge)}) label counts differ")
    if not human:
        raise ValueError("no items supplied")

    n_total = len(human)
    scored_pairs = [(h, j) for h, j in zip(human, judge, strict=True) if j is not None]
    n_abstained = n_total - len(scored_pairs)
    notes: list[str] = []

    if not scored_pairs:
        raise ValueError("the judge abstained on every item — nothing to grade")

    h = np.array([p[0] for p in scored_pairs], dtype=bool)
    j = np.array([p[1] for p in scored_pairs], dtype=bool)
    n_scored = len(h)

    tp = int(np.sum(h & j))
    fn = int(np.sum(h & ~j))
    fp = int(np.sum(~h & j))
    tn = int(np.sum(~h & ~j))
    prevalence = float(np.mean(h))
    raw_agreement = float(np.mean(h == j))

    kappa: float | None
    kappa_ci: Interval | None = None
    if prevalence in (0.0, 1.0):
        kappa = None
        notes.append(
            "anchor set contains only one outcome class among scored items — the exam "
            "cannot measure discrimination; add both real passes and real failures"
        )
    else:
        kappa = cohen_kappa(h, j)
        if kappa is None:  # pragma: no cover — unreachable when classes are mixed
            notes.append("kappa undefined (degenerate marginals)")
        else:
            minority = int(min(np.sum(h), np.sum(~h)))
            if minority < MIN_CLASS_FOR_CI:
                notes.append(
                    f"κ CI withheld: only {minority} item(s) in the minority outcome class — "
                    "a percentile bootstrap materially overstates confidence in this regime "
                    f"(measured coverage down to ~77%); label more "
                    f"{'failures' if np.sum(~h) < np.sum(h) else 'passes'} "
                    f"(need ≥ {MIN_CLASS_FOR_CI} per class)"
                )
            else:
                kappa_ci, n_dropped = _kappa_bootstrap_ci(h, j, n_resamples, seed, level)
                if kappa_ci is None:
                    notes.append(
                        "kappa CI unavailable: most bootstrap resamples were single-class "
                        "(anchor set too small or too imbalanced)"
                    )
                elif n_dropped:
                    notes.append(
                        f"kappa CI computed on {n_resamples - n_dropped}/{n_resamples} resamples "
                        "(single-class resamples dropped)"
                    )
        if min(prevalence, 1 - prevalence) < 0.15:
            notes.append(
                f"anchor set is imbalanced (pass rate {prevalence:.0%}): the kappa paradox "
                "applies — κ can look poor despite high agreement; read TPR/FPR alongside"
            )

    n_human_pass, n_human_fail = tp + fn, fp + tn
    tpr = tpr_ci = fpr = fpr_ci = None
    if n_human_pass:
        tpr = tp / n_human_pass
        tpr_ci = beta_binomial_ci(n_human_pass, tp, k=1, level=level)
    if n_human_fail:
        fpr = fp / n_human_fail
        fpr_ci = beta_binomial_ci(n_human_fail, fp, k=1, level=level)

    abstention_rate = n_abstained / n_total
    if abstention_rate > 0:
        notes.append(
            f"judge abstained on {n_abstained}/{n_total} items — abstentions are excluded "
            "from κ/TPR/FPR, never counted as wrong"
        )

    return ExamResult(
        n_total=n_total,
        n_scored=n_scored,
        n_abstained=n_abstained,
        abstention_rate=abstention_rate,
        prevalence=prevalence,
        raw_agreement=raw_agreement,
        kappa=kappa,
        kappa_ci=kappa_ci,
        tpr=tpr,
        tpr_ci=tpr_ci,
        fpr=fpr,
        fpr_ci=fpr_ci,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        notes=notes,
    )
