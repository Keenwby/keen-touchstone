"""License issuance and the CI-blocking gate.

This module is the moat: everyone computes agreement, nobody blocks on it.
``issue_license`` turns an exam into an auditable JudgeCalibration artifact;
``check_license`` is the gate — NEEDS_HUMAN (or a missing/invalid license)
must fail the build.

Licensing rules, in order (all failures accumulate into ``reasons``):

1. Fewer than ``min_items`` human-labeled items → the κ point estimate is
   WITHHELD entirely (a κ from 12 items is noise wearing a number) and the
   judge cannot be licensed. TPR/FPR still shown — their wide Jeffreys CIs
   tell the truth on their own.
2. κ undefined (one-class anchor set) → cannot be licensed.
3. Abstention rate above ``max_abstention`` → cannot be licensed (a judge
   that shrugs at a fifth of real cases is not a gate you can rely on).
4. κ below ``kappa_licensed`` (on the point estimate by default; on the CI
   lower bound in strict ``gate_on="ci_low"`` mode) → cannot be licensed.
5. If the alt-test ran (≥3 annotators) and FAILED, the judge cannot be
   licensed: a direct statistical verdict that it may not replace your
   annotators outranks a passable κ.

A license that passes on the point estimate while the CI still includes
sub-threshold values is granted but says so out loud.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from keen_touchstone import __version__
from keen_touchstone.artifacts import (
    AltTestResult,
    CalibrationThresholds,
    JudgeCalibration,
)

from .alt_test import AltTestOutcome
from .kappa import ExamResult


def _alt_result(alt: AltTestOutcome | None) -> AltTestResult | None:
    if alt is None:
        return None
    return AltTestResult(
        applicable=alt.applicable,
        reason=alt.reason,
        epsilon=alt.epsilon,
        q=alt.q,
        omega=alt.omega,
        passed=alt.passed,
        avg_advantage_probability=alt.avg_advantage_probability,
        n_annotators=len(alt.annotators) if alt.applicable else None,
    )


def issue_license(
    exam: ExamResult,
    judge_id: str,
    alt: AltTestOutcome | None = None,
    judge_model: str | None = None,
    judge_prompt_hash: str | None = None,
    anchor_set_ref: str | None = None,
    n_human_annotators: int = 1,
    thresholds: CalibrationThresholds | None = None,
) -> JudgeCalibration:
    thresholds = thresholds or CalibrationThresholds()
    reasons: list[str] = list(exam.notes)
    blocking: list[str] = []

    withhold_kappa = exam.n_total < thresholds.min_items
    if withhold_kappa:
        blocking.append(
            f"only {exam.n_total} human-labeled items (< {thresholds.min_items}): calibration "
            "unreliable — κ point estimate withheld; label more real cases (Anthropic floor: "
            "20–50 from real failures, ≥30 for a usable κ)"
        )
    elif exam.kappa is None:
        blocking.append("κ undefined on this anchor set — see notes above")
    else:
        gate_value = exam.kappa
        gate_label = "point estimate"
        if thresholds.gate_on == "ci_low":
            if exam.kappa_ci is None:
                blocking.append("strict mode gates on the κ CI lower bound, but no CI is available")
                gate_value = None
            else:
                gate_value = exam.kappa_ci.low
                gate_label = "CI lower bound"
        if gate_value is not None and gate_value < thresholds.kappa_licensed:
            blocking.append(
                f"κ {gate_label} {gate_value:.3f} is below the licensing threshold "
                f"{thresholds.kappa_licensed} — the judge does not agree with humans enough "
                "to be trusted unsupervised"
            )
        elif (
            thresholds.gate_on == "point"
            and exam.kappa_ci is not None
            and exam.kappa_ci.low < thresholds.kappa_licensed
        ):
            reasons.append(
                f"licensed on the κ point estimate ({exam.kappa:.3f}), but the 95% CI "
                f"[{exam.kappa_ci.low:.3f}, {exam.kappa_ci.high:.3f}] still includes "
                "sub-threshold values — more anchor items would firm this up"
            )

    if exam.abstention_rate > thresholds.max_abstention:
        blocking.append(
            f"abstention rate {exam.abstention_rate:.0%} exceeds {thresholds.max_abstention:.0%} "
            "— a judge that shrugs this often cannot carry a gate"
        )
    if alt is not None and alt.applicable and alt.passed is False:
        blocking.append(
            f"alt-test FAILED (ω={alt.omega:.2f} < 0.5): statistically, this judge may not "
            "replace your human annotators (ACL 2025 criterion) — this outranks κ"
        )

    status = "NEEDS_HUMAN" if blocking else "JUDGE_LICENSED"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    calibration_id = "cal-" + hashlib.sha256(
        f"{judge_id}|{judge_prompt_hash}|{anchor_set_ref}|{created_at}".encode()
    ).hexdigest()[:12]

    return JudgeCalibration(
        calibration_id=calibration_id,
        judge_id=judge_id,
        judge_model=judge_model,
        judge_prompt_hash=judge_prompt_hash,
        anchor_set_ref=anchor_set_ref,
        anchor_n_items=exam.n_total,
        n_human_annotators=n_human_annotators,
        prevalence=exam.prevalence,
        kappa=None if withhold_kappa else exam.kappa,
        kappa_ci_low=None if withhold_kappa or exam.kappa_ci is None else exam.kappa_ci.low,
        kappa_ci_high=None if withhold_kappa or exam.kappa_ci is None else exam.kappa_ci.high,
        raw_agreement=exam.raw_agreement,
        tpr=exam.tpr,
        tpr_ci_low=exam.tpr_ci.low if exam.tpr_ci else None,
        tpr_ci_high=exam.tpr_ci.high if exam.tpr_ci else None,
        fpr=exam.fpr,
        fpr_ci_low=exam.fpr_ci.low if exam.fpr_ci else None,
        fpr_ci_high=exam.fpr_ci.high if exam.fpr_ci else None,
        abstention_rate=exam.abstention_rate,
        alt_test=_alt_result(alt),
        thresholds=thresholds,
        status=status,
        reasons=blocking + reasons,
        created_at=created_at,
        keen_touchstone_version=__version__,
    )


def check_license(calibration: JudgeCalibration) -> tuple[bool, str]:
    """The gate. Returns (ok, plain-language message). Callers (CLI, ingest)
    must treat ok=False as build-blocking."""
    if calibration.status == "JUDGE_LICENSED":
        kappa = f"κ={calibration.kappa:.3f}" if calibration.kappa is not None else "κ withheld"
        return True, (
            f"JUDGE_LICENSED: {calibration.judge_id} ({kappa}, "
            f"n={calibration.anchor_n_items}, issued {calibration.created_at})"
        )
    why = "; ".join(calibration.reasons[:3]) or "no reasons recorded"
    return False, (
        f"NEEDS_HUMAN: {calibration.judge_id} is not licensed — {why}. "
        "Scores from this judge are not evidence; fix the judge or label more anchors."
    )
