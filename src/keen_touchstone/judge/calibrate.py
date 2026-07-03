"""Orchestrate one full exam: anchors + judge labels → license.

The seam between the pieces: consensus labels feed the κ exam; the full
annotator matrix feeds the alt-test (when ≥3 annotators); everything lands in
one auditable JudgeCalibration.
"""

from __future__ import annotations

from collections.abc import Mapping

from keen_touchstone.artifacts import CalibrationThresholds, JudgeCalibration

from .alt_test import AltTestOutcome, alt_test
from .anchors import AnchorSet
from .kappa import ExamResult, exam
from .license import issue_license


def calibrate(
    anchors: AnchorSet,
    judge_labels: Mapping[str, bool | None],
    judge_id: str,
    judge_model: str | None = None,
    judge_prompt_hash: str | None = None,
    thresholds: CalibrationThresholds | None = None,
    epsilon: float = 0.2,
    q: float = 0.05,
    seed: int = 2026,
    extra_notes: list[str] | None = None,
) -> tuple[JudgeCalibration, ExamResult, AltTestOutcome | None]:
    notes: list[str] = list(anchors.warnings) + list(extra_notes or [])

    missing = [i.item_id for i in anchors.items if i.item_id not in judge_labels]
    if missing:
        notes.append(
            f"judge supplied no answer for {len(missing)} anchor item(s) — treated as "
            "abstentions, never guessed"
        )

    kept, human, ties = anchors.consensus_labels()
    if ties:
        notes.append(
            f"{len(ties)} item(s) excluded from the κ exam: the human panel tied — a tied "
            "panel is not ground truth"
        )
    if not kept:
        raise ValueError("every anchor item was a tied panel — no ground truth to exam against")

    judge_on_kept = [judge_labels.get(item.item_id) for item in kept]
    exam_result = exam(human, judge_on_kept, seed=seed)

    alt: AltTestOutcome | None = None
    if len(anchors.annotators) >= 3:
        alt = alt_test(
            anchors.annotator_matrix(),
            [judge_labels.get(item.item_id) for item in anchors.items],
            epsilon=epsilon,
            q=q,
        )
        notes.extend(alt.notes)

    calibration = issue_license(
        exam_result,
        judge_id=judge_id,
        alt=alt,
        judge_model=judge_model,
        judge_prompt_hash=judge_prompt_hash,
        anchor_set_ref=anchors.path,
        n_human_annotators=len(anchors.annotators),
        thresholds=thresholds,
        extra_notes=notes,
    )
    return calibration, exam_result, alt
