"""Contract layer: EvalVerdict conditional rule + JudgeCalibration schema round-trip."""

import jsonschema
import pytest
from pydantic import ValidationError

from keen_touchstone.artifacts import (
    AltTestResult,
    CalibrationThresholds,
    EvalVerdict,
    JudgeCalibration,
    load_schema,
)


def _verdict(**over):
    base = dict(
        verdict_id="v1", trace_id="t1", scorer_id="s", scorer_version="1",
        scorer_kind="programmatic", tier="T0_deterministic", value=True,
    )
    base.update(over)
    return EvalVerdict(**base)


def test_programmatic_verdict_needs_no_license() -> None:
    v = _verdict()
    jsonschema.validate(v.to_schema_dict(), load_schema("eval-verdict"))


def test_model_graded_without_license_rejected_by_pydantic_and_schema() -> None:
    with pytest.raises(ValidationError, match="judge_calibration_ref"):
        _verdict(scorer_kind="model_graded", tier="T2_ungrounded", judge_model="m")
    # belt and braces: the raw schema enforces the same conditional
    raw = _verdict().to_schema_dict()
    raw["scorer_kind"] = "model_graded"
    raw["judge_calibration_ref"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, load_schema("eval-verdict"))


def test_model_graded_with_license_valid() -> None:
    v = _verdict(
        scorer_kind="model_graded", tier="T2_ungrounded",
        judge_model="m", judge_calibration_ref="cal-abc123",
    )
    data = v.to_schema_dict()
    assert data["judge_calibration_ref"] == "cal-abc123"


def _calibration(**over) -> JudgeCalibration:
    base = dict(
        calibration_id="cal-1", judge_id="my-judge", anchor_n_items=40,
        n_human_annotators=1, thresholds=CalibrationThresholds(),
        status="NEEDS_HUMAN", reasons=["exam not run yet"],
        created_at="2026-07-03T00:00:00Z", keen_touchstone_version="0.1.0.dev0",
    )
    base.update(over)
    return JudgeCalibration(**base)


def test_judge_calibration_schema_roundtrip() -> None:
    cal = _calibration(
        kappa=0.72, kappa_ci_low=0.55, kappa_ci_high=0.85, raw_agreement=0.88,
        tpr=0.9, fpr=0.15, prevalence=0.6, abstention_rate=0.05,
        alt_test=AltTestResult(applicable=False, reason="needs >= 3 human annotators"),
        status="JUDGE_LICENSED", reasons=[],
    )
    data = cal.to_schema_dict()
    jsonschema.validate(data, load_schema("judge-calibration"))
    assert data["human_label_source"] == "human"


def test_anti_circularity_is_structural() -> None:
    # the model refuses any non-human label source at construction time
    with pytest.raises(ValidationError):
        _calibration(human_label_source="llm_generated")
    # and the schema refuses it at data level
    data = _calibration().to_schema_dict()
    data["human_label_source"] = "llm_generated"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, load_schema("judge-calibration"))


def test_schema_rejects_extra_fields() -> None:
    data = _calibration().to_schema_dict()
    data["sneaky"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, load_schema("judge-calibration"))
