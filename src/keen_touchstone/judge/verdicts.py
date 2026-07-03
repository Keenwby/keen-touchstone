"""Verdict files + the gate enforced in the DATA PATH.

This is where the two phases join: a licensed judge's verdicts become the
outcome source for unlabeled production traces, and pass^k flows from there.

The enforcement rules (the moat, made concrete):

1. A ``model_graded`` verdict is accepted ONLY when a license is presented,
   the verdict's ``judge_calibration_ref`` matches that license's
   ``calibration_id``, and the license status is ``JUDGE_LICENSED``.
   NEEDS_HUMAN → the whole ingest refuses: those scores are not evidence.
2. ``programmatic`` / ``trajectory`` verdicts (deterministic checks) need no
   license — deterministic graders don't drift and aren't on trial.
3. Verdict values coerce conservatively: booleans, "pass"/"fail" strings, or
   numbers vs a threshold. Anything else is rejected loudly, never guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from keen_touchstone.artifacts import EvalVerdict, JudgeCalibration, load_schema

from .license import check_license


def read_verdicts(path: str | Path) -> list[EvalVerdict]:
    """Read EvalVerdict JSONL — every line validated against the contract."""
    path = Path(path)
    verdicts: list[EvalVerdict] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({err.msg})") from err
            try:
                verdicts.append(EvalVerdict.model_validate(raw))
            except Exception as err:
                raise ValueError(f"{path}:{lineno}: invalid EvalVerdict — {err}") from err
    if not verdicts:
        raise ValueError(f"{path}: no verdicts found")
    return verdicts


def load_license(path: str | Path) -> JudgeCalibration:
    import jsonschema

    data = json.loads(Path(path).read_text())
    try:
        jsonschema.validate(data, load_schema("judge-calibration"))
    except jsonschema.ValidationError as err:
        raise ValueError(f"{path} is not a valid license: {err.message}") from err
    return JudgeCalibration.model_validate(data)


def _coerce_outcome(value: bool | float | str, threshold: float) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= threshold
    text = str(value).strip().lower()
    if text in ("pass", "passed", "success", "true", "c"):
        return True
    if text in ("fail", "failed", "failure", "false", "i"):
        return False
    raise ValueError(
        f"verdict value {value!r} is not coercible to pass/fail — refuse to guess"
    )


def outcomes_from_verdicts(
    verdicts: list[EvalVerdict],
    license_: JudgeCalibration | None,
    threshold: float = 1.0,
) -> dict[str, bool]:
    """trace_id → outcome, with the license gate enforced per verdict."""
    model_graded = [v for v in verdicts if v.scorer_kind == "model_graded"]
    if model_graded:
        if license_ is None:
            raise ValueError(
                f"{len(model_graded)} model-graded verdict(s) present but no --license given — "
                "an uncalibrated judge's scores are not evidence (run `touchstone judge "
                "calibrate` first)"
            )
        ok, message = check_license(license_)
        if not ok:
            raise ValueError(message)
        mismatched = [
            v.verdict_id
            for v in model_graded
            if v.judge_calibration_ref != license_.calibration_id
        ]
        if mismatched:
            raise ValueError(
                f"verdict(s) {mismatched[:5]} reference a different judge_calibration_ref than "
                f"the presented license ({license_.calibration_id}) — a license is not "
                "transferable between judges or prompts"
            )

    outcomes: dict[str, bool] = {}
    for v in verdicts:
        if v.trace_id in outcomes:
            raise ValueError(
                f"duplicate verdicts for trace_id {v.trace_id!r} — one verdict per run"
            )
        outcomes[v.trace_id] = _coerce_outcome(v.value, threshold)
    return outcomes
