"""Pydantic models for the KeenTouchstone data-model artifacts (Phase 1 subset).

v0.1 implements Artifact D (`ReliabilityAggregate`) end-to-end. Every emitted
dict is runtime-validated against the canonical packaged JSON Schema — the
schema is the contract, the pydantic model is merely its typed convenience.

Note: the schemas set ``additionalProperties: false`` and ``task_key_source``
is optional-but-not-nullable, so serialization omits it when unset (nullable
fields serialize explicit nulls).
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, model_validator

SUITE_TASK_KEY = "__suite__"
"""Reserved task_key for the suite-level rollup (documented in CHANGELOG)."""


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a packaged canonical schema, e.g. ``load_schema("reliability-aggregate")``."""
    text = (files("keen_touchstone") / "schemas" / f"{name}.schema.json").read_text()
    return json.loads(text)


class DecayCurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(ge=1)
    pass_hat_k: float = Field(ge=0, le=1)
    ci_low: float | None = None
    ci_high: float | None = None
    pass_at_k: float | None = Field(default=None, ge=0, le=1)
    """The capability twin at the same k — kept beside pass^k so the
    'pass@k climbs while pass^k collapses' contrast is one artifact."""


class Attribution(BaseModel):
    """SPEC §6 — carried for schema completeness; populated from Phase 5."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_share: float | None = None
    harness_share: float | None = None
    method: Literal["measured_ab", "inferred_layer_hypothesis"]
    confidence_band: str | None = None
    etclovg_layer: str | None = None


class ScoreHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    author: str | None = None
    reason: str | None = None
    value: bool | float | str | None = None
    timestamp: str | None = None


class EvalVerdict(BaseModel):
    """Artifact C: a grader's judgment on a run, with provenance and the
    anti-circularity calibration link. The structural rule this class exists
    to enforce: a model-graded verdict without a judge license is invalid."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    verdict_id: str
    trace_id: str
    sample_id: str | None = None
    scorer_id: str
    scorer_version: str
    scorer_kind: Literal["programmatic", "model_graded", "trajectory"]
    tier: Literal["T0_deterministic", "T1_reference", "T2_ungrounded"]
    value: bool | float | str
    explanation: str | None = None
    judge_model: str | None = None
    judge_calibration_ref: str | None = None
    trajectory_invariant: bool | None = None
    score_history: list[ScoreHistoryEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _model_graded_requires_calibration(self) -> EvalVerdict:
        if self.scorer_kind == "model_graded" and not self.judge_calibration_ref:
            raise ValueError(
                "a model_graded verdict must carry judge_calibration_ref — an uncalibrated "
                "judge's score is not evidence (SPEC §3, anti-circularity)"
            )
        # Laundering guard (adversarial-review round 2): a verdict carrying
        # LLM tells cannot self-declare as deterministic to dodge the license.
        if self.scorer_kind != "model_graded" and (
            self.judge_model or self.judge_calibration_ref
        ):
            raise ValueError(
                f"scorer_kind={self.scorer_kind!r} but the verdict carries "
                f"judge_model/judge_calibration_ref — a model-graded verdict cannot be "
                "laundered as deterministic; set scorer_kind=model_graded (and present a "
                "license) or drop the judge fields"
            )
        return self

    def to_schema_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        jsonschema.validate(
            data, load_schema("eval-verdict"), cls=jsonschema.Draft202012Validator
        )
        return data


class AltTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicable: bool
    reason: str | None = None
    epsilon: float | None = None
    q: float | None = None  # FDR level of the BY correction
    omega: float | None = Field(default=None, ge=0, le=1)
    passed: bool | None = None
    avg_advantage_probability: float | None = Field(default=None, ge=0, le=1)
    n_annotators: int | None = None


class CalibrationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kappa_licensed: float = Field(default=0.6, ge=0, le=1)  # Landis-Koch "acceptable"; 0.8 strong
    min_items: int = Field(default=30, ge=1)  # below this, no kappa point estimate at all
    max_abstention: float = Field(default=0.2, ge=0, le=1)
    gate_on: Literal["point", "ci_low"] = "point"  # strict mode gates on the CI lower bound


class JudgeCalibration(BaseModel):
    """SPEC §3b concretized: the judge's exam record and license."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    calibration_id: str
    judge_id: str
    judge_model: str | None = None
    judge_prompt_hash: str | None = None
    anchor_set_ref: str | None = None
    anchor_n_items: int = Field(ge=1)
    n_human_annotators: int = Field(ge=1)
    human_label_source: Literal["human"] = "human"
    prevalence: float | None = Field(default=None, ge=0, le=1)

    kappa: float | None = Field(default=None, ge=-1, le=1)
    kappa_ci_low: float | None = Field(default=None, ge=-1, le=1)
    kappa_ci_high: float | None = Field(default=None, ge=-1, le=1)
    raw_agreement: float | None = Field(default=None, ge=0, le=1)
    tpr: float | None = Field(default=None, ge=0, le=1)
    tpr_ci_low: float | None = Field(default=None, ge=0, le=1)
    tpr_ci_high: float | None = Field(default=None, ge=0, le=1)
    fpr: float | None = Field(default=None, ge=0, le=1)
    fpr_ci_low: float | None = Field(default=None, ge=0, le=1)
    fpr_ci_high: float | None = Field(default=None, ge=0, le=1)
    abstention_rate: float | None = Field(default=None, ge=0, le=1)

    alt_test: AltTestResult | None = None
    thresholds: CalibrationThresholds
    status: Literal["JUDGE_LICENSED", "NEEDS_HUMAN"]
    reasons: list[str] = Field(default_factory=list)
    created_at: str
    keen_touchstone_version: str

    def to_schema_dict(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        jsonschema.validate(
            data, load_schema("judge-calibration"), cls=jsonschema.Draft202012Validator
        )
        return data


class ReliabilityAggregate(BaseModel):
    """Artifact D: N rollouts of one task (or the suite rollup) as a distribution."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    task_key: str
    task_key_source: Literal["dataset_id", "task_signature", "declared_tag"] | None = None
    agent_config_hash: str
    model: str
    n_rollouts: int = Field(ge=1)
    context: Literal["offline", "online", "replay"]

    pass_rate: float = Field(ge=0, le=1)
    pass_at_k: float | None = Field(default=None, ge=0, le=1)
    pass_hat_k: float | None = Field(default=None, ge=0, le=1)
    headline_k: int | None = Field(default=None, ge=1)
    pass_hat_k_ci_low: float | None = Field(default=None, ge=0, le=1)
    pass_hat_k_ci_high: float | None = Field(default=None, ge=0, le=1)
    ci_method: (
        Literal[
            "bootstrap", "bootstrap_posterior_envelope", "clustered_se", "beta_binomial", "wilson"
        ]
        | None
    ) = None
    reliability_decay_curve: list[DecayCurvePoint] = Field(default_factory=list)
    variance: float | None = None
    skew: float | None = None

    cost_mean: float | None = None
    cost_p95: float | None = None
    token_mean: float | None = None

    power_status: Literal["SIGNIFICANT", "NOISE", "UNDERPOWERED_NEED_MORE_N"] | None = None
    attribution: Attribution | None = None

    def to_schema_dict(self) -> dict[str, Any]:
        """Serialize to a dict that satisfies the canonical JSON Schema (validated)."""
        data = self.model_dump(mode="json")
        if data.get("task_key_source") is None:
            data.pop("task_key_source", None)  # optional-but-not-nullable in the schema
        jsonschema.validate(
            data,
            load_schema("reliability-aggregate"),
            cls=jsonschema.Draft202012Validator,
        )
        return data
