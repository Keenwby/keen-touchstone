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
from pydantic import BaseModel, ConfigDict, Field

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
    ci_method: Literal["bootstrap", "clustered_se", "beta_binomial", "wilson"] | None = None
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
