"""Thin exam runner: put a real LLM judge in front of the anchor set.

Deliberately thin (Phase 0 anti-goal: "not another LLM-judge wrapper"). One
fixed binary prompt, sequential calls via Inspect's model interface (so
`mockllm/model` with custom_outputs is deterministic and keyless for demos,
and `anthropic/...` / `ollama/...` work unchanged). The prompt is versioned by
hash: change the prompt and you have a NEW judge that needs a NEW license.

The judge answers PASS / FAIL / UNKNOWN. Anything unparseable counts as an
abstention (never guessed into a verdict) and is reported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import anyio

from .anchors import AnchorSet

JUDGE_PROMPT = """You are grading whether an AI agent completed its task successfully.

Task given to the agent:
{input}

What the agent produced:
{output}

Did the agent genuinely succeed at the task? Consider correctness and completeness, not style.
Reply with exactly one word: PASS, FAIL, or UNKNOWN (only if you truly cannot tell)."""

JUDGE_PROMPT_HASH = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class JudgeRun:
    labels: dict[str, bool | None]  # item_id -> verdict (None = abstained/unparseable)
    model: str
    prompt_hash: str
    n_unparseable: int
    warnings: list[str] = field(default_factory=list)


def parse_verdict(completion: str) -> tuple[bool | None, bool]:
    """Returns (label, parseable). First PASS/FAIL/UNKNOWN token wins."""
    for token in completion.upper().split():
        word = token.strip(".,:;!*\"'()[]")
        if word == "PASS":
            return True, True
        if word == "FAIL":
            return False, True
        if word == "UNKNOWN":
            return None, True
    return None, False


def run_judge(
    anchors: AnchorSet,
    model: str,
    model_args: dict[str, Any] | None = None,
) -> JudgeRun:
    """Ask the judge model to grade every anchor item, sequentially."""
    from inspect_ai.model import get_model

    judge_model = get_model(model, **(model_args or {}))

    async def _run() -> tuple[dict[str, bool | None], int]:
        labels: dict[str, bool | None] = {}
        unparseable = 0
        for item in anchors.items:  # sequential: deterministic with mockllm
            prompt = JUDGE_PROMPT.format(input=item.input, output=item.output)
            result = await judge_model.generate(prompt)
            label, parseable = parse_verdict(result.completion)
            if not parseable:
                unparseable += 1
            labels[item.item_id] = label
        return labels, unparseable

    labels, n_unparseable = anyio.run(_run)
    warnings = []
    if n_unparseable:
        warnings.append(
            f"{n_unparseable} judge reply(ies) were unparseable — counted as abstentions, "
            "never guessed into verdicts"
        )
    return JudgeRun(
        labels=labels,
        model=model,
        prompt_hash=JUDGE_PROMPT_HASH,
        n_unparseable=n_unparseable,
        warnings=warnings,
    )
