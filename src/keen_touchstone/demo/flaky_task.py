"""The bundled demo: a simulated flaky agent, runnable with zero API keys.

What this is (honestly): the model is Inspect's ``mockllm/model`` and the
scorer *simulates* an agent that succeeds with a per-task probability p —
deterministically, keyed on (seed, sample_id, epoch), so runs reproduce
exactly regardless of sample concurrency. It exists to exercise the real
pipeline (Inspect runner → .eval log → pass^k stats → report) and to show
what the decay curve looks like when pass@1 flatters an agent.

Swap in your own task + model for real numbers:  ``touchstone run <task.py>``.
"""

from __future__ import annotations

import random

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

DEMO_SEED = 2026

# per-task true success probability — descending, so the suite decay curve
# shows the "looks great at pass@1, undeployable at pass^8" story
FLAKY_TASK_P: dict[str, float] = {
    "checkout-refund": 0.95,
    "flight-rebooking": 0.85,
    "invoice-reconcile": 0.75,
    "db-migration": 0.65,
    "multi-tool-research": 0.55,
    "legacy-api-integration": 0.45,
}


@scorer(metrics=[accuracy(), stderr()])
def simulated_flaky_scorer(seed: int = DEMO_SEED):
    async def score(state: TaskState, target: Target) -> Score:
        p = FLAKY_TASK_P.get(str(state.sample_id), 0.5)
        rng = random.Random(f"{seed}:{state.sample_id}:{state.epoch}")
        succeeded = rng.random() < p
        return Score(
            value=CORRECT if succeeded else INCORRECT,
            explanation=f"simulated flaky agent: true p={p}, epoch={state.epoch}",
        )

    return score


@task
def touchstone_demo(seed: int = DEMO_SEED) -> Task:
    dataset = [
        Sample(
            input=f"Simulated agentic task: {name} (true per-trial success p={p})",
            target="done",
            id=name,
        )
        for name, p in FLAKY_TASK_P.items()
    ]
    return Task(dataset=dataset, solver=generate(), scorer=simulated_flaky_scorer(seed))
