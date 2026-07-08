"""Failure-rate decomposition: how much of the failure is the model's, how
much is your harness's — MEASURED on your own tasks, never inferred.

The experiment (SPEC §6, "counterfactual adjudication"): run the same task
suite under up to four configurations —

    baseline      (model₁, harness₁)      the thing that's failing
    model_swap    (model₂, harness₁)      only the model changed
    harness_swap  (model₁, harness₂)      only the harness changed
    both_swap     (model₂, harness₂)      optional fourth cell

Per shared task, failure rate f = 1 − c/n. Shares are PAIRED per-task deltas:

    model_share_i   = f_baseline,i − f_model_swap,i
    harness_share_i = f_baseline,i − f_harness_swap,i

With the fourth cell the 2×2 factorial identity holds EXACTLY:

    f_base = model_share + harness_share + interaction + f_both
    interaction_i = f_mswap,i + f_hswap,i − f_base,i − f_both,i

A POSITIVE interaction means some failures need BOTH fixes (either alone
doesn't save them); a NEGATIVE interaction means the two shares double-count
failures that either fix alone would cure. A negative share means the swap
made things WORSE. All three are findings, not errors, and say so. Without the fourth cell,
interaction is honestly "not estimable" and the shares may double-count
overlapping failures — said in the output, never papered over.

Statistics reuse the house rules: CIs bootstrap over TASKS (never pooled
attempts); significance is the Phase 4 sign-flip permutation test on the
paired deltas. The literature's "harness fixes buy 15–50%" figure is a
single-team, unreplicated number — this module never cites it as a
conclusion; it measures YOUR data instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from keen_touchstone.online.compare import MIN_SHARED_FOR_POWER, _sign_flip_p_value

CellTasks = dict[str, tuple[int, int]]  # task_key -> (n, c)


@dataclass(frozen=True)
class Cell:
    name: str  # baseline | model_swap | harness_swap | both_swap
    tasks: CellTasks


@dataclass(frozen=True)
class Share:
    """Failure mass (in probability points) removed by one swap, with the
    uncertainty that a headline number must carry."""

    name: str
    delta: float  # mean per-task failure reduction (negative = swap made it worse)
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    method: str  # exact_sign_flip | sampled_sign_flip


@dataclass(frozen=True)
class AttributionResult:
    n_shared_tasks: int
    dropped_per_cell: dict[str, int]
    baseline_failure: float
    baseline_failure_ci: tuple[float, float]
    model_share: Share
    harness_share: Share
    interaction: Share | None  # None without the fourth cell
    irreducible_failure: float | None  # f(both_swap); None without the fourth cell
    has_both_cell: bool
    underpowered: bool
    notes: list[str] = field(default_factory=list)

    def sentence(self) -> str:
        """The sentence no other tool can produce."""
        def pp(share: Share) -> str:
            sig = "significant" if share.significant else "not significant — could be noise"
            if self.underpowered:
                # r5-F6: a bootstrap over ≤4 points prints false precision
                # (a width-zero bracket beside p=1.0) — annotate, don't fake
                return f"{share.delta * 100:+.1f}pp [too few tasks for a CI] ({sig})"
            return (
                f"{share.delta * 100:+.1f}pp [{share.ci_low * 100:+.1f}, "
                f"{share.ci_high * 100:+.1f}] ({sig})"
            )

        parts = [
            f"baseline failure rate {self.baseline_failure:.1%}",
            f"swapping the model removes {pp(self.model_share)}",
            f"fixing the harness removes {pp(self.harness_share)}",
        ]
        if self.interaction is not None:
            gloss = ""
            if self.interaction.delta > 1e-9:
                gloss = " (some failures need BOTH fixes)"
            elif self.interaction.delta < -1e-9:
                gloss = " (the shares double-count failures either fix alone would cure)"
            parts.append(f"interaction {self.interaction.delta * 100:+.1f}pp{gloss}")
            parts.append(f"irreducible under both swaps: {self.irreducible_failure:.1%}")
        else:
            parts.append(
                "interaction not estimable without a both-swap cell (shares may double-count "
                "overlapping failures)"
            )
        return "; ".join(parts)


def _failure_vector(cell: Cell, keys: list[str]) -> np.ndarray:
    return np.array([1.0 - cell.tasks[key][1] / cell.tasks[key][0] for key in keys])


def _share(
    name: str, deltas: np.ndarray, seed: int, n_resamples: int, alpha: float
) -> Share:
    p_value, method = _sign_flip_p_value(deltas, seed=seed, n_resamples=n_resamples)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(2000, len(deltas)))
    lo, hi = (float(x) for x in np.percentile(deltas[idx].mean(axis=1), [2.5, 97.5]))
    return Share(
        name=name,
        delta=float(deltas.mean()),
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        significant=p_value < alpha,
        method=method,
    )


def decompose(
    baseline: CellTasks,
    model_swap: CellTasks,
    harness_swap: CellTasks,
    both_swap: CellTasks | None = None,
    alpha: float = 0.05,
    seed: int = 2026,
    n_resamples: int = 10000,
) -> AttributionResult:
    cells = {"baseline": baseline, "model_swap": model_swap, "harness_swap": harness_swap}
    if both_swap is not None:
        cells["both_swap"] = both_swap

    shared = sorted(set.intersection(*(set(c) for c in cells.values())))
    if not shared:
        raise ValueError(
            "no tasks shared across all cells — attribution needs the SAME task suite in every "
            "configuration (check task signatures)"
        )
    dropped = {name: len(set(tasks) - set(shared)) for name, tasks in cells.items()}
    notes: list[str] = []
    if any(dropped.values()):
        notes.append(
            f"tasks not present in every cell were dropped: { {k: v for k, v in dropped.items() if v} }"
        )

    f = {name: _failure_vector(Cell(name, tasks), shared) for name, tasks in cells.items()}

    # r5-F7: unequal trial counts across cells keep the paired math unbiased
    # but make per-pair precision heterogeneous — say so instead of implying
    # equal-precision shares
    unequal = [
        key for key in shared
        if len({cells[name][key][0] for name in cells}) > 1
    ]
    if unequal:
        example = unequal[0]
        counts = {name: cells[name][example][0] for name in cells}
        notes.append(
            f"{len(unequal)} task(s) have UNEQUAL trial counts across cells (e.g. "
            f"{example}: {counts}) — shares remain unbiased but their precision differs "
            "per cell; the CIs reflect the realized variance"
        )

    model_deltas = f["baseline"] - f["model_swap"]
    harness_deltas = f["baseline"] - f["harness_swap"]
    model_share = _share("model", model_deltas, seed, n_resamples, alpha)
    harness_share = _share("harness", harness_deltas, seed + 1, n_resamples, alpha)

    interaction: Share | None = None
    irreducible: float | None = None
    if both_swap is not None:
        inter_deltas = f["model_swap"] + f["harness_swap"] - f["baseline"] - f["both_swap"]
        interaction = _share("interaction", inter_deltas, seed + 2, n_resamples, alpha)
        irreducible = float(f["both_swap"].mean())
        # the exact 2x2 identity — asserted, because arithmetic drift here
        # would mean the tool is lying about its own decomposition
        lhs = float(f["baseline"].mean())
        rhs = model_share.delta + harness_share.delta + interaction.delta + irreducible
        if abs(lhs - rhs) > 1e-9:  # pragma: no cover — identity violation is a bug
            raise AssertionError(f"decomposition identity broken: {lhs} != {rhs}")

    if model_share.delta < -1e-9:
        notes.append(
            "the model swap made things WORSE (negative share) — that is a finding, not an error"
        )
    if harness_share.delta < -1e-9:
        notes.append(
            "the harness swap made things WORSE (negative share) — that is a finding, not an error"
        )

    rng = np.random.default_rng(seed + 3)
    base = f["baseline"]
    idx = rng.integers(0, len(base), size=(2000, len(base)))
    base_ci = tuple(float(x) for x in np.percentile(base[idx].mean(axis=1), [2.5, 97.5]))

    underpowered = len(shared) < MIN_SHARED_FOR_POWER
    if underpowered:
        notes.append(
            f"only {len(shared)} shared task(s) (< {MIN_SHARED_FOR_POWER}): shares are "
            "direction-only evidence at this size"
        )
    return AttributionResult(
        n_shared_tasks=len(shared),
        dropped_per_cell=dropped,
        baseline_failure=float(base.mean()),
        baseline_failure_ci=base_ci,  # type: ignore[arg-type]
        model_share=model_share,
        harness_share=harness_share,
        interaction=interaction,
        irreducible_failure=irreducible,
        has_both_cell=both_swap is not None,
        underpowered=underpowered,
        notes=notes,
    )
