"""pass^k / pass@k estimators and the reliability decay curve.

The estimand vocabulary (τ-bench, arXiv 2406.12045):

- ``pass@k`` — P(at least one of k independent attempts succeeds). The
  capability question: "can it work?"
- ``pass^k`` (pass-hat-k) — P(all k independent attempts succeed). The
  reliability question: "does it work every time?" This is the headline.

Per task with ``n`` recorded trials and ``c`` successes, both use exact
draw-without-replacement estimators that are unbiased for the with-replacement
quantities when trials are i.i.d. Bernoulli(p):

- ``pass^k``: C(c, k) / C(n, k)          (unbiased for p^k)
- ``pass@k``: 1 - C(n-c, k) / C(n, k)    (Chen et al. 2021, unbiased for 1-(1-p)^k)

These are the *same formulas* Inspect AI's ``pass_k_{k}`` / ``pass_at_{k}``
reducers compute per sample — parity is guarded by tests so "one metric
definition" stays true.

Suite aggregation is the mean of per-task estimates (the τ-bench/Inspect
convention). The decay curve evaluates the aggregate at k = 1..k_max with
**k_max = min(n_i) by default**, so the task set is constant across k and the
curve never bends because of composition drift. Extending past min(n_i) is
opt-in and loudly recorded per point.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from statistics import fmean


@dataclass(frozen=True)
class TaskTrials:
    """Repeated trials of one task: the stats core's only input shape.

    ``tokens``/``costs`` are optional per-trial measurements used for the
    cost-as-co-equal-axis columns; they never influence the estimators.
    """

    task_key: str
    n: int
    c: int
    tokens: tuple[int, ...] | None = None
    costs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"task {self.task_key!r}: n must be >= 1, got {self.n}")
        if not 0 <= self.c <= self.n:
            raise ValueError(
                f"task {self.task_key!r}: c must be in [0, n={self.n}], got {self.c}"
            )
        for name, values in (("tokens", self.tokens), ("costs", self.costs)):
            if values is not None and len(values) != self.n:
                raise ValueError(
                    f"task {self.task_key!r}: {name} has {len(values)} entries, expected n={self.n}"
                )

    @property
    def pass_rate(self) -> float:
        return self.c / self.n


def _validate_nck(n: int, c: int, k: int) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n={n}], got {c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n={n}], got {k} (pass^k is undefined for k > n)")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(all k attempts succeed) = p^k: C(c,k)/C(n,k)."""
    _validate_nck(n, c, k)
    return comb(c, k) / comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(>=1 of k attempts succeeds): 1 - C(n-c,k)/C(n,k)."""
    _validate_nck(n, c, k)
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate_pass_hat_k(tasks: list[TaskTrials], k: int) -> float:
    """Suite-level pass^k: mean of per-task estimates. All tasks must have n >= k."""
    _require_tasks(tasks)
    return fmean(pass_hat_k(t.n, t.c, k) for t in tasks)


def aggregate_pass_at_k(tasks: list[TaskTrials], k: int) -> float:
    """Suite-level pass@k: mean of per-task estimates. All tasks must have n >= k."""
    _require_tasks(tasks)
    return fmean(pass_at_k(t.n, t.c, k) for t in tasks)


def _require_tasks(tasks: list[TaskTrials]) -> None:
    if not tasks:
        raise ValueError("no tasks supplied")


@dataclass(frozen=True)
class DecayPoint:
    """One point of the reliability decay curve (point estimates only;
    confidence intervals are attached by the intervals layer)."""

    k: int
    pass_hat_k: float
    pass_at_k: float
    n_tasks: int
    dropped_task_keys: tuple[str, ...] = ()


def max_common_k(tasks: list[TaskTrials]) -> int:
    """The largest k for which every task still has enough trials: min(n_i)."""
    _require_tasks(tasks)
    return min(t.n for t in tasks)


def decay_curve(
    tasks: list[TaskTrials],
    k_max: int | None = None,
    allow_task_drop: bool = False,
) -> list[DecayPoint]:
    """The k=1..k_max reliability decay curve over a task suite.

    Default k_max = min(n_i): the task set is identical at every k, so the
    curve reflects reliability decay only, never task-set composition drift.
    Passing k_max beyond min(n_i) requires ``allow_task_drop=True``; points
    past the common range then include only tasks with n_i >= k, and each
    point records exactly which tasks were dropped.
    """
    _require_tasks(tasks)
    common = max_common_k(tasks)
    if k_max is None:
        k_max = common
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if k_max > common and not allow_task_drop:
        raise ValueError(
            f"k_max={k_max} exceeds min(n_i)={common}; pass allow_task_drop=True to "
            "extend the curve by dropping under-sampled tasks (dropped tasks are "
            "recorded per point)"
        )
    hardest = max(t.n for t in tasks)
    if k_max > hardest:
        raise ValueError(f"k_max={k_max} exceeds max(n_i)={hardest}; no task has that many trials")

    points: list[DecayPoint] = []
    for k in range(1, k_max + 1):
        included = [t for t in tasks if t.n >= k]
        dropped = tuple(t.task_key for t in tasks if t.n < k)
        points.append(
            DecayPoint(
                k=k,
                pass_hat_k=aggregate_pass_hat_k(included, k),
                pass_at_k=aggregate_pass_at_k(included, k),
                n_tasks=len(included),
                dropped_task_keys=dropped,
            )
        )
    return points
