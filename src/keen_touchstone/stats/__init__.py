"""The KeenTouchstone stats core: pass^k / pass@k estimation, CIs, decomposition."""

from .decomposition import (
    VarianceDecomposition,
    power_status,
    variance_decomposition,
)
from .estimators import (
    DecayPoint,
    TaskTrials,
    aggregate_pass_at_k,
    aggregate_pass_hat_k,
    decay_curve,
    max_common_k,
    pass_at_k,
    pass_hat_k,
)
from .intervals import (
    SMALL_SAMPLE_TASKS,
    CurveCI,
    Interval,
    beta_binomial_ci,
    bootstrap_ci_curve,
    wilson_ci,
)

__all__ = [
    "SMALL_SAMPLE_TASKS",
    "CurveCI",
    "DecayPoint",
    "Interval",
    "TaskTrials",
    "VarianceDecomposition",
    "aggregate_pass_at_k",
    "aggregate_pass_hat_k",
    "beta_binomial_ci",
    "bootstrap_ci_curve",
    "decay_curve",
    "max_common_k",
    "pass_at_k",
    "pass_hat_k",
    "power_status",
    "variance_decomposition",
    "wilson_ci",
]
