"""Phase 4: the online loop — continuous reliability over live traffic."""

from .compare import (
    ComparisonResult,
    GateResult,
    compare,
    load_aggregate_tasks,
    slo_gate,
)
from .watch import WatchReport, WindowResult, parse_slo, watch_stream

__all__ = [
    "ComparisonResult",
    "GateResult",
    "WatchReport",
    "WindowResult",
    "compare",
    "load_aggregate_tasks",
    "parse_slo",
    "slo_gate",
    "watch_stream",
]
