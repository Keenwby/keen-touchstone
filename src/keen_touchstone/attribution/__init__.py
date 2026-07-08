"""Phase 5: model-vs-harness attribution — measured, never inferred."""

from .decompose import AttributionResult, Cell, Share, decompose
from .diagnose import DIAGNOSE_DISCLAIMER, DiagnosisReport, diagnose_cassette

__all__ = [
    "DIAGNOSE_DISCLAIMER",
    "AttributionResult",
    "Cell",
    "DiagnosisReport",
    "Share",
    "decompose",
    "diagnose_cassette",
]
