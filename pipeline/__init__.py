"""Phase 4 pipeline assembly primitives.

The package is intentionally model-agnostic.  Scripts provide model adapters;
the reusable layer here owns contracts, strict JSON parsing, prompts, strategy
runners, and Phase 4 metrics.
"""

from pipeline.contracts import (
    ALLOWED_ERROR_TYPES,
    Feedback,
    PipelineOutput,
    TextBox,
    ErrorFinding,
)
from pipeline.strategies import SinglePassStrategy, TwoStageStrategy

__all__ = [
    "ALLOWED_ERROR_TYPES",
    "ErrorFinding",
    "Feedback",
    "PipelineOutput",
    "SinglePassStrategy",
    "TextBox",
    "TwoStageStrategy",
]
