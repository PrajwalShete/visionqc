"""Pure, unit-testable decision logic.

Given an anomaly score (0..1) and the active recipe's parameters, produce a
PASS / REJECT / FAULT disposition with a machine-readable reason code. A missing
inference result (worker timeout or failure) always yields FAULT — the fail-safe
default, never a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Disposition(str, Enum):
    """Terminal quality disposition for a product."""

    PASS = "PASS"
    REJECT = "REJECT"
    FAULT = "FAULT"


class ReasonCode(str, Enum):
    """Machine-readable justification attached to a decision."""

    WITHIN_TOLERANCE = "within_tolerance"
    ANOMALY_ABOVE_THRESHOLD = "anomaly_above_threshold"
    MARGINAL_PASS = "marginal_pass"
    MARGINAL_REJECT = "marginal_reject"
    INFERENCE_UNAVAILABLE = "inference_unavailable"
    INVALID_SCORE = "invalid_score"


@dataclass(frozen=True)
class Region:
    """An anomalous sub-region reported by the model (optional detail)."""

    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass(frozen=True)
class InferenceResult:
    """Output of the inference worker for a single frame."""

    score: float
    model_version: str = "unknown"
    latency_ms: float = 0.0
    regions: list[Region] = field(default_factory=list)


@dataclass(frozen=True)
class RecipeParams:
    """The subset of recipe parameters the decision engine consumes."""

    anomaly_threshold: float
    #: Half-width of the "marginal" band around the threshold, in score units.
    #: A score within this band is flagged marginal (still decided, not FAULT).
    confidence_margin: float = 0.0


@dataclass(frozen=True)
class Decision:
    """The engine's verdict."""

    disposition: Disposition
    reason: ReasonCode
    score: float | None
    marginal: bool = False


def decide(result: InferenceResult | None, recipe: RecipeParams) -> Decision:
    """Return a :class:`Decision` for ``result`` under ``recipe``.

    * ``result is None`` → FAULT (``inference_unavailable``).
    * score outside ``[0, 1]`` or NaN → FAULT (``invalid_score``).
    * score >= threshold → REJECT; below → PASS.
    * scores within ``confidence_margin`` of the threshold are flagged
      ``marginal`` with a ``marginal_pass`` / ``marginal_reject`` reason.
    """

    if result is None:
        return Decision(Disposition.FAULT, ReasonCode.INFERENCE_UNAVAILABLE, None)

    score = result.score
    if score != score or score < 0.0 or score > 1.0:  # NaN or out of range
        return Decision(Disposition.FAULT, ReasonCode.INVALID_SCORE, None)

    threshold = recipe.anomaly_threshold
    margin = recipe.confidence_margin
    marginal = margin > 0.0 and abs(score - threshold) <= margin

    if score >= threshold:
        reason = ReasonCode.MARGINAL_REJECT if marginal else ReasonCode.ANOMALY_ABOVE_THRESHOLD
        return Decision(Disposition.REJECT, reason, score, marginal)

    reason = ReasonCode.MARGINAL_PASS if marginal else ReasonCode.WITHIN_TOLERANCE
    return Decision(Disposition.PASS, reason, score, marginal)


__all__ = [
    "Decision",
    "Disposition",
    "InferenceResult",
    "ReasonCode",
    "RecipeParams",
    "Region",
    "decide",
]
