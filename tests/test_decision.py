"""Decision engine — pure logic cases."""

from __future__ import annotations

import math

from visionqc.decision.engine import (
    Disposition,
    InferenceResult,
    ReasonCode,
    RecipeParams,
    decide,
)


def _result(score: float) -> InferenceResult:
    return InferenceResult(score=score, model_version="test", latency_ms=1.0)


def test_pass_below_threshold() -> None:
    decision = decide(_result(0.1), RecipeParams(anomaly_threshold=0.5))
    assert decision.disposition is Disposition.PASS
    assert decision.reason is ReasonCode.WITHIN_TOLERANCE
    assert decision.marginal is False


def test_reject_at_or_above_threshold() -> None:
    decision = decide(_result(0.5), RecipeParams(anomaly_threshold=0.5))
    assert decision.disposition is Disposition.REJECT
    assert decision.reason is ReasonCode.ANOMALY_ABOVE_THRESHOLD


def test_reject_strictly_above() -> None:
    decision = decide(_result(0.9), RecipeParams(anomaly_threshold=0.5))
    assert decision.disposition is Disposition.REJECT


def test_marginal_pass_within_margin() -> None:
    decision = decide(_result(0.48), RecipeParams(anomaly_threshold=0.5, confidence_margin=0.05))
    assert decision.disposition is Disposition.PASS
    assert decision.reason is ReasonCode.MARGINAL_PASS
    assert decision.marginal is True


def test_marginal_reject_within_margin() -> None:
    decision = decide(_result(0.52), RecipeParams(anomaly_threshold=0.5, confidence_margin=0.05))
    assert decision.disposition is Disposition.REJECT
    assert decision.reason is ReasonCode.MARGINAL_REJECT
    assert decision.marginal is True


def test_none_result_is_fault() -> None:
    decision = decide(None, RecipeParams(anomaly_threshold=0.5))
    assert decision.disposition is Disposition.FAULT
    assert decision.reason is ReasonCode.INFERENCE_UNAVAILABLE
    assert decision.score is None


def test_out_of_range_score_is_fault() -> None:
    assert decide(_result(1.5), RecipeParams(anomaly_threshold=0.5)).disposition is (
        Disposition.FAULT
    )
    assert decide(_result(-0.1), RecipeParams(anomaly_threshold=0.5)).disposition is (
        Disposition.FAULT
    )


def test_nan_score_is_fault() -> None:
    decision = decide(_result(math.nan), RecipeParams(anomaly_threshold=0.5))
    assert decision.disposition is Disposition.FAULT
    assert decision.reason is ReasonCode.INVALID_SCORE
