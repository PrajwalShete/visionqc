"""Event schemas for the in-process event bus.

Every event is an envelope (:class:`Event`) carrying a stable ``event_id``, the
wall-clock and monotonic timestamps stamped by the bus at publish time, a
``type`` drawn from :class:`EventType`, and a typed ``payload`` model.

Concrete event classes narrow ``type`` and ``payload`` so that subscribers can
pattern-match on ``event.type`` and read strongly typed payload fields.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Enumeration of every event type published on the bus."""

    TRIGGER_FIRED = "TriggerFired"
    FRAME_CAPTURED = "FrameCaptured"
    INFERENCE_COMPLETED = "InferenceCompleted"
    INFERENCE_FAILED = "InferenceFailed"
    DECISION_MADE = "DecisionMade"
    REJECT_COMMANDED = "RejectCommanded"
    REJECT_CONFIRMED = "RejectConfirmed"
    PRODUCT_FINALIZED = "ProductFinalized"
    ALARM_RAISED = "AlarmRaised"
    ALARM_CLEARED = "AlarmCleared"
    LINE_STATE_CHANGED = "LineStateChanged"


# --------------------------------------------------------------------------- #
# Payload models
# --------------------------------------------------------------------------- #
class Payload(BaseModel):
    """Base class for all event payloads."""


class TriggerFiredPayload(Payload):
    product_id: str
    trigger_ts: float
    camera_id: str = "cam0"
    recipe_id: int | None = None


class FrameCapturedPayload(Payload):
    product_id: str
    camera_id: str = "cam0"
    capture_ms: float | None = None
    width: int | None = None
    height: int | None = None


class InferenceCompletedPayload(Payload):
    product_id: str
    score: float
    model_version: str
    latency_ms: float


class InferenceFailedPayload(Payload):
    product_id: str
    reason: str
    error: str | None = None


class DecisionMadePayload(Payload):
    product_id: str
    outcome: str
    reason: str
    score: float | None = None
    recipe_id: int | None = None


class RejectCommandedPayload(Payload):
    product_id: str
    station: str = "reject0"


class RejectConfirmedPayload(Payload):
    product_id: str
    station: str = "reject0"


class ProductFinalizedPayload(Payload):
    product_id: str
    outcome: str
    reason: str
    anomaly_score: float | None = None
    recipe_id: int | None = None
    model_version: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class AlarmRaisedPayload(Payload):
    alarm_id: int | None = None
    code: str
    severity: str
    source: str
    message: str
    product_id: str | None = None


class AlarmClearedPayload(Payload):
    alarm_id: int | None = None
    code: str


class LineStateChangedPayload(Payload):
    state: str
    reason: str | None = None


# --------------------------------------------------------------------------- #
# Event envelope + concrete events
# --------------------------------------------------------------------------- #
class Event(BaseModel):
    """Generic event envelope.

    ``ts_wall`` and ``ts_mono`` are populated by the bus at publish time; they
    are left ``None`` until then so publishers never have to stamp them.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    ts_wall: datetime | None = None
    ts_mono: float | None = None
    type: EventType
    payload: Payload = Field(default_factory=Payload)

    def to_wire(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for WebSocket broadcast."""

        return {
            "event_id": self.event_id,
            "ts_wall": self.ts_wall.isoformat() if self.ts_wall else None,
            "ts_mono": self.ts_mono,
            "type": self.type.value,
            "payload": self.payload.model_dump(mode="json"),
        }


class TriggerFired(Event):
    type: EventType = EventType.TRIGGER_FIRED
    payload: TriggerFiredPayload


class FrameCaptured(Event):
    type: EventType = EventType.FRAME_CAPTURED
    payload: FrameCapturedPayload


class InferenceCompleted(Event):
    type: EventType = EventType.INFERENCE_COMPLETED
    payload: InferenceCompletedPayload


class InferenceFailed(Event):
    type: EventType = EventType.INFERENCE_FAILED
    payload: InferenceFailedPayload


class DecisionMade(Event):
    type: EventType = EventType.DECISION_MADE
    payload: DecisionMadePayload


class RejectCommanded(Event):
    type: EventType = EventType.REJECT_COMMANDED
    payload: RejectCommandedPayload


class RejectConfirmed(Event):
    type: EventType = EventType.REJECT_CONFIRMED
    payload: RejectConfirmedPayload


class ProductFinalized(Event):
    type: EventType = EventType.PRODUCT_FINALIZED
    payload: ProductFinalizedPayload


class AlarmRaised(Event):
    type: EventType = EventType.ALARM_RAISED
    payload: AlarmRaisedPayload


class AlarmCleared(Event):
    type: EventType = EventType.ALARM_CLEARED
    payload: AlarmClearedPayload


class LineStateChanged(Event):
    type: EventType = EventType.LINE_STATE_CHANGED
    payload: LineStateChangedPayload


__all__ = [
    "AlarmCleared",
    "AlarmClearedPayload",
    "AlarmRaised",
    "AlarmRaisedPayload",
    "DecisionMade",
    "DecisionMadePayload",
    "Event",
    "EventType",
    "FrameCaptured",
    "FrameCapturedPayload",
    "InferenceCompleted",
    "InferenceCompletedPayload",
    "InferenceFailed",
    "InferenceFailedPayload",
    "LineStateChanged",
    "LineStateChangedPayload",
    "Payload",
    "ProductFinalized",
    "ProductFinalizedPayload",
    "RejectCommanded",
    "RejectCommandedPayload",
    "RejectConfirmed",
    "RejectConfirmedPayload",
    "TriggerFired",
    "TriggerFiredPayload",
]
