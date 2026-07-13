"""Alarm engine subscriber.

Consumes fail-safe-relevant events (inference failures, lifecycle FAULTs, line
state changes), persists an alarm row, and republishes an :class:`AlarmRaised`
event carrying the new alarm id so the dashboard banner can render it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from ..db.repository import Repository
from ..events.bus import EventBus, OverflowPolicy, Subscription
from ..events.schemas import (
    AlarmRaised,
    AlarmRaisedPayload,
    Event,
    EventType,
)

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Alarm severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class _AlarmSpec:
    code: str
    severity: Severity
    source: str
    message: str
    product_id: str | None = None


_SUBSCRIBED = [
    EventType.INFERENCE_FAILED,
    EventType.PRODUCT_FINALIZED,
    EventType.LINE_STATE_CHANGED,
]


def _spec_for(event: Event) -> _AlarmSpec | None:
    """Map an event to an alarm spec, or ``None`` if it warrants no alarm."""

    payload = event.payload.model_dump(mode="json")
    if event.type is EventType.INFERENCE_FAILED:
        return _AlarmSpec(
            code="inference_failed",
            severity=Severity.CRITICAL,
            source="inference",
            message=payload.get("reason") or "inference worker unavailable",
            product_id=payload.get("product_id"),
        )
    if event.type is EventType.PRODUCT_FINALIZED and payload.get("outcome") == "FAULT":
        return _AlarmSpec(
            code="lifecycle_fault",
            severity=Severity.CRITICAL,
            source="lifecycle",
            message=payload.get("reason") or "product finalized as FAULT",
            product_id=payload.get("product_id"),
        )
    if event.type is EventType.LINE_STATE_CHANGED:
        state = payload.get("state")
        if state in {"DEGRADED", "STOPPED"}:
            return _AlarmSpec(
                code=f"line_{state.lower()}",
                severity=Severity.WARNING,
                source="line",
                message=payload.get("reason") or f"line state: {state}",
            )
    return None


class AlarmEngine:
    """Bus subscriber that turns fault events into persisted alarms."""

    def __init__(self, bus: EventBus, repo: Repository) -> None:
        self._bus = bus
        self._repo = repo
        self._sub: Subscription | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Subscribe (critical: BLOCK, large queue) and start consuming."""

        self._sub = self._bus.subscribe(
            "alarm-engine",
            event_types=_SUBSCRIBED,
            maxsize=1024,
            overflow=OverflowPolicy.BLOCK,
        )
        self._task = asyncio.create_task(self._run(), name="alarm-engine")

    async def stop(self) -> None:
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run(self) -> None:
        assert self._sub is not None
        async for event in self._sub:
            try:
                await self._handle(event)
            except Exception:
                logger.exception("alarm engine failed on %s", event.type)

    async def _handle(self, event: Event) -> None:
        spec = _spec_for(event)
        if spec is None:
            return
        alarm_id = await self._repo.insert_alarm(
            code=spec.code,
            severity=spec.severity.value,
            source=spec.source,
            message=spec.message,
            product_id=spec.product_id,
        )
        await self._bus.publish(
            AlarmRaised(
                payload=AlarmRaisedPayload(
                    alarm_id=alarm_id,
                    code=spec.code,
                    severity=spec.severity.value,
                    source=spec.source,
                    message=spec.message,
                    product_id=spec.product_id,
                )
            )
        )


__all__ = ["AlarmEngine", "Severity"]
