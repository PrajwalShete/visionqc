"""DB-writer event-bus subscriber.

This is the critical, must-not-drop subscriber: it consumes lifecycle events off
the bus (with a BLOCK overflow policy and a large queue) and persists every
product state transition. Product-image evidence and alarm rows are written by
their own components; this subscriber owns the ``products`` and
``product_events`` tables.
"""

from __future__ import annotations

import asyncio
import logging

from ..events.bus import EventBus, OverflowPolicy, Subscription
from ..events.schemas import Event, EventType
from .repository import Repository

logger = logging.getLogger(__name__)

# Lifecycle events this subscriber persists, and the state they move a product
# to (``None`` means "no state change, record the event only").
_TO_STATE: dict[EventType, str | None] = {
    EventType.TRIGGER_FIRED: "TRIGGERED",
    EventType.FRAME_CAPTURED: "CAPTURED",
    EventType.INFERENCE_COMPLETED: "INFERRED",
    EventType.INFERENCE_FAILED: None,
    EventType.DECISION_MADE: "DECIDED",
    EventType.REJECT_COMMANDED: None,
    EventType.REJECT_CONFIRMED: None,
    EventType.PRODUCT_FINALIZED: None,  # resolved from the outcome payload
}


class DBWriterSubscriber:
    """Consumes lifecycle events and persists them through the repository."""

    def __init__(self, bus: EventBus, repo: Repository) -> None:
        self._bus = bus
        self._repo = repo
        self._sub: Subscription | None = None
        self._task: asyncio.Task[None] | None = None
        self._last_state: dict[str, str] = {}

    def start(self) -> None:
        """Subscribe (critical: large queue, BLOCK) and start consuming."""

        self._sub = self._bus.subscribe(
            "db-writer",
            event_types=list(_TO_STATE.keys()),
            maxsize=4096,
            overflow=OverflowPolicy.BLOCK,
        )
        self._task = asyncio.create_task(self._run(), name="db-writer-subscriber")

    async def stop(self) -> None:
        """Stop consuming and release the subscription."""

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
                await self._persist(event)
            except Exception:
                logger.exception("failed to persist event %s", event.type)

    async def _persist(self, event: Event) -> None:
        payload = event.payload.model_dump(mode="json")
        product_id = payload.get("product_id")
        if not product_id:
            return

        from_state = self._last_state.get(product_id)
        to_state = _TO_STATE.get(event.type)

        if event.type is EventType.TRIGGER_FIRED:
            await self._repo.insert_product(
                product_id=product_id,
                trigger_ts=float(payload.get("trigger_ts", 0.0)),
                state="TRIGGERED",
                recipe_id=payload.get("recipe_id"),
            )
        elif event.type is EventType.FRAME_CAPTURED:
            await self._repo.update_product(product_id, state="CAPTURED")
        elif event.type is EventType.INFERENCE_COMPLETED:
            await self._repo.update_product(
                product_id,
                state="INFERRED",
                anomaly_score=payload.get("score"),
                model_version=payload.get("model_version"),
            )
        elif event.type is EventType.DECISION_MADE:
            await self._repo.update_product(
                product_id,
                state="DECIDED",
                decision_reason=payload.get("reason"),
            )
        elif event.type is EventType.PRODUCT_FINALIZED:
            outcome = payload.get("outcome", "FAULT")
            to_state = outcome
            await self._repo.update_product(
                product_id,
                state=outcome,
                outcome=outcome,
                anomaly_score=payload.get("anomaly_score"),
                decision_reason=payload.get("reason"),
                model_version=payload.get("model_version"),
                timings=payload.get("timings") or None,
            )

        await self._repo.append_product_event(
            product_id,
            event_type=event.type.value,
            from_state=from_state,
            to_state=to_state,
            ts_wall=event.ts_wall.isoformat() if event.ts_wall else None,
            ts_mono=event.ts_mono,
            payload=payload,
        )
        if to_state is not None:
            self._last_state[product_id] = to_state


__all__ = ["DBWriterSubscriber"]
