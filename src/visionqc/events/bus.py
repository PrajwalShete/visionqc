"""In-process async event bus.

A hand-rolled ``asyncio.Queue``-per-subscriber bus. The publisher stamps
``event_id`` (already defaulted), wall-clock and monotonic timestamps, then fans
the event out to every matching subscriber's bounded queue.

Overflow is governed per subscription:

* :attr:`OverflowPolicy.BLOCK` — the publisher awaits queue space. Used for
  critical, must-not-drop subscribers (the DB writer) together with a large
  ``maxsize`` so blocking is effectively never observed.
* :attr:`OverflowPolicy.DROP_OLDEST` — on a full queue the oldest event is
  discarded to make room. Used for UI-ish subscribers (WebSocket hub, stats)
  where a slow consumer must never stall the publisher.

Consumption happens in each subscriber's own task, so a subscriber raising an
exception can never propagate into the publisher.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from enum import Enum

from .schemas import Event, EventType

logger = logging.getLogger(__name__)

_SHUTDOWN = object()


class OverflowPolicy(str, Enum):
    """How a subscription behaves when its queue is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"


class Subscription:
    """A single subscriber's view onto the bus.

    Async-iterable: ``async for event in subscription`` yields events until the
    subscription (or the bus) is closed.
    """

    def __init__(
        self,
        name: str,
        event_types: Iterable[EventType] | None,
        maxsize: int,
        overflow: OverflowPolicy,
    ) -> None:
        self.name = name
        self.event_types: frozenset[EventType] | None = (
            frozenset(event_types) if event_types is not None else None
        )
        self.overflow = overflow
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self.dropped = 0

    def accepts(self, event_type: EventType) -> bool:
        """Return whether this subscription wants ``event_type``."""

        return self.event_types is None or event_type in self.event_types

    async def _deliver(self, event: Event) -> None:
        """Enqueue ``event`` according to the overflow policy."""

        if self._closed:
            return
        if self.overflow is OverflowPolicy.BLOCK:
            await self._queue.put(event)
            return
        # DROP_OLDEST
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - race safety
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - race safety
                self.dropped += 1

    async def get(self) -> Event | None:
        """Await the next event, or ``None`` once closed and drained."""

        item = await self._queue.get()
        if item is _SHUTDOWN:
            return None
        return item  # type: ignore[return-value]

    def close(self) -> None:
        """Mark closed and wake any pending consumer with a sentinel."""

        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except asyncio.QueueFull:
            # Drop-oldest to guarantee the sentinel lands.
            with contextlib.suppress(asyncio.QueueEmpty):  # pragma: no cover
                self._queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):  # pragma: no cover
                self._queue.put_nowait(_SHUTDOWN)

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterator()

    async def _iterator(self) -> AsyncIterator[Event]:
        while True:
            event = await self.get()
            if event is None:
                return
            yield event


class EventBus:
    """Fan-out event bus with per-subscriber bounded queues."""

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._closed = False

    def subscribe(
        self,
        name: str,
        event_types: Iterable[EventType] | None = None,
        maxsize: int = 256,
        overflow: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> Subscription:
        """Register a subscriber and return its :class:`Subscription`.

        :param name: Human-readable subscriber name (for logging/metrics).
        :param event_types: Restrict to these types, or ``None`` for all.
        :param maxsize: Bounded queue size.
        :param overflow: Overflow policy for a full queue.
        """

        sub = Subscription(name, event_types, maxsize, overflow)
        self._subscriptions.append(sub)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove and close a subscription."""

        subscription.close()
        with contextlib.suppress(ValueError):  # pragma: no cover - idempotent
            self._subscriptions.remove(subscription)

    async def publish(self, event: Event) -> None:
        """Stamp timestamps on ``event`` and fan out to all subscribers.

        Exceptions from any single subscriber's delivery are logged and never
        propagated to the publisher or to other subscribers.
        """

        if self._closed:
            return
        event.ts_wall = datetime.now(UTC)
        event.ts_mono = time.monotonic()
        for sub in list(self._subscriptions):
            if not sub.accepts(event.type):
                continue
            try:
                await sub._deliver(event)
            except Exception:
                logger.exception("event delivery to subscriber %s failed", sub.name)

    async def close(self) -> None:
        """Close the bus and all subscriptions for clean shutdown."""

        self._closed = True
        for sub in list(self._subscriptions):
            sub.close()
        self._subscriptions.clear()


__all__ = ["EventBus", "OverflowPolicy", "Subscription"]
