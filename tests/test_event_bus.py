"""Event bus — fan-out, bounded overflow, subscriber isolation, filtering."""

from __future__ import annotations

import asyncio

import pytest

from visionqc.events.bus import EventBus, OverflowPolicy
from visionqc.events.schemas import (
    Event,
    EventType,
    TriggerFired,
    TriggerFiredPayload,
)


def _trigger(pid: str) -> TriggerFired:
    return TriggerFired(payload=TriggerFiredPayload(product_id=pid, trigger_ts=0.0))


async def test_publish_stamps_timestamps() -> None:
    bus = EventBus()
    sub = bus.subscribe("s")
    await bus.publish(_trigger("p1"))
    event = await sub.get()
    assert event is not None
    assert event.ts_wall is not None
    assert event.ts_mono is not None
    assert event.type is EventType.TRIGGER_FIRED


async def test_fanout_to_all_subscribers() -> None:
    bus = EventBus()
    a = bus.subscribe("a")
    b = bus.subscribe("b")
    await bus.publish(_trigger("p1"))
    ea = await a.get()
    eb = await b.get()
    assert ea is not None and eb is not None
    assert ea.payload.product_id == "p1"  # type: ignore[attr-defined]
    assert eb.payload.product_id == "p1"  # type: ignore[attr-defined]


async def test_event_type_filtering() -> None:
    bus = EventBus()
    sub = bus.subscribe("filtered", event_types=[EventType.FRAME_CAPTURED])
    await bus.publish(_trigger("p1"))  # not FRAME_CAPTURED -> filtered out
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sub.get(), timeout=0.05)


async def test_drop_oldest_overflow() -> None:
    bus = EventBus()
    sub = bus.subscribe("slow", maxsize=2, overflow=OverflowPolicy.DROP_OLDEST)
    for i in range(5):
        await bus.publish(_trigger(f"p{i}"))
    # Queue holds the 2 newest; 3 oldest dropped.
    first = await sub.get()
    second = await sub.get()
    assert first is not None and second is not None
    assert first.payload.product_id == "p3"  # type: ignore[attr-defined]
    assert second.payload.product_id == "p4"  # type: ignore[attr-defined]
    assert sub.dropped == 3


async def test_block_policy_keeps_all_events() -> None:
    bus = EventBus()
    sub = bus.subscribe("critical", maxsize=100, overflow=OverflowPolicy.BLOCK)
    for i in range(50):
        await bus.publish(_trigger(f"p{i}"))
    received = [await sub.get() for _ in range(50)]
    assert [e.payload.product_id for e in received] == [  # type: ignore[union-attr]
        f"p{i}" for i in range(50)
    ]
    assert sub.dropped == 0


async def test_subscriber_isolation_does_not_break_publish() -> None:
    bus = EventBus()
    good = bus.subscribe("good")

    # A subscription whose delivery raises must not stop other deliveries.
    bad = bus.subscribe("bad")

    async def boom(_: Event) -> None:
        raise RuntimeError("subscriber exploded")

    bad._deliver = boom  # type: ignore[assignment]

    await bus.publish(_trigger("p1"))  # must not raise
    event = await good.get()
    assert event is not None
    assert event.payload.product_id == "p1"  # type: ignore[attr-defined]


async def test_async_iterator_and_close() -> None:
    bus = EventBus()
    sub = bus.subscribe("iter")
    await bus.publish(_trigger("p1"))
    await bus.publish(_trigger("p2"))

    seen: list[str] = []

    async def consume() -> None:
        async for event in sub:
            seen.append(event.payload.product_id)  # type: ignore[attr-defined]

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    await bus.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert seen == ["p1", "p2"]
