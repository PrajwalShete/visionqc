"""Alarm engine — inference failures, lifecycle FAULTs, line-state alarms."""

from __future__ import annotations

import asyncio

from visionqc.alarms.engine import AlarmEngine, Severity
from visionqc.db.database import Database
from visionqc.db.repository import Repository
from visionqc.events.bus import EventBus
from visionqc.events.schemas import (
    InferenceFailed,
    InferenceFailedPayload,
    LineStateChanged,
    LineStateChangedPayload,
    ProductFinalized,
    ProductFinalizedPayload,
)


async def _wait_for_alarms(repo: Repository, expected: int) -> list[dict]:
    for _ in range(50):
        alarms = await repo.list_alarms()
        if len(alarms) >= expected:
            return alarms
        await asyncio.sleep(0.02)
    return await repo.list_alarms()


async def test_inference_failure_raises_critical(database: Database) -> None:
    bus = EventBus()
    repo = Repository(database)
    engine = AlarmEngine(bus, repo)
    engine.start()
    try:
        await bus.publish(
            InferenceFailed(
                payload=InferenceFailedPayload(
                    product_id="p1", reason="inference_timeout", error="boom"
                )
            )
        )
        alarms = await _wait_for_alarms(repo, 1)
    finally:
        await engine.stop()
        await bus.close()

    assert len(alarms) == 1
    assert alarms[0]["code"] == "inference_failed"
    assert alarms[0]["severity"] == Severity.CRITICAL.value
    assert alarms[0]["product_id"] == "p1"


async def test_lifecycle_fault_raises_alarm(database: Database) -> None:
    bus = EventBus()
    repo = Repository(database)
    engine = AlarmEngine(bus, repo)
    engine.start()
    try:
        await bus.publish(
            ProductFinalized(
                payload=ProductFinalizedPayload(
                    product_id="p2", outcome="FAULT", reason="lifecycle_timeout"
                )
            )
        )
        alarms = await _wait_for_alarms(repo, 1)
    finally:
        await engine.stop()
        await bus.close()

    assert alarms[0]["code"] == "lifecycle_fault"
    assert alarms[0]["severity"] == Severity.CRITICAL.value


async def test_pass_finalize_raises_no_alarm(database: Database) -> None:
    bus = EventBus()
    repo = Repository(database)
    engine = AlarmEngine(bus, repo)
    engine.start()
    try:
        await bus.publish(
            ProductFinalized(
                payload=ProductFinalizedPayload(
                    product_id="p3", outcome="PASS", reason="within_tolerance"
                )
            )
        )
        await asyncio.sleep(0.1)
        alarms = await repo.list_alarms()
    finally:
        await engine.stop()
        await bus.close()
    assert alarms == []


async def test_line_degraded_raises_warning(database: Database) -> None:
    bus = EventBus()
    repo = Repository(database)
    engine = AlarmEngine(bus, repo)
    engine.start()
    try:
        await bus.publish(
            LineStateChanged(
                payload=LineStateChangedPayload(state="DEGRADED", reason="worker down")
            )
        )
        alarms = await _wait_for_alarms(repo, 1)
    finally:
        await engine.stop()
        await bus.close()
    assert alarms[0]["code"] == "line_degraded"
    assert alarms[0]["severity"] == Severity.WARNING.value
