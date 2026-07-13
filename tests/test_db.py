"""Database layer — migrations idempotency, single-writer concurrency, writer."""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from visionqc.db.database import Database
from visionqc.db.migrations import run_migrations
from visionqc.db.repository import Repository
from visionqc.db.writer import DBWriterSubscriber
from visionqc.events.bus import EventBus
from visionqc.events.schemas import TriggerFired, TriggerFiredPayload


async def test_migrations_idempotent(tmp_path) -> None:
    db_path = tmp_path / "m.db"
    conn = await aiosqlite.connect(db_path)
    try:
        v1 = await run_migrations(conn)
        v2 = await run_migrations(conn)  # second run must be a no-op
        assert v1 == v2 == 1
        cursor = await conn.execute("SELECT COUNT(*) FROM schema_migrations")
        (count,) = await cursor.fetchone()
        assert count == 1
    finally:
        await conn.close()


async def test_concurrent_writes_do_not_lock(database: Database) -> None:
    repo = Repository(database)

    async def insert(i: int) -> None:
        await repo.insert_product(f"p{i}", trigger_ts=float(i), state="TRIGGERED", recipe_id=None)

    # Fire many writes concurrently; the single writer serializes them safely.
    await asyncio.gather(*(insert(i) for i in range(50)))
    rows = await repo.recent_products(limit=100)
    assert len(rows) == 50


async def test_reads_and_writes_interleave(database: Database) -> None:
    repo = Repository(database)

    async def writer() -> None:
        for i in range(30):
            await repo.insert_product(
                f"w{i}", trigger_ts=float(i), state="TRIGGERED", recipe_id=None
            )

    async def reader() -> None:
        for _ in range(30):
            await repo.recent_products(limit=10)

    await asyncio.gather(writer(), reader(), reader())
    assert len(await repo.recent_products(limit=100)) == 30


async def test_write_error_propagates_to_caller(database: Database) -> None:
    async def boom(conn: aiosqlite.Connection) -> None:
        await conn.execute("INSERT INTO nonexistent_table VALUES (1)")

    with pytest.raises(aiosqlite.OperationalError):
        await database.execute_write(boom)

    # Writer survives the failure and still serves subsequent writes.
    repo = Repository(database)
    await repo.insert_product("ok", trigger_ts=1.0, state="TRIGGERED", recipe_id=None)
    assert len(await repo.recent_products()) == 1


async def test_db_writer_subscriber_persists_lifecycle(database: Database) -> None:
    bus = EventBus()
    repo = Repository(database)
    writer = DBWriterSubscriber(bus, repo)
    writer.start()
    try:
        await bus.publish(
            TriggerFired(payload=TriggerFiredPayload(product_id="p1", trigger_ts=1.0))
        )
        # Give the subscriber a moment to consume + persist.
        for _ in range(50):
            detail = await repo.product_detail("p1")
            if detail is not None:
                break
            await asyncio.sleep(0.02)
    finally:
        await writer.stop()
        await bus.close()

    assert detail is not None
    assert detail["state"] == "TRIGGERED"
    assert len(detail["events"]) == 1
    assert detail["events"][0]["event_type"] == "TriggerFired"
