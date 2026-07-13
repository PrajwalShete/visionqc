"""aiosqlite data layer: single dedicated writer + read-only pool.

Design (per the architecture doc, gotcha #3):

* **One** dedicated write connection, fed by an internal ``asyncio.Queue``. All
  writes are serialized through a single writer task, so SQLite never sees
  competing writers ("database is locked" root cause).
* Write transactions use ``BEGIN IMMEDIATE`` to take the write lock up front —
  ``busy_timeout`` alone does not help a read→write upgrade.
* A small pool of **read-only** connections serves dashboard queries without
  ever contending for the write lock.
* WAL pragmas exactly as specified in the architecture doc.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import aiosqlite

from .migrations import run_migrations

logger = logging.getLogger(__name__)

T = TypeVar("T")

WriteFn = Callable[[aiosqlite.Connection], Awaitable[T]]

_PRAGMAS = (
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA busy_timeout=5000;",
    "PRAGMA cache_size=-65536;",
    "PRAGMA temp_store=MEMORY;",
)


class _WriteRequest:
    """A queued write, paired with a future that resolves to its result."""

    __slots__ = ("fn", "future")

    def __init__(self, fn: WriteFn[object], future: asyncio.Future[object]) -> None:
        self.fn = fn
        self.future = future


class Database:
    """Owns the write connection, writer task, and read-only pool."""

    def __init__(self, db_path: Path, read_pool_size: int = 4) -> None:
        self._db_path = db_path
        self._read_pool_size = read_pool_size
        self._write_conn: aiosqlite.Connection | None = None
        self._read_pool: asyncio.Queue[aiosqlite.Connection] | None = None
        self._read_conns: list[aiosqlite.Connection] = []
        self._write_queue: asyncio.Queue[_WriteRequest] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._started = False

    # ---- lifecycle ----------------------------------------------------
    async def start(self) -> None:
        """Open connections, apply pragmas + migrations, launch the writer."""

        if self._started:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_conn = await aiosqlite.connect(self._db_path)
        for pragma in _PRAGMAS:
            await self._write_conn.execute(pragma)
        await self._write_conn.commit()
        await run_migrations(self._write_conn)

        self._read_pool = asyncio.Queue()
        for _ in range(self._read_pool_size):
            conn = await aiosqlite.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA query_only=ON;")
            await conn.execute("PRAGMA busy_timeout=5000;")
            self._read_conns.append(conn)
            self._read_pool.put_nowait(conn)

        self._writer_task = asyncio.create_task(self._writer_loop(), name="db-writer")
        self._started = True
        logger.info("database started at %s", self._db_path)

    async def close(self) -> None:
        """Drain the writer and close all connections."""

        if not self._started:
            return
        self._started = False
        if self._writer_task is not None:
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
        for conn in self._read_conns:
            await conn.close()
        self._read_conns.clear()
        if self._write_conn is not None:
            await self._write_conn.close()
            self._write_conn = None
        logger.info("database closed")

    # ---- writes -------------------------------------------------------
    async def execute_write(self, fn: WriteFn[T]) -> T:
        """Run ``fn`` on the single write connection inside ``BEGIN IMMEDIATE``.

        The call is enqueued and executed by the dedicated writer task; the
        awaited result (or exception) is returned to the caller.
        """

        if not self._started:
            raise RuntimeError("database not started")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()
        await self._write_queue.put(_WriteRequest(fn, future))  # type: ignore[arg-type]
        return await future  # type: ignore[return-value]

    async def _writer_loop(self) -> None:
        """Consume write requests one at a time; serialize all writes."""

        assert self._write_conn is not None
        conn = self._write_conn
        while True:
            request = await self._write_queue.get()
            try:
                await conn.execute("BEGIN IMMEDIATE")
                try:
                    result = await request.fn(conn)
                except Exception:
                    await conn.rollback()
                    raise
                else:
                    await conn.commit()
                if not request.future.done():
                    request.future.set_result(result)
            except asyncio.CancelledError:
                if not request.future.done():
                    request.future.cancel()
                raise
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)
            finally:
                self._write_queue.task_done()

    # ---- reads --------------------------------------------------------
    async def fetch_all(self, sql: str, params: tuple[object, ...] = ()) -> list[aiosqlite.Row]:
        """Run a read query on a pooled read-only connection."""

        conn = await self._acquire_read()
        try:
            cursor = await conn.execute(sql, params)
            return list(await cursor.fetchall())
        finally:
            self._release_read(conn)

    async def fetch_one(self, sql: str, params: tuple[object, ...] = ()) -> aiosqlite.Row | None:
        """Run a read query returning a single row (or ``None``)."""

        conn = await self._acquire_read()
        try:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchone()
        finally:
            self._release_read(conn)

    async def _acquire_read(self) -> aiosqlite.Connection:
        if self._read_pool is None:
            raise RuntimeError("database not started")
        return await self._read_pool.get()

    def _release_read(self, conn: aiosqlite.Connection) -> None:
        assert self._read_pool is not None
        self._read_pool.put_nowait(conn)


__all__ = ["Database", "WriteFn"]
