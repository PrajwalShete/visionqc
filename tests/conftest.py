"""Shared pytest fixtures: temp SQLite DB, repository, and a TestClient wired
with a deterministic :class:`FakeInferenceClient`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from visionqc.config import get_settings
from visionqc.db.database import Database
from visionqc.db.repository import Repository
from visionqc.events.bus import EventBus
from visionqc.inference_client.client import FakeInferenceClient
from visionqc.lifecycle.tracker import ProductTracker
from visionqc.main import create_app


@pytest.fixture
def tmp_settings(tmp_path: Path):
    """Settings pointing at a temp DB + evidence dir, watchdog tuned fast."""

    return get_settings(
        db_path=tmp_path / "test.db",
        evidence_dir=tmp_path / "evidence",
        lifecycle_timeout_s=0.2,
        watchdog_interval_s=0.05,
    )


@pytest_asyncio.fixture
async def database(tmp_settings) -> AsyncIterator[Database]:
    """A started :class:`Database` on a temp file, torn down after the test."""

    db = Database(tmp_settings.db_path, read_pool_size=tmp_settings.read_pool_size)
    await db.start()
    try:
        yield db
    finally:
        await db.close()


@pytest_asyncio.fixture
async def repo(database: Database) -> Repository:
    return Repository(database)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def tracker(bus: EventBus) -> ProductTracker:
    return ProductTracker(bus, lifecycle_timeout_s=0.2, watchdog_interval_s=0.05)


@pytest.fixture
def client(tmp_settings) -> Iterator[TestClient]:
    """A TestClient with lifespan run and a fake inference client injected."""

    app = create_app(tmp_settings, inference_client=FakeInferenceClient())
    with TestClient(app) as test_client:
        yield test_client
