"""Production-line simulator: image sources, full simulated runs, fault injection."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from visionqc.alarms.engine import AlarmEngine, Severity
from visionqc.db.database import Database
from visionqc.db.repository import Repository
from visionqc.events.bus import EventBus, OverflowPolicy, Subscription
from visionqc.events.schemas import Event, EventType
from visionqc.evidence.store import EvidenceStore
from visionqc.inference_client.client import FakeInferenceClient
from visionqc.lifecycle.tracker import ProductTracker
from visionqc.orchestrator import Orchestrator
from visionqc.recipes.service import RecipeService
from visionqc.simulator.line import CAMERA_LOSS, REJECT_FAILURE, LineSimulator
from visionqc.simulator.source import (
    DirectoryImageSource,
    ImageSource,
    SourceImage,
    SyntheticImageSource,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
JPEG_SOI = b"\xff\xd8"


async def _wait(predicate: Callable[[], bool], *, timeout: float = 3.0) -> bool:
    """Poll ``predicate`` until true or the timeout elapses."""

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def _collect(sub: Subscription) -> list[Event]:
    """Drain every currently-queued event from a capture subscription."""

    events: list[Event] = []
    while True:
        try:
            event = await asyncio.wait_for(sub.get(), timeout=0.1)
        except TimeoutError:
            break
        if event is None:
            break
        events.append(event)
    return events


@contextlib.asynccontextmanager
async def _line(
    database: Database,
    tmp_path: Path,
    *,
    threshold: float = 0.5,
    source: ImageSource | None = None,
    interval_s: float = 0.01,
    lifecycle_timeout_s: float = 0.2,
    reject_actuation_s: float = 0.0,
    with_alarms: bool = False,
) -> AsyncIterator[tuple[LineSimulator, ProductTracker, EventBus, Repository]]:
    """Build a fully-wired simulator over a temp DB, and tear it down cleanly."""

    bus = EventBus()
    repo = Repository(database)
    tracker = ProductTracker(bus, lifecycle_timeout_s=lifecycle_timeout_s, watchdog_interval_s=0.02)
    tracker.start_watchdog()
    recipes = RecipeService(repo)
    recipe = await recipes.create_version("synthetic", "synthetic", "padim", threshold)
    await recipes.activate(recipe["id"])
    inference = FakeInferenceClient()
    orchestrator = Orchestrator(
        tracker=tracker,
        inference=inference,
        recipes=recipes,
        evidence=EvidenceStore(tmp_path / "evidence"),
        repo=repo,
        inference_timeout_s=1.0,
    )
    simulator = LineSimulator(
        bus=bus,
        tracker=tracker,
        orchestrator=orchestrator,
        recipes=recipes,
        inference=inference,
        source=source or SyntheticImageSource(defect_rate=0.3, seed=1),
        interval_s=interval_s,
        inference_timeout_s=1.0,
        reject_actuation_s=reject_actuation_s,
    )
    alarms = AlarmEngine(bus, repo) if with_alarms else None
    if alarms is not None:
        alarms.start()
    try:
        yield simulator, tracker, bus, repo
    finally:
        await simulator.stop()
        await tracker.stop_watchdog()
        if alarms is not None:
            await alarms.stop()
        await bus.close()


def _capture(bus: EventBus) -> Subscription:
    """A non-lossy subscription capturing every event for later inspection."""

    return bus.subscribe(
        "test-capture", event_types=list(EventType), maxsize=10000, overflow=OverflowPolicy.BLOCK
    )


# --------------------------------------------------------------------------- #
# DirectoryImageSource
# --------------------------------------------------------------------------- #
async def test_directory_source_loops_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "mvtec"
    (root / "good").mkdir(parents=True)
    (root / "broken").mkdir(parents=True)
    (root / "good" / "g0.png").write_bytes(b"\x89PNG-good-0")
    (root / "good" / "g1.png").write_bytes(b"\x89PNG-good-1")
    (root / "broken" / "b0.png").write_bytes(b"\x89PNG-broken-0")

    src = DirectoryImageSource(root)
    frames = [await src.next_image() for _ in range(6)]  # two full cycles of 3 files

    # Sorted path order: broken/b0, good/g0, good/g1.
    labels = [f.label for f in frames]
    assert labels == ["broken", "good", "good", "broken", "good", "good"]
    assert frames[0].is_defect is True
    assert frames[1].is_defect is False
    # Looping: frame 3 repeats frame 0's bytes.
    assert frames[3].data == frames[0].data
    assert all(f.source == "directory" for f in frames)
    assert src.describe()["image_count"] == 3


async def test_directory_source_requires_images(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        DirectoryImageSource(empty)
    with pytest.raises(FileNotFoundError):
        DirectoryImageSource(tmp_path / "does-not-exist")


# --------------------------------------------------------------------------- #
# SyntheticImageSource
# --------------------------------------------------------------------------- #
async def test_synthetic_source_defect_ratio() -> None:
    src = SyntheticImageSource(defect_rate=0.3, seed=42)
    frames = [await src.next_image() for _ in range(300)]

    defects = sum(1 for f in frames if f.is_defect)
    ratio = defects / len(frames)
    assert 0.22 <= ratio <= 0.38  # ~0.3 within sampling tolerance
    # Every frame is a valid JPEG, and labels match the defect flag.
    assert all(f.data[:2] == JPEG_SOI for f in frames)
    assert all((f.label == "defect") == f.is_defect for f in frames)
    assert src.describe()["generated"] == 300


async def test_synthetic_source_zero_and_full_defect_rates() -> None:
    none = SyntheticImageSource(defect_rate=0.0, seed=7)
    none_defects = [(await none.next_image()).is_defect for _ in range(40)]
    assert not any(none_defects)
    every = SyntheticImageSource(defect_rate=1.0, seed=7)
    every_defects = [(await every.next_image()).is_defect for _ in range(40)]
    assert all(every_defects)


async def test_synthetic_source_rejects_bad_defect_rate() -> None:
    with pytest.raises(ValueError):
        SyntheticImageSource(defect_rate=1.5)


# --------------------------------------------------------------------------- #
# Full simulated runs
# --------------------------------------------------------------------------- #
async def test_full_run_reaches_terminal_no_loss(database: Database, tmp_path: Path) -> None:
    async with _line(database, tmp_path, threshold=0.5) as (sim, tracker, _bus, _repo):
        await sim.start()
        assert await _wait(lambda: tracker.reconcile().terminal >= 8)
        await sim.stop()
        # After stopping, the watchdog drains any product left mid-flight.
        assert await _wait(lambda: tracker.reconcile().in_flight == 0)

    rec = tracker.reconcile()
    assert rec.triggered >= 8
    assert rec.terminal == rec.triggered
    assert rec.in_flight == 0
    assert rec.lost == 0


async def test_reject_products_emit_confirmation(database: Database, tmp_path: Path) -> None:
    # threshold 0.0 → every product scores >= threshold → REJECT.
    async with _line(database, tmp_path, threshold=0.0) as (sim, tracker, bus, _repo):
        sub = _capture(bus)
        await sim.start()
        assert await _wait(lambda: tracker.reconcile().rejected >= 5)
        await sim.stop()
        events = await _collect(sub)

    finalized_rejects = {
        e.payload.product_id
        for e in events
        if e.type is EventType.PRODUCT_FINALIZED and e.payload.outcome == "REJECT"
    }
    confirmed = {e.payload.product_id for e in events if e.type is EventType.REJECT_CONFIRMED}
    assert finalized_rejects  # rejects actually happened
    assert finalized_rejects <= confirmed  # every REJECT was physically confirmed
    assert tracker.reconcile().lost == 0


# --------------------------------------------------------------------------- #
# Fault injection
# --------------------------------------------------------------------------- #
async def test_camera_loss_faults_and_line_keeps_running(
    database: Database, tmp_path: Path
) -> None:
    async with _line(database, tmp_path, with_alarms=True) as (sim, tracker, _bus, repo):
        sim.set_fault(CAMERA_LOSS, True)
        await sim.start()
        # The line keeps running while faulting product after product.
        assert await _wait(lambda: tracker.reconcile().fault >= 3 and sim.running)
        assert sim.running is True
        await sim.stop()
        alarms = await _wait_alarms(repo, minimum=1)

    rec = tracker.reconcile()
    assert rec.fault >= 3
    assert rec.passed == 0 and rec.rejected == 0
    assert rec.lost == 0
    critical = [a for a in alarms if a["severity"] == Severity.CRITICAL.value]
    assert any(a["code"] == "lifecycle_fault" for a in critical)


async def test_reject_failure_watchdog_faults_with_critical_alarm(
    database: Database, tmp_path: Path
) -> None:
    # threshold 0.0 → all REJECT; reject_failure withholds the confirmation.
    async with _line(
        database, tmp_path, threshold=0.0, lifecycle_timeout_s=0.15, with_alarms=True
    ) as (sim, tracker, bus, repo):
        sub = _capture(bus)
        sim.set_fault(REJECT_FAILURE, True)
        await sim.start()
        assert await _wait(lambda: tracker.reconcile().fault >= 2, timeout=4.0)
        assert sim.running is True  # line survives the watchdog faults
        await sim.stop()
        events = await _collect(sub)
        alarms = await _wait_alarms(repo, minimum=1)

    commanded = {e.payload.product_id for e in events if e.type is EventType.REJECT_COMMANDED}
    confirmed = {e.payload.product_id for e in events if e.type is EventType.REJECT_CONFIRMED}
    assert commanded  # rejects were commanded
    assert not confirmed  # ...but never confirmed (actuator jammed)
    rec = tracker.reconcile()
    assert rec.fault >= 2
    assert rec.rejected == 0
    assert rec.lost == 0
    assert any(
        a["code"] == "lifecycle_fault" and a["severity"] == Severity.CRITICAL.value for a in alarms
    )


async def test_tick_exception_is_isolated(database: Database, tmp_path: Path) -> None:
    """A crashing image source faults its product but never kills the loop."""

    class _BrokenSource(ImageSource):
        name = "broken"

        async def next_image(self) -> SourceImage:
            raise RuntimeError("camera exploded")

        def describe(self) -> dict[str, Any]:
            return {"type": "broken", "name": self.name}

    async with _line(database, tmp_path, source=_BrokenSource()) as (sim, tracker, _bus, _repo):
        await sim.start()
        # Loop keeps ticking and faulting despite every tick raising.
        assert await _wait(lambda: tracker.reconcile().fault >= 3 and sim.status()["ticks"] >= 3)
        assert sim.running is True
        await sim.stop()

    assert tracker.reconcile().lost == 0


# --------------------------------------------------------------------------- #
# Runtime controls
# --------------------------------------------------------------------------- #
async def test_runtime_control_validation(database: Database, tmp_path: Path) -> None:
    async with _line(database, tmp_path) as (sim, _tracker, _bus, _repo):
        with pytest.raises(ValueError):
            sim.set_interval(0.0)
        with pytest.raises(ValueError):
            sim.set_fault("not_a_fault", True)
        sim.set_interval(0.5)
        assert sim.status()["interval_s"] == 0.5
        sim.set_fault(CAMERA_LOSS, True)
        assert CAMERA_LOSS in sim.status()["active_faults"]
        sim.set_fault(CAMERA_LOSS, False)
        assert CAMERA_LOSS not in sim.status()["active_faults"]


async def _wait_alarms(repo: Repository, *, minimum: int) -> list[dict[str, Any]]:
    for _ in range(100):
        alarms = await repo.list_alarms()
        if len(alarms) >= minimum:
            return alarms
        await asyncio.sleep(0.02)
    return await repo.list_alarms()
