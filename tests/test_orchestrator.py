"""Orchestrator — end-to-end pipeline with the fake inference client."""

from __future__ import annotations

from pathlib import Path

from visionqc.db.database import Database
from visionqc.db.repository import Repository
from visionqc.events.bus import EventBus
from visionqc.evidence.store import EvidenceStore
from visionqc.inference_client.client import FakeInferenceClient
from visionqc.lifecycle.tracker import ProductTracker
from visionqc.orchestrator import Orchestrator
from visionqc.recipes.service import RecipeService


async def _build(
    database: Database, tmp_path: Path, *, fail: bool, threshold: float
) -> tuple[Orchestrator, ProductTracker, Repository]:
    bus = EventBus()
    repo = Repository(database)
    tracker = ProductTracker(bus)
    recipes = RecipeService(repo)
    recipe = await recipes.create_version("bottle", "bottle", "padim", threshold)
    await recipes.activate(recipe["id"])
    orch = Orchestrator(
        tracker=tracker,
        inference=FakeInferenceClient(fail=fail),
        recipes=recipes,
        evidence=EvidenceStore(tmp_path / "evidence"),
        repo=repo,
        inference_timeout_s=1.0,
    )
    return orch, tracker, repo


IMAGE = b"\xff\xd8sample-frame\xff\xd9"


async def test_inspect_pass(database: Database, tmp_path: Path) -> None:
    score = FakeInferenceClient.score_for(IMAGE)
    orch, tracker, _ = await _build(database, tmp_path, fail=False, threshold=score + 0.1)
    pid = await tracker.trigger()
    await orch.inspect(pid, IMAGE)
    rec = tracker.reconcile()
    assert rec.passed == 1
    assert rec.lost == 0
    assert orch.degraded is False


async def test_inspect_reject(database: Database, tmp_path: Path) -> None:
    score = FakeInferenceClient.score_for(IMAGE)
    threshold = max(score - 0.1, 0.0)
    orch, tracker, _ = await _build(database, tmp_path, fail=False, threshold=threshold)
    pid = await tracker.trigger()
    await orch.inspect(pid, IMAGE)
    rec = tracker.reconcile()
    assert rec.rejected == 1
    assert rec.lost == 0


async def test_inspect_inference_failure_faults_and_degrades(
    database: Database, tmp_path: Path
) -> None:
    orch, tracker, _ = await _build(database, tmp_path, fail=True, threshold=0.5)
    pid = await tracker.trigger()
    await orch.inspect(pid, IMAGE)
    rec = tracker.reconcile()
    assert rec.fault == 1
    assert rec.lost == 0
    assert orch.degraded is True


async def test_inspect_saves_evidence(database: Database, tmp_path: Path) -> None:
    score = FakeInferenceClient.score_for(IMAGE)
    orch, tracker, repo = await _build(database, tmp_path, fail=False, threshold=score + 0.1)
    pid = await tracker.trigger()
    await orch.inspect(pid, IMAGE)
    evidence = await repo.evidence_for(pid)
    assert any(e["kind"] == "raw" for e in evidence)


async def test_inspect_without_active_recipe_faults(database: Database, tmp_path: Path) -> None:
    bus = EventBus()
    repo = Repository(database)
    tracker = ProductTracker(bus)
    orch = Orchestrator(
        tracker=tracker,
        inference=FakeInferenceClient(),
        recipes=RecipeService(repo),  # no recipe created/activated
        evidence=EvidenceStore(tmp_path / "e"),
        repo=repo,
        inference_timeout_s=1.0,
    )
    pid = await tracker.trigger()
    await orch.inspect(pid, IMAGE)
    assert tracker.reconcile().fault == 1
