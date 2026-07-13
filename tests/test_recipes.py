"""Recipes — immutability, versioning, single-active activation."""

from __future__ import annotations

from visionqc.db.repository import Repository
from visionqc.recipes.service import RecipeService, recipe_to_params


async def test_create_version_increments(repo: Repository) -> None:
    service = RecipeService(repo)
    v1 = await service.create_version("bottle", "bottle", "padim", 0.5)
    v2 = await service.create_version("bottle", "bottle", "patchcore", 0.6)
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v1["id"] != v2["id"]


async def test_versions_are_immutable(repo: Repository) -> None:
    service = RecipeService(repo)
    v1 = await service.create_version("bottle", "bottle", "padim", 0.5)
    # Creating a "new" recipe never mutates the old row.
    await service.create_version("bottle", "bottle", "padim", 0.9)
    reloaded = await service.get(v1["id"])
    assert reloaded is not None
    assert reloaded["anomaly_threshold"] == 0.5
    assert reloaded["version"] == 1


async def test_activation_is_exclusive(repo: Repository) -> None:
    service = RecipeService(repo)
    v1 = await service.create_version("bottle", "bottle", "padim", 0.5)
    v2 = await service.create_version("bottle", "bottle", "patchcore", 0.6)

    await service.activate(v1["id"])
    active = await service.get_active()
    assert active is not None and active["id"] == v1["id"]

    await service.activate(v2["id"])
    active = await service.get_active()
    assert active is not None and active["id"] == v2["id"]

    # Only one active at a time.
    all_active = [r for r in await service.list_all() if r["active"] == 1]
    assert len(all_active) == 1


async def test_activate_unknown_raises(repo: Repository) -> None:
    service = RecipeService(repo)
    import pytest

    with pytest.raises(KeyError):
        await service.activate(999)


async def test_recipe_to_params_reads_margin(repo: Repository) -> None:
    service = RecipeService(repo)
    recipe = await service.create_version("bottle", "bottle", "padim", 0.5, confidence_margin=0.05)
    params = recipe_to_params(recipe)
    assert params.anomaly_threshold == 0.5
    assert params.confidence_margin == 0.05
