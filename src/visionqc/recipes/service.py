"""Recipe service: immutable, versioned inspection recipes backed by the DB.

A recipe is never updated in place. Commissioning a change means creating a new
version; operators then *activate* exactly one version. Every product record
stores the ``recipe_id`` it was inspected under, so traceability survives future
recipe changes.
"""

from __future__ import annotations

import json
from typing import Any

from ..db.repository import Repository
from ..decision.engine import RecipeParams


def recipe_to_params(recipe: dict[str, Any]) -> RecipeParams:
    """Convert a stored recipe row into :class:`RecipeParams` for the engine."""

    params = recipe.get("params_json")
    extra = json.loads(params) if isinstance(params, str) and params else {}
    return RecipeParams(
        anomaly_threshold=float(recipe["anomaly_threshold"]),
        confidence_margin=float(extra.get("confidence_margin", 0.0)),
    )


class RecipeService:
    """Thin domain wrapper over the recipe repository methods."""

    def __init__(self, repo: Repository) -> None:
        self._repo = repo

    async def create_version(
        self,
        name: str,
        category: str,
        model_name: str,
        anomaly_threshold: float,
        confidence_margin: float = 0.0,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Create a new immutable recipe version (inactive until activated)."""

        return await self._repo.create_recipe_version(
            name=name,
            category=category,
            model_name=model_name,
            anomaly_threshold=anomaly_threshold,
            params={"confidence_margin": confidence_margin},
            notes=notes,
        )

    async def activate(self, recipe_id: int) -> dict[str, Any]:
        """Activate a version, deactivating any previously active recipe."""

        return await self._repo.activate_recipe(recipe_id)

    async def get_active(self) -> dict[str, Any] | None:
        """Return the currently active recipe row, or ``None``."""

        return await self._repo.get_active_recipe()

    async def get_active_params(self) -> RecipeParams | None:
        """Return decision params for the active recipe, or ``None``."""

        active = await self._repo.get_active_recipe()
        return recipe_to_params(active) if active else None

    async def list_all(self) -> list[dict[str, Any]]:
        """List every recipe version, grouped by name then newest version."""

        return await self._repo.list_recipes()

    async def get(self, recipe_id: int) -> dict[str, Any] | None:
        return await self._repo.get_recipe(recipe_id)


__all__ = ["RecipeService", "recipe_to_params"]
