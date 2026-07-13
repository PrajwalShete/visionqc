"""Inspection orchestrator — glue that runs a product through the pipeline.

Ties the lifecycle tracker, inference client, decision engine, recipes, and
evidence store together for a single product:

``capture → infer → (decide → PASS/REJECT) | FAULT``

Inference calls are wrapped in ``asyncio.wait_for`` with the configured timeout;
an :class:`InferenceUnavailable` (or timeout) maps to ``InferenceFailed`` +
forced FAULT + a degraded-mode flag, never a hang and never a silent pass. The
simulator (separate task) will drive this with real frames; tests drive it with
the fake client.
"""

from __future__ import annotations

import asyncio
import logging

from .db.repository import Repository
from .decision.engine import Disposition, InferenceResult, decide
from .evidence.store import EvidenceStore
from .inference_client.client import InferenceClient, InferenceUnavailable
from .lifecycle.tracker import ProductTracker
from .recipes.service import RecipeService, recipe_to_params

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs the per-product inspection pipeline."""

    def __init__(
        self,
        *,
        tracker: ProductTracker,
        inference: InferenceClient,
        recipes: RecipeService,
        evidence: EvidenceStore,
        repo: Repository,
        inference_timeout_s: float,
    ) -> None:
        self._tracker = tracker
        self._inference = inference
        self._recipes = recipes
        self._evidence = evidence
        self._repo = repo
        self._timeout_s = inference_timeout_s
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """Whether the last inference attempt failed (worker degraded)."""

        return self._degraded

    async def inspect(self, product_id: str, image: bytes) -> None:
        """Run a captured product through inference, decision, and finalization.

        The product must already be TRIGGERED (via ``tracker.trigger``).
        """

        await self._tracker.mark_captured(product_id, width=None, height=None)
        await self._save_evidence(product_id, "raw", image)

        try:
            response = await asyncio.wait_for(self._inference.infer(image), timeout=self._timeout_s)
        except (InferenceUnavailable, TimeoutError) as exc:
            self._degraded = True
            reason = "inference_timeout" if isinstance(exc, TimeoutError) else "inference_error"
            await self._tracker.mark_inference_failed(product_id, reason=reason, error=str(exc))
            await self._tracker.force_fault(product_id, reason=reason)
            return

        self._degraded = False
        await self._tracker.mark_inferred(
            product_id,
            score=response.score,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )
        if response.heatmap_jpeg:
            await self._save_evidence(product_id, "heatmap", response.heatmap_jpeg)

        active = await self._recipes.get_active()
        if active is None:
            await self._tracker.force_fault(product_id, reason="no_active_recipe")
            return

        params = recipe_to_params(active)
        result = InferenceResult(
            score=response.score,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )
        decision = decide(result, params)

        await self._tracker.mark_decided(
            product_id,
            outcome=decision.disposition.value,
            reason=decision.reason.value,
            score=decision.score,
            recipe_id=int(active["id"]),
        )

        if decision.disposition is Disposition.PASS:
            await self._tracker.finalize_pass(
                product_id, reason=decision.reason.value, score=decision.score
            )
        elif decision.disposition is Disposition.REJECT:
            await self._tracker.finalize_reject(
                product_id, reason=decision.reason.value, score=decision.score
            )
        else:  # FAULT from an invalid score
            await self._tracker.force_fault(product_id, reason=decision.reason.value)

    async def _save_evidence(self, product_id: str, kind: str, data: bytes) -> None:
        try:
            ref = await self._evidence.save_jpeg(product_id, kind, data)
            await self._repo.insert_evidence(product_id, ref.kind, ref.path, ref.sha256, ref.mime)
        except Exception:
            logger.exception("failed to store %s evidence for %s", kind, product_id)


__all__ = ["Orchestrator"]
