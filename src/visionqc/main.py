"""FastAPI application factory and lifespan wiring.

``create_app`` builds the modular monolith: the event bus, database + repository,
the DB-writer subscriber, the lifecycle tracker + watchdog, the alarm engine,
the recipe service, the inference client, the orchestrator, and the WebSocket
hub. The lifespan starts every subscriber and background task on startup and
tears them down cleanly on shutdown.

Pass ``inference_client`` to inject a :class:`FakeInferenceClient` (tests /
GPU-less local runs); otherwise an :class:`HTTPInferenceClient` targets the
configured worker URL.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI

from .alarms.engine import AlarmEngine
from .api.ws import ConnectionManager
from .config import Settings, get_settings
from .db.database import Database
from .db.repository import Repository
from .db.writer import DBWriterSubscriber
from .events.bus import EventBus, OverflowPolicy, Subscription
from .events.schemas import EventType, LineStateChanged, LineStateChangedPayload
from .evidence.store import EvidenceStore
from .inference_client.client import HTTPInferenceClient, InferenceClient
from .lifecycle.tracker import ProductTracker
from .orchestrator import Orchestrator
from .recipes.service import RecipeService

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Container for every long-lived component, stored on ``app.state.ctx``."""

    settings: Settings
    bus: EventBus
    db: Database
    repo: Repository
    tracker: ProductTracker
    recipes: RecipeService
    alarms: AlarmEngine
    db_writer: DBWriterSubscriber
    inference: InferenceClient
    evidence: EvidenceStore
    orchestrator: Orchestrator
    ws_manager: ConnectionManager
    line_state: str = "RUNNING"
    _ws_sub: Subscription | None = field(default=None, repr=False)
    _ws_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def worker_status(self) -> str:
        """Report inference worker health for the health endpoint."""

        return "degraded" if self.orchestrator.degraded else "ok"

    async def set_line_state(self, state: str, reason: str | None = None) -> None:
        """Update the line state and publish a ``LineStateChanged`` event."""

        self.line_state = state
        await self.bus.publish(
            LineStateChanged(payload=LineStateChangedPayload(state=state, reason=reason))
        )


def _build_context(settings: Settings, inference_client: InferenceClient | None) -> AppContext:
    bus = EventBus()
    db = Database(settings.db_path, read_pool_size=settings.read_pool_size)
    repo = Repository(db)
    tracker = ProductTracker(
        bus,
        lifecycle_timeout_s=settings.lifecycle_timeout_s,
        watchdog_interval_s=settings.watchdog_interval_s,
    )
    recipes = RecipeService(repo)
    alarms = AlarmEngine(bus, repo)
    db_writer = DBWriterSubscriber(bus, repo)
    inference: InferenceClient = inference_client or HTTPInferenceClient(
        settings.inference_worker_url, settings.inference_timeout_s
    )
    evidence = EvidenceStore(settings.evidence_dir)
    orchestrator = Orchestrator(
        tracker=tracker,
        inference=inference,
        recipes=recipes,
        evidence=evidence,
        repo=repo,
        inference_timeout_s=settings.inference_timeout_s,
    )
    ws_manager = ConnectionManager(queue_maxsize=settings.ws_queue_maxsize)
    return AppContext(
        settings=settings,
        bus=bus,
        db=db,
        repo=repo,
        tracker=tracker,
        recipes=recipes,
        alarms=alarms,
        db_writer=db_writer,
        inference=inference,
        evidence=evidence,
        orchestrator=orchestrator,
        ws_manager=ws_manager,
    )


async def _ws_bridge(ctx: AppContext) -> None:
    """Forward every bus event to all WebSocket clients (drop-oldest)."""

    assert ctx._ws_sub is not None
    async for event in ctx._ws_sub:
        ctx.ws_manager.broadcast(event.to_wire())


def create_app(
    settings: Settings | None = None,
    *,
    inference_client: InferenceClient | None = None,
) -> FastAPI:
    """Build and return the wired FastAPI application."""

    settings = settings or get_settings()
    ctx = _build_context(settings, inference_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await ctx.db.start()
        ctx.db_writer.start()
        ctx.alarms.start()
        ctx.tracker.start_watchdog()
        ctx._ws_sub = ctx.bus.subscribe(
            "ws-bridge",
            event_types=list(EventType),
            maxsize=256,
            overflow=OverflowPolicy.DROP_OLDEST,
        )
        ctx._ws_task = asyncio.create_task(_ws_bridge(ctx), name="ws-bridge")
        logger.info("VisionQC line controller started")
        try:
            yield
        finally:
            await ctx.tracker.stop_watchdog()
            if ctx._ws_task is not None:
                ctx._ws_task.cancel()
            if ctx._ws_sub is not None:
                ctx.bus.unsubscribe(ctx._ws_sub)
            await ctx.alarms.stop()
            await ctx.db_writer.stop()
            await ctx.bus.close()
            await ctx.inference.close()
            await ctx.db.close()
            logger.info("VisionQC line controller stopped")

    app = FastAPI(title="VisionQC Line Controller", version="0.1.0", lifespan=lifespan)
    app.state.ctx = ctx

    from .api.routes import router as rest_router
    from .api.routes import ws_router

    app.include_router(rest_router)
    app.include_router(ws_router)
    return app


__all__ = ["AppContext", "create_app"]
