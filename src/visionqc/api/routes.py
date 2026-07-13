"""REST + WebSocket routes for the dashboard and operators.

Routes read the shared :class:`~visionqc.main.AppContext` off ``app.state.ctx``.
The dashboard task will extend the WebSocket message protocol; the ``/ws``
endpoint already streams every bus event to connected clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..main import AppContext

router = APIRouter()
ws_router = APIRouter()


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class CreateRecipeRequest(BaseModel):
    """Payload to commission a new immutable recipe version."""

    name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    anomaly_threshold: float = Field(..., ge=0.0, le=1.0)
    confidence_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Health & stats
# --------------------------------------------------------------------------- #
@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Line state, inference worker status, and reconciliation counters."""

    ctx = _ctx(request)
    reconciliation = ctx.tracker.reconcile().as_dict()
    worker = await ctx.worker_status()
    return {
        "status": "ok",
        "line_state": ctx.line_state,
        "worker_status": worker["status"],
        "worker": worker,
        "ws_clients": ctx.ws_manager.client_count,
        "reconciliation": reconciliation,
        "zero_silent_loss": reconciliation["lost"] == 0,
    }


@router.get("/stats")
async def stats(request: Request) -> dict[str, Any]:
    """Aggregate counters from live reconciliation and the persisted DB."""

    ctx = _ctx(request)
    return {
        "live": ctx.tracker.reconcile().as_dict(),
        "persisted": await ctx.repo.reconciliation(),
        "active_alarms": len(await ctx.repo.list_alarms(active_only=True)),
    }


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
@router.get("/products")
async def list_products(
    request: Request,
    outcome: str | None = Query(default=None),
    state: str | None = Query(default=None),
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """List/search product records for the traceability view."""

    ctx = _ctx(request)
    if any(v is not None for v in (outcome, state, since, until)):
        items = await ctx.repo.search_products(
            outcome=outcome, state=state, since=since, until=until, limit=limit
        )
    else:
        items = await ctx.repo.recent_products(limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/products/{product_id}")
async def product_detail(request: Request, product_id: str) -> dict[str, Any]:
    """Full product record: row + ordered events + evidence."""

    ctx = _ctx(request)
    detail = await ctx.repo.product_detail(product_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="product not found")
    return detail


# --------------------------------------------------------------------------- #
# Alarms
# --------------------------------------------------------------------------- #
@router.get("/alarms")
async def list_alarms(
    request: Request,
    active_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List alarm records, most recent first."""

    ctx = _ctx(request)
    items = await ctx.repo.list_alarms(active_only=active_only, limit=limit)
    return {"items": items, "count": len(items)}


@router.post("/alarms/{alarm_id}/clear")
async def clear_alarm(request: Request, alarm_id: int) -> dict[str, Any]:
    """Clear (acknowledge) an active alarm."""

    ctx = _ctx(request)
    await ctx.repo.clear_alarm(alarm_id)
    return {"cleared": alarm_id}


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
@router.get("/recipes")
async def list_recipes(request: Request) -> dict[str, Any]:
    """List every recipe version plus the currently active one."""

    ctx = _ctx(request)
    return {
        "items": await ctx.recipes.list_all(),
        "active": await ctx.recipes.get_active(),
    }


@router.post("/recipes", status_code=201)
async def create_recipe(request: Request, body: CreateRecipeRequest) -> dict[str, Any]:
    """Commission a new immutable recipe version (inactive until activated)."""

    ctx = _ctx(request)
    return await ctx.recipes.create_version(
        name=body.name,
        category=body.category,
        model_name=body.model_name,
        anomaly_threshold=body.anomaly_threshold,
        confidence_margin=body.confidence_margin,
        notes=body.notes,
    )


@router.post("/recipes/{recipe_id}/activate")
async def activate_recipe(request: Request, recipe_id: int) -> dict[str, Any]:
    """Activate a recipe version, deactivating any previously active one."""

    ctx = _ctx(request)
    try:
        return await ctx.recipes.activate(recipe_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #
@ws_router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Stream bus events to a dashboard client (drop-oldest backpressure)."""

    ctx: AppContext = websocket.app.state.ctx
    await ctx.ws_manager.serve(websocket)


__all__ = ["router", "ws_router"]
