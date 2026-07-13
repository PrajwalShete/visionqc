"""Line-control REST endpoints for the demo operator.

These routes let an operator drive the virtual production line live: start/stop
the conveyor, read its status, change the trigger speed, toggle injectable
faults, and switch the image source between synthetic and a directory of real
images. Every route reads the shared :class:`~visionqc.main.AppContext` off
``app.state.ctx`` and delegates to the :class:`~visionqc.simulator.line.LineSimulator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..simulator.line import KNOWN_FAULTS
from ..simulator.source import build_source

if TYPE_CHECKING:
    from ..main import AppContext
    from ..simulator.line import LineSimulator

router = APIRouter(prefix="/line", tags=["line"])


def _simulator(request: Request) -> LineSimulator:
    ctx: AppContext = request.app.state.ctx
    if ctx.simulator is None:  # pragma: no cover - always wired in create_app
        raise HTTPException(status_code=503, detail="simulator not configured")
    return ctx.simulator


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class SpeedRequest(BaseModel):
    """New trigger interval, in seconds."""

    interval_s: float = Field(..., gt=0.0, le=60.0)


class FaultRequest(BaseModel):
    """Toggle a single injectable fault on or off."""

    fault: str = Field(..., min_length=1)
    enabled: bool = True


class SourceRequest(BaseModel):
    """Select the active image source."""

    type: Literal["synthetic", "directory"]
    path: str | None = None
    defect_rate: float = Field(default=0.2, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post("/start")
async def start_line(request: Request) -> dict[str, Any]:
    """Start the conveyor loop (idempotent)."""

    sim = _simulator(request)
    await sim.start()
    return sim.status()


@router.post("/stop")
async def stop_line(request: Request) -> dict[str, Any]:
    """Stop the conveyor loop (idempotent)."""

    sim = _simulator(request)
    await sim.stop()
    return sim.status()


@router.get("/status")
async def line_status(request: Request) -> dict[str, Any]:
    """Report line state, speed, counts, active faults, and source info."""

    return _simulator(request).status()


@router.post("/speed")
async def set_speed(request: Request, body: SpeedRequest) -> dict[str, Any]:
    """Set the trigger interval (takes effect on the next tick)."""

    sim = _simulator(request)
    try:
        sim.set_interval(body.interval_s)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return sim.status()


@router.post("/faults")
async def set_fault(request: Request, body: FaultRequest) -> dict[str, Any]:
    """Enable or disable an injectable fault by name."""

    sim = _simulator(request)
    if body.fault not in KNOWN_FAULTS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown fault {body.fault!r}; known: {sorted(KNOWN_FAULTS)}",
        )
    sim.set_fault(body.fault, body.enabled)
    return sim.status()


@router.post("/source")
async def set_source(request: Request, body: SourceRequest) -> dict[str, Any]:
    """Switch the active image source (synthetic or a directory of images)."""

    sim = _simulator(request)
    try:
        source = build_source(body.type, path=body.path, defect_rate=body.defect_rate)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sim.set_source(source)
    return sim.status()


__all__ = ["router"]
