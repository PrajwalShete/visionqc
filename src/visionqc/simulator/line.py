"""The conveyor loop — trigger generator, virtual reject station, fault injection.

:class:`LineSimulator` is a single background task modelling a physical
inspection line. On each tick it:

1. fires a trigger (creating a product in the lifecycle tracker),
2. captures a frame from the active :class:`~visionqc.simulator.source.ImageSource`,
3. drives the real orchestrator through ``capture → infer → decide``, and
4. simulates the physical reject: for a REJECT the reject chain
   (``RejectCommanded`` → actuation delay → ``RejectConfirmed``) completes the
   proof-of-rejection.

**Fault injection** is the demo's headline feature. Two faults can be toggled
live via the API:

* ``camera_loss`` — the virtual camera returns nothing, so the product is forced
  to FAULT (raising a CRITICAL alarm) while the line keeps running.
* ``reject_failure`` — the reject actuator jams: the reject is commanded but the
  confirmation never arrives, so the lifecycle watchdog forces the product to
  FAULT (a CRITICAL alarm) after the configured timeout.

Robustness: every tick is wrapped so a single failure can never kill the loop,
and an unexpected error still forces the in-flight product to FAULT (raising an
alarm) rather than silently losing it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from ..decision.engine import Disposition, InferenceResult, decide
from ..events.bus import EventBus
from ..events.schemas import (
    LineStateChanged,
    LineStateChangedPayload,
    RejectCommanded,
    RejectCommandedPayload,
)
from ..inference_client.client import InferenceClient, InferenceUnavailable
from ..lifecycle.tracker import ProductTracker
from ..orchestrator import Orchestrator
from ..recipes.service import RecipeService, recipe_to_params
from .source import ImageSource

logger = logging.getLogger(__name__)

#: Fault: the virtual camera yields no frame → product FAULTs immediately.
CAMERA_LOSS = "camera_loss"
#: Fault: the reject actuator never confirms → watchdog FAULTs the product.
REJECT_FAILURE = "reject_failure"
#: Every fault an operator may toggle at runtime.
KNOWN_FAULTS: frozenset[str] = frozenset({CAMERA_LOSS, REJECT_FAILURE})

StateHook = Callable[[str, str | None], Awaitable[None]]


class LineState(str, Enum):
    """Coarse operational state of the simulated line."""

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class LineSimulator:
    """Async conveyor loop driving the orchestrator with injected frames/faults."""

    def __init__(
        self,
        *,
        bus: EventBus,
        tracker: ProductTracker,
        orchestrator: Orchestrator,
        recipes: RecipeService,
        inference: InferenceClient,
        source: ImageSource,
        interval_s: float = 2.0,
        inference_timeout_s: float = 2.5,
        reject_actuation_s: float = 0.05,
        set_line_state: StateHook | None = None,
    ) -> None:
        self._bus = bus
        self._tracker = tracker
        self._orchestrator = orchestrator
        self._recipes = recipes
        self._inference = inference
        self._source = source
        self._interval_s = interval_s
        self._inference_timeout_s = inference_timeout_s
        self._reject_actuation_s = reject_actuation_s
        self._set_line_state = set_line_state

        self._state = LineState.STOPPED
        self._faults: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        # Telemetry.
        self._ticks = 0
        self._camera_losses = 0
        self._reject_failures = 0
        self._last_cycle_ms: float | None = None
        self._cycle_total_ms = 0.0
        self._cycle_samples = 0
        self._last_label: str | None = None

    # ---- lifecycle ----------------------------------------------------
    @property
    def running(self) -> bool:
        """Whether the conveyor loop is currently active."""

        return self._state is LineState.RUNNING

    async def start(self) -> None:
        """Start the conveyor loop (idempotent) and publish ``RUNNING``."""

        async with self._lock:
            if self._task is not None and not self._task.done():
                return
            self._state = LineState.RUNNING
            self._task = asyncio.create_task(self._run(), name="line-simulator")
        await self._announce(LineState.RUNNING, reason="operator start")

    async def stop(self) -> None:
        """Stop the conveyor loop (idempotent) and publish ``STOPPED``."""

        async with self._lock:
            task = self._task
            self._task = None
            self._state = LineState.STOPPED
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._announce(LineState.STOPPED, reason="operator stop")

    async def _announce(self, state: LineState, *, reason: str | None) -> None:
        if self._set_line_state is not None:
            await self._set_line_state(state.value, reason)
        else:  # standalone / test usage: publish directly.
            await self._bus.publish(
                LineStateChanged(payload=LineStateChangedPayload(state=state.value, reason=reason))
            )

    # ---- runtime controls --------------------------------------------
    def set_interval(self, interval_s: float) -> None:
        """Set the trigger interval in seconds (takes effect next tick)."""

        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s

    def set_fault(self, fault: str, enabled: bool) -> None:
        """Enable or disable an injectable fault by name."""

        if fault not in KNOWN_FAULTS:
            raise ValueError(f"unknown fault: {fault!r} (known: {sorted(KNOWN_FAULTS)})")
        if enabled:
            self._faults.add(fault)
        else:
            self._faults.discard(fault)

    def set_source(self, source: ImageSource) -> None:
        """Swap the active image source (takes effect next tick)."""

        self._source = source

    # ---- status -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot for ``GET /line/status``."""

        avg = self._cycle_total_ms / self._cycle_samples if self._cycle_samples else None
        return {
            "state": self._state.value,
            "running": self.running,
            "interval_s": self._interval_s,
            "active_faults": sorted(self._faults),
            "source": self._source.describe(),
            "counters": self._tracker.reconcile().as_dict(),
            "ticks": self._ticks,
            "camera_losses": self._camera_losses,
            "reject_failures": self._reject_failures,
            "last_label": self._last_label,
            "last_cycle_ms": self._last_cycle_ms,
            "avg_cycle_ms": avg,
        }

    # ---- the loop -----------------------------------------------------
    async def _run(self) -> None:
        logger.info("line simulator loop started")
        try:
            while True:
                await self._tick()
                await asyncio.sleep(self._interval_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive: loop must never die
            logger.exception("line simulator loop crashed; stopping")
            self._state = LineState.STOPPED
            with contextlib.suppress(Exception):
                await self._announce(LineState.STOPPED, reason="simulator crash")
        finally:
            logger.info("line simulator loop exited")

    async def _tick(self) -> None:
        """Run one conveyor cycle; robust to any per-product failure."""

        t0 = time.monotonic()
        pid = await self._tracker.trigger()
        try:
            await self._process(pid)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("simulator tick failed for product %s", pid)
            # Never silently lose a product: force it to FAULT (raises an alarm).
            with contextlib.suppress(Exception):
                await self._tracker.force_fault(pid, reason="simulator_error")
        finally:
            self._ticks += 1
            self._record_cycle(t0)

    async def _process(self, pid: str) -> None:
        # Virtual camera: on camera loss no frame is produced at all.
        if CAMERA_LOSS in self._faults:
            self._camera_losses += 1
            self._last_label = None
            await self._tracker.force_fault(pid, reason=CAMERA_LOSS)
            return

        frame = await self._source.next_image()
        self._last_label = frame.label

        if REJECT_FAILURE in self._faults:
            await self._inspect_reject_prone(pid, frame.data)
        else:
            await self._orchestrator.inspect(pid, frame.data)

    async def _inspect_reject_prone(self, pid: str, image: bytes) -> None:
        """Drive ``capture → infer → decide`` and jam the reject actuator.

        Mirrors :meth:`Orchestrator.inspect` but, on a REJECT, commands the
        reject and then withholds the confirmation, so the lifecycle watchdog
        forces the product to FAULT with a CRITICAL alarm.
        """

        await self._tracker.mark_captured(pid)
        try:
            response = await asyncio.wait_for(
                self._inference.infer(image), timeout=self._inference_timeout_s
            )
        except (InferenceUnavailable, TimeoutError) as exc:
            reason = "inference_timeout" if isinstance(exc, TimeoutError) else "inference_error"
            await self._tracker.mark_inference_failed(pid, reason=reason, error=str(exc))
            await self._tracker.force_fault(pid, reason=reason)
            return

        await self._tracker.mark_inferred(
            pid,
            score=response.score,
            model_version=response.model_version,
            latency_ms=response.latency_ms,
        )
        active = await self._recipes.get_active()
        if active is None:
            await self._tracker.force_fault(pid, reason="no_active_recipe")
            return

        decision = decide(
            InferenceResult(
                score=response.score,
                model_version=response.model_version,
                latency_ms=response.latency_ms,
            ),
            recipe_to_params(active),
        )
        await self._tracker.mark_decided(
            pid,
            outcome=decision.disposition.value,
            reason=decision.reason.value,
            score=decision.score,
            recipe_id=int(active["id"]),
        )

        if decision.disposition is Disposition.PASS:
            await self._tracker.finalize_pass(
                pid, reason=decision.reason.value, score=decision.score
            )
        elif decision.disposition is Disposition.REJECT:
            # Reject actuator jammed: command the reject, but the confirm never
            # comes. The product is left in DECIDED for the watchdog to FAULT.
            await self._bus.publish(RejectCommanded(payload=RejectCommandedPayload(product_id=pid)))
            await asyncio.sleep(self._reject_actuation_s)
            self._reject_failures += 1
        else:  # FAULT from an invalid score
            await self._tracker.force_fault(pid, reason=decision.reason.value)

    def _record_cycle(self, t0: float) -> None:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._last_cycle_ms = elapsed_ms
        self._cycle_total_ms += elapsed_ms
        self._cycle_samples += 1


__all__ = [
    "CAMERA_LOSS",
    "KNOWN_FAULTS",
    "REJECT_FAILURE",
    "LineSimulator",
    "LineState",
]
