"""Product tracker: the in-flight state machine + stuck-product watchdog.

The tracker owns every product's current state, drives legal transitions,
publishes the corresponding lifecycle event on each transition, and guarantees
that no product is ever silently lost: reconciliation counters are maintained
such that ``lost`` is always 0.

The watchdog forces a FAULT (reason ``lifecycle_timeout``) on any product that
sits in a non-terminal state longer than the configured timeout, so a wedged
inference call or dropped frame can never leave a product in limbo.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

from ..events.bus import EventBus
from ..events.schemas import (
    DecisionMade,
    DecisionMadePayload,
    FrameCaptured,
    FrameCapturedPayload,
    InferenceCompleted,
    InferenceCompletedPayload,
    InferenceFailed,
    InferenceFailedPayload,
    ProductFinalized,
    ProductFinalizedPayload,
    RejectCommanded,
    RejectCommandedPayload,
    RejectConfirmed,
    RejectConfirmedPayload,
    TriggerFired,
    TriggerFiredPayload,
)
from .states import ProductState, check_transition

logger = logging.getLogger(__name__)


@dataclass
class ProductRecord:
    """In-memory tracking record for a single in-flight product."""

    product_id: str
    state: ProductState
    trigger_ts: float
    recipe_id: int | None
    last_change_mono: float
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Reconciliation:
    """Zero-silent-loss counters for the dashboard."""

    triggered: int
    in_flight: int
    passed: int
    rejected: int
    fault: int

    @property
    def terminal(self) -> int:
        return self.passed + self.rejected + self.fault

    @property
    def lost(self) -> int:
        """Products triggered but neither in-flight nor terminal — always 0."""

        return self.triggered - self.in_flight - self.terminal

    def as_dict(self) -> dict[str, int]:
        return {
            "triggered": self.triggered,
            "in_flight": self.in_flight,
            "pass": self.passed,
            "reject": self.rejected,
            "fault": self.fault,
            "terminal": self.terminal,
            "lost": self.lost,
        }


class ProductTracker:
    """Deterministic per-product state machine with a stuck-product watchdog."""

    def __init__(
        self,
        bus: EventBus,
        *,
        lifecycle_timeout_s: float = 10.0,
        watchdog_interval_s: float = 1.0,
    ) -> None:
        self._bus = bus
        self._timeout_s = lifecycle_timeout_s
        self._interval_s = watchdog_interval_s
        self._active: dict[str, ProductRecord] = {}
        self._triggered_total = 0
        self._passed = 0
        self._rejected = 0
        self._fault = 0
        self._watchdog_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # ---- watchdog -----------------------------------------------------
    def start_watchdog(self) -> None:
        """Launch the background watchdog task."""

        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="lifecycle-watchdog"
            )

    async def stop_watchdog(self) -> None:
        """Cancel and await the watchdog task."""

        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.sweep_stuck()
            except Exception:
                logger.exception("watchdog sweep failed")

    async def sweep_stuck(self) -> list[str]:
        """Force FAULT on products stuck past the timeout. Returns their ids."""

        now = time.monotonic()
        stuck: list[str] = []
        async with self._lock:
            for record in list(self._active.values()):
                if now - record.last_change_mono > self._timeout_s:
                    stuck.append(record.product_id)
        for product_id in stuck:
            await self.force_fault(product_id, reason="lifecycle_timeout")
        return stuck

    # ---- transitions --------------------------------------------------
    async def trigger(self, product_id: str | None = None, recipe_id: int | None = None) -> str:
        """Create a new product in TRIGGERED and publish ``TriggerFired``."""

        pid = product_id or uuid4().hex
        now_mono = time.monotonic()
        trigger_ts = time.time()
        async with self._lock:
            if pid in self._active:
                raise ValueError(f"product {pid} already in flight")
            self._active[pid] = ProductRecord(
                product_id=pid,
                state=ProductState.TRIGGERED,
                trigger_ts=trigger_ts,
                recipe_id=recipe_id,
                last_change_mono=now_mono,
            )
            self._triggered_total += 1
        await self._bus.publish(
            TriggerFired(
                payload=TriggerFiredPayload(
                    product_id=pid, trigger_ts=trigger_ts, recipe_id=recipe_id
                )
            )
        )
        return pid

    async def mark_captured(
        self, product_id: str, *, width: int | None = None, height: int | None = None
    ) -> None:
        """Transition TRIGGERED → CAPTURED and publish ``FrameCaptured``."""

        await self._advance(product_id, ProductState.CAPTURED)
        await self._bus.publish(
            FrameCaptured(
                payload=FrameCapturedPayload(product_id=product_id, width=width, height=height)
            )
        )

    async def mark_inferred(
        self, product_id: str, *, score: float, model_version: str, latency_ms: float
    ) -> None:
        """Transition CAPTURED → INFERRED and publish ``InferenceCompleted``."""

        await self._advance(product_id, ProductState.INFERRED, timing=("inference_ms", latency_ms))
        await self._bus.publish(
            InferenceCompleted(
                payload=InferenceCompletedPayload(
                    product_id=product_id,
                    score=score,
                    model_version=model_version,
                    latency_ms=latency_ms,
                )
            )
        )

    async def mark_inference_failed(
        self, product_id: str, *, reason: str, error: str | None = None
    ) -> None:
        """Publish ``InferenceFailed`` (no state change; caller then FAULTs)."""

        await self._bus.publish(
            InferenceFailed(
                payload=InferenceFailedPayload(product_id=product_id, reason=reason, error=error)
            )
        )

    async def mark_decided(
        self,
        product_id: str,
        *,
        outcome: str,
        reason: str,
        score: float | None,
        recipe_id: int | None = None,
    ) -> None:
        """Transition INFERRED → DECIDED and publish ``DecisionMade``."""

        await self._advance(product_id, ProductState.DECIDED)
        await self._bus.publish(
            DecisionMade(
                payload=DecisionMadePayload(
                    product_id=product_id,
                    outcome=outcome,
                    reason=reason,
                    score=score,
                    recipe_id=recipe_id,
                )
            )
        )

    async def finalize_pass(
        self, product_id: str, *, reason: str, score: float | None = None
    ) -> None:
        """Transition DECIDED → PASS (terminal) and publish ``ProductFinalized``."""

        await self._finalize(product_id, ProductState.PASS, reason=reason, score=score)

    async def finalize_reject(
        self, product_id: str, *, reason: str, score: float | None = None
    ) -> None:
        """Transition DECIDED → REJECT, command+confirm the reject, finalize."""

        await self._bus.publish(
            RejectCommanded(payload=RejectCommandedPayload(product_id=product_id))
        )
        await self._finalize(product_id, ProductState.REJECT, reason=reason, score=score)
        await self._bus.publish(
            RejectConfirmed(payload=RejectConfirmedPayload(product_id=product_id))
        )

    async def force_fault(self, product_id: str, *, reason: str) -> None:
        """Force any non-terminal product to FAULT (terminal). Idempotent."""

        async with self._lock:
            record = self._active.get(product_id)
            if record is None:
                return  # already terminal / unknown — nothing to fault
        await self._finalize(product_id, ProductState.FAULT, reason=reason, score=None)

    # ---- internals ----------------------------------------------------
    async def _advance(
        self,
        product_id: str,
        to: ProductState,
        *,
        timing: tuple[str, float] | None = None,
    ) -> None:
        async with self._lock:
            record = self._require(product_id)
            check_transition(record.state, to)
            record.state = to
            record.last_change_mono = time.monotonic()
            if timing is not None:
                record.timings[timing[0]] = timing[1]

    async def _finalize(
        self,
        product_id: str,
        to: ProductState,
        *,
        reason: str,
        score: float | None,
    ) -> None:
        async with self._lock:
            record = self._require(product_id)
            check_transition(record.state, to)
            record.state = to
            timings = dict(record.timings)
            recipe_id = record.recipe_id
            del self._active[product_id]
            if to is ProductState.PASS:
                self._passed += 1
            elif to is ProductState.REJECT:
                self._rejected += 1
            else:
                self._fault += 1
        await self._bus.publish(
            ProductFinalized(
                payload=ProductFinalizedPayload(
                    product_id=product_id,
                    outcome=to.value,
                    reason=reason,
                    anomaly_score=score,
                    recipe_id=recipe_id,
                    timings=timings,
                )
            )
        )

    def _require(self, product_id: str) -> ProductRecord:
        record = self._active.get(product_id)
        if record is None:
            raise KeyError(f"unknown or already-terminal product: {product_id}")
        return record

    # ---- introspection ------------------------------------------------
    def state_of(self, product_id: str) -> ProductState | None:
        """Return the live state of an in-flight product, or ``None``."""

        record = self._active.get(product_id)
        return record.state if record else None

    def in_flight_ids(self) -> list[str]:
        return list(self._active.keys())

    def reconcile(self) -> Reconciliation:
        """Return the current zero-silent-loss reconciliation counters."""

        in_flight = len(self._active)
        return Reconciliation(
            triggered=self._triggered_total,
            in_flight=in_flight,
            passed=self._passed,
            rejected=self._rejected,
            fault=self._fault,
        )


__all__ = ["ProductRecord", "ProductTracker", "Reconciliation"]
