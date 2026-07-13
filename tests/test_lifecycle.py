"""Lifecycle — terminal paths, illegal transitions, watchdog, reconciliation."""

from __future__ import annotations

import asyncio

import pytest

from visionqc.events.bus import EventBus
from visionqc.events.schemas import EventType
from visionqc.lifecycle.states import IllegalTransition, ProductState, is_terminal
from visionqc.lifecycle.tracker import ProductTracker


async def _run_pass(tracker: ProductTracker, pid: str) -> None:
    await tracker.trigger(pid)
    await tracker.mark_captured(pid)
    await tracker.mark_inferred(pid, score=0.1, model_version="m", latency_ms=1.0)
    await tracker.mark_decided(pid, outcome="PASS", reason="within_tolerance", score=0.1)
    await tracker.finalize_pass(pid, reason="within_tolerance", score=0.1)


async def test_pass_path_terminates(tracker: ProductTracker) -> None:
    await _run_pass(tracker, "p1")
    assert tracker.state_of("p1") is None  # removed from in-flight
    rec = tracker.reconcile()
    assert rec.passed == 1
    assert rec.in_flight == 0
    assert rec.lost == 0


async def test_reject_path_terminates(tracker: ProductTracker) -> None:
    await tracker.trigger("p1")
    await tracker.mark_captured("p1")
    await tracker.mark_inferred("p1", score=0.9, model_version="m", latency_ms=1.0)
    await tracker.mark_decided("p1", outcome="REJECT", reason="anomaly", score=0.9)
    await tracker.finalize_reject("p1", reason="anomaly", score=0.9)
    rec = tracker.reconcile()
    assert rec.rejected == 1
    assert rec.lost == 0


async def test_reject_emits_command_and_confirm() -> None:
    bus = EventBus()
    sub = bus.subscribe("cap")
    tracker = ProductTracker(bus)
    await tracker.trigger("p1")
    await tracker.mark_captured("p1")
    await tracker.mark_inferred("p1", score=0.9, model_version="m", latency_ms=1.0)
    await tracker.mark_decided("p1", outcome="REJECT", reason="anomaly", score=0.9)
    await tracker.finalize_reject("p1", reason="anomaly", score=0.9)

    types: list[EventType] = []
    while True:
        try:
            event = await asyncio.wait_for(sub.get(), timeout=0.05)
        except TimeoutError:
            break
        if event is None:
            break
        types.append(event.type)
    assert EventType.REJECT_COMMANDED in types
    assert EventType.REJECT_CONFIRMED in types
    # command precedes confirm
    assert types.index(EventType.REJECT_COMMANDED) < types.index(EventType.REJECT_CONFIRMED)


async def test_fault_from_any_nonterminal(tracker: ProductTracker) -> None:
    await tracker.trigger("p1")
    await tracker.force_fault("p1", reason="lifecycle_timeout")
    assert tracker.state_of("p1") is None
    rec = tracker.reconcile()
    assert rec.fault == 1
    assert rec.lost == 0


async def test_illegal_transition_raises(tracker: ProductTracker) -> None:
    await tracker.trigger("p1")
    # Cannot go straight to INFERRED without CAPTURED.
    with pytest.raises(IllegalTransition):
        await tracker.mark_inferred("p1", score=0.1, model_version="m", latency_ms=1.0)


async def test_finalize_unknown_product_raises(tracker: ProductTracker) -> None:
    with pytest.raises(KeyError):
        await tracker.mark_captured("does-not-exist")


async def test_force_fault_on_terminal_is_noop(tracker: ProductTracker) -> None:
    await _run_pass(tracker, "p1")
    # Already PASS/terminal — force_fault is a safe no-op.
    await tracker.force_fault("p1", reason="late")
    rec = tracker.reconcile()
    assert rec.passed == 1
    assert rec.fault == 0


async def test_watchdog_forces_fault() -> None:
    bus = EventBus()
    tracker = ProductTracker(bus, lifecycle_timeout_s=0.1, watchdog_interval_s=0.02)
    tracker.start_watchdog()
    try:
        await tracker.trigger("stuck")
        # Never advance it — the watchdog should FAULT it.
        for _ in range(50):
            if tracker.state_of("stuck") is None:
                break
            await asyncio.sleep(0.02)
    finally:
        await tracker.stop_watchdog()
    rec = tracker.reconcile()
    assert rec.fault == 1
    assert rec.lost == 0


async def test_reconciliation_lost_always_zero(tracker: ProductTracker) -> None:
    await _run_pass(tracker, "pass1")
    await tracker.trigger("reject1")
    await tracker.mark_captured("reject1")
    await tracker.mark_inferred("reject1", score=0.9, model_version="m", latency_ms=1.0)
    await tracker.mark_decided("reject1", outcome="REJECT", reason="a", score=0.9)
    await tracker.finalize_reject("reject1", reason="a", score=0.9)
    await tracker.trigger("fault1")
    await tracker.force_fault("fault1", reason="x")
    await tracker.trigger("inflight1")  # still in-flight

    rec = tracker.reconcile()
    assert rec.triggered == 4
    assert rec.passed == 1
    assert rec.rejected == 1
    assert rec.fault == 1
    assert rec.in_flight == 1
    assert rec.lost == 0


def test_terminal_state_helper() -> None:
    assert is_terminal(ProductState.PASS)
    assert is_terminal(ProductState.REJECT)
    assert is_terminal(ProductState.FAULT)
    assert not is_terminal(ProductState.TRIGGERED)
