"""Deterministic product lifecycle: state machine, tracker, watchdog."""

from .states import (
    ALLOWED,
    TERMINAL_STATES,
    IllegalTransition,
    ProductState,
    check_transition,
    is_terminal,
)
from .tracker import ProductRecord, ProductTracker, Reconciliation

__all__ = [
    "ALLOWED",
    "TERMINAL_STATES",
    "IllegalTransition",
    "ProductRecord",
    "ProductState",
    "ProductTracker",
    "Reconciliation",
    "check_transition",
    "is_terminal",
]
