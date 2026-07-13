"""Product lifecycle states and the legal-transition table.

The state machine is deterministic and explicit:

``TRIGGERED → CAPTURED → INFERRED → DECIDED → PASS | REJECT``

and, from any non-terminal state, an error path to ``FAULT``. Any transition
not in :data:`ALLOWED` raises :class:`IllegalTransition`.
"""

from __future__ import annotations

from enum import Enum


class ProductState(str, Enum):
    """Every state a product can occupy."""

    TRIGGERED = "TRIGGERED"
    CAPTURED = "CAPTURED"
    INFERRED = "INFERRED"
    DECIDED = "DECIDED"
    PASS = "PASS"
    REJECT = "REJECT"
    FAULT = "FAULT"


#: Terminal states — a product in one of these is finished, forever.
TERMINAL_STATES: frozenset[ProductState] = frozenset(
    {ProductState.PASS, ProductState.REJECT, ProductState.FAULT}
)

#: Legal forward transitions. FAULT is reachable from every non-terminal state
#: and is added programmatically below.
ALLOWED: dict[ProductState, frozenset[ProductState]] = {
    ProductState.TRIGGERED: frozenset({ProductState.CAPTURED, ProductState.FAULT}),
    ProductState.CAPTURED: frozenset({ProductState.INFERRED, ProductState.FAULT}),
    ProductState.INFERRED: frozenset({ProductState.DECIDED, ProductState.FAULT}),
    ProductState.DECIDED: frozenset({ProductState.PASS, ProductState.REJECT, ProductState.FAULT}),
}


class IllegalTransition(Exception):
    """Raised when a state transition is not permitted by :data:`ALLOWED`."""

    def __init__(self, frm: ProductState, to: ProductState) -> None:
        super().__init__(f"illegal transition {frm.value} -> {to.value}")
        self.frm = frm
        self.to = to


def is_terminal(state: ProductState) -> bool:
    """Return whether ``state`` is a terminal state."""

    return state in TERMINAL_STATES


def check_transition(frm: ProductState, to: ProductState) -> None:
    """Raise :class:`IllegalTransition` if ``frm → to`` is not allowed."""

    if frm in TERMINAL_STATES or to not in ALLOWED.get(frm, frozenset()):
        raise IllegalTransition(frm, to)


__all__ = [
    "ALLOWED",
    "TERMINAL_STATES",
    "IllegalTransition",
    "ProductState",
    "check_transition",
    "is_terminal",
]
