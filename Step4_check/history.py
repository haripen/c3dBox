"""History utilities for manual selection edits.

This module implements a lightweight command-pattern history for
undo/redo of cycle selection changes inside the Cycle Selection Tool.

Classes
-------
- SelectionEdit: a concrete command that applies a batch of selection
  state changes (e.g., toggling `manually_selected` flags) against a
  provided mutable mapping-like store.
- History: an undo/redo stack manager with push/undo/redo operations.

Notes
-----
- Keyboard shortcuts (wired in main.py):
  * Undo: Ctrl+Z
  * Redo: Ctrl+Shift+Z
- The store provided to `SelectionEdit` can be any MutableMapping
  from cycle_ref_id -> value (usually 0/1), e.g. a dict maintained by
  the UI/model layer. Optional `on_apply` callback lets the UI trigger
  redraws or counters after each set.

Example
-------
>>> store = {}
>>> edit = SelectionEdit(store, [("c1", 1, 0), ("c2", 1, 0)])
>>> hist = History()
>>> hist.push(edit)  # applies .do()
>>> store
{'c1': 0, 'c2': 0}
>>> hist.undo()
True
>>> store
{'c1': 1, 'c2': 1}
>>> hist.redo()
True
>>> store
{'c1': 0, 'c2': 0}

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, List, MutableMapping, Optional, Protocol, Sequence, Tuple
import logging

logger = logging.getLogger(__name__)


# ------------------------------
# Command protocol
# ------------------------------
class Command(Protocol):
    """Minimal command protocol for history management."""

    def do(self) -> None:  # apply
        ...

    def undo(self) -> None:  # rollback
        ...

    @property
    def description(self) -> str:  # short label for UI (optional)
        ...


# ------------------------------
# SelectionEdit command
# ------------------------------
@dataclass(frozen=True)
class SelectionChange:
    """A single selection change unit.

    Attributes
    ----------
    cycle_ref_id: Hashable
        Identifier for the cycle to modify (e.g., a string key).
    old_val: Any
        Previous value (usually 0/1). Used for undo.
    new_val: Any
        New value to apply on do().
    """

    cycle_ref_id: Hashable
    old_val: Any
    new_val: Any


class SelectionEdit(Command):
    """Batch selection edit with undo/redo.

    Parameters
    ----------
    store : MutableMapping[Hashable, Any]
        Mapping of cycle_ref_id -> selection value. Typically managed by the
        controller/model layer of the app.
    changes : Sequence[Tuple[Hashable, Any, Any]] | Sequence[SelectionChange]
        List of triplets (cycle_ref_id, old_val, new_val) or SelectionChange
        objects representing the atomic changes in this edit.
    description : str, optional
        Short label for UI (e.g., "Rectangle select", "Manual toggle").
    on_apply : Callable[[Hashable, Any], None], optional
        Callback invoked after each individual assignment (id, value) so the
        UI can refresh a single line/axes or counters efficiently.

    Behavior
    --------
    - do(): sets each cycle to its `new_val` (in listed order)
    - undo(): restores each cycle to its `old_val` (reverse order)
    - The store is treated generically; no assumptions beyond __setitem__.
    - If a key is missing, it is created on do() and undo() alike.
    """

    __slots__ = ("_store", "_changes", "_description", "_on_apply")

    def __init__(
        self,
        store: MutableMapping[Hashable, Any],
        changes: Sequence[Tuple[Hashable, Any, Any]] | Sequence[SelectionChange],
        description: str = "Selection edit",
        on_apply: Optional[Callable[[Hashable, Any], None]] = None,
    ) -> None:
        self._store = store
        # Normalize input to SelectionChange list
        self._changes: List[SelectionChange] = [
            ch if isinstance(ch, SelectionChange) else SelectionChange(*ch)  # type: ignore[arg-type]
            for ch in changes
        ]
        self._description = description
        self._on_apply = on_apply

    # --- Command API ---
    def do(self) -> None:
        for ch in self._changes:
            self._apply(ch.cycle_ref_id, ch.new_val)
        logger.debug("SelectionEdit.do applied %d changes", len(self._changes))

    def undo(self) -> None:
        # reverse order to better respect dependencies (if any)
        for ch in reversed(self._changes):
            self._apply(ch.cycle_ref_id, ch.old_val)
        logger.debug("SelectionEdit.undo reverted %d changes", len(self._changes))

    @property
    def description(self) -> str:
        return self._description

    # --- Helpers ---
    def _apply(self, key: Hashable, value: Any) -> None:
        self._store[key] = value
        if self._on_apply is not None:
            try:
                self._on_apply(key, value)
            except Exception:  # guard UI callbacks
                logger.exception("on_apply callback failed for key=%r", key)

    @property
    def changes(self) -> List[SelectionChange]:
        return list(self._changes)

    @property
    def changed_ids(self) -> List[Hashable]:
        return [ch.cycle_ref_id for ch in self._changes]


# ------------------------------
# History manager
# ------------------------------
class History:
    """Undo/redo history stack.

    A simple ring-buffer-like stack managed by an index cursor. New pushes
    truncate any redo branch. Undo decrements the cursor and calls the
    command's undo(); redo applies the command at the cursor and increments.

    Parameters
    ----------
    max_depth : int, optional
        Maximum number of commands to retain. When exceeded, the oldest
        entries are dropped while keeping the current state consistent.
    on_change : Callable[[], None], optional
        Called after every push/undo/redo to let the UI update button states.

    Notes
    -----
    - Designed to be thread-agnostic; UI code should marshal calls into the
      Qt main thread as needed.
    - Commands are expected to be side-effect free beyond the intended
      state mutations.
    """

    __slots__ = ("_stack", "_index", "_max", "_on_change")

    def __init__(self, max_depth: int = 500, on_change: Optional[Callable[[], None]] = None) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self._stack: List[Command] = []
        self._index: int = 0  # next position; 0 <= index <= len(stack)
        self._max = max_depth
        self._on_change = on_change

    # Core API --------------------------------------------------------------
    def push(self, cmd: Command) -> None:
        """Execute and push a command onto the history.

        Any pending redo branch is discarded. If the stack grows beyond
        `max_depth`, the oldest item is dropped and the index is shifted
        accordingly.
        """
        # Discard redo branch
        if self._index < len(self._stack):
            del self._stack[self._index :]

        # Execute command now
        cmd.do()

        # Enforce capacity
        if len(self._stack) == self._max:
            # drop oldest, shift index left by one
            del self._stack[0]
            self._index -= 1
            if self._index < 0:
                self._index = 0

        self._stack.append(cmd)
        self._index = len(self._stack)
        self._notify()
        logger.debug("History.push: size=%d index=%d", len(self._stack), self._index)

    def undo(self) -> bool:
        """Undo the last command. Returns True if something was undone."""
        if not self.can_undo:
            return False
        self._index -= 1
        cmd = self._stack[self._index]
        cmd.undo()
        self._notify()
        logger.debug("History.undo: size=%d index=%d", len(self._stack), self._index)
        return True

    def redo(self) -> bool:
        """Redo the next command. Returns True if something was redone."""
        if not self.can_redo:
            return False
        cmd = self._stack[self._index]
        cmd.do()
        self._index += 1
        self._notify()
        logger.debug("History.redo: size=%d index=%d", len(self._stack), self._index)
        return True

    # Utilities -------------------------------------------------------------
    def clear(self) -> None:
        self._stack.clear()
        self._index = 0
        self._notify()

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._stack)

    @property
    def size(self) -> int:
        return len(self._stack)

    @property
    def index(self) -> int:
        """The cursor position (0..size). Next redo is at `index`.

        After push, `index == size`. After one undo, `index == size-1`.
        """
        return self._index

    @property
    def next_undo_label(self) -> Optional[str]:
        if not self.can_undo:
            return None
        return getattr(self._stack[self._index - 1], "description", None)

    @property
    def next_redo_label(self) -> Optional[str]:
        if not self.can_redo:
            return None
        return getattr(self._stack[self._index], "description", None)

    # Internal --------------------------------------------------------------
    def _notify(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                logger.exception("History on_change callback failed")


# ------------------------------
# Smoke test
# ------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    # Minimal round-trip demonstration
    store: MutableMapping[Hashable, Any] = {
        "left/cycle1": 1,
        "left/cycle2": 1,
        "right/cycle1": 1,
    }

    def on_apply(key: Hashable, value: Any) -> None:
        logger.info("apply: %r -> %r", key, value)

    h = History(max_depth=5, on_change=lambda: logger.info("history changed: size=%d index=%d", h.size, h.index))

    # Rectangle de-select two lines
    cmd1 = SelectionEdit(
        store,
        [
            ("left/cycle1", store.get("left/cycle1", 1), 0),
            ("right/cycle1", store.get("right/cycle1", 1), 0),
        ],
        description="Rect deselect",
        on_apply=on_apply,
    )
    h.push(cmd1)

    # Manual re-select one line
    cmd2 = SelectionEdit(
        store,
        [("left/cycle1", store.get("left/cycle1", 0), 1)],
        description="Manual select",
        on_apply=on_apply,
    )
    h.push(cmd2)

    assert store["left/cycle1"] == 1 and store["right/cycle1"] == 0

    h.undo()  # undo manual select
    assert store["left/cycle1"] == 0

    h.undo()  # undo rect deselect
    assert store["left/cycle1"] == 1 and store["right/cycle1"] == 1

    h.redo()  # redo rect deselect
    assert store["left/cycle1"] == 0 and store["right/cycle1"] == 0

    h.redo()  # redo manual select
    assert store["left/cycle1"] == 1 and store["right/cycle1"] == 0

    print("Smoke test passed.")
