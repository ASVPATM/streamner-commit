"""Small structural interfaces shared by model backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

type EntityRecord = dict[str, Any]
type EntitySnapshot = list[EntityRecord]


class StreamingNERBackend(Protocol):
    """Public operations required by a streaming trace runner."""

    def append(
        self,
        chunk: str,
        labels: Sequence[str],
        session_id: str,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        recompute: bool = False,
    ) -> EntitySnapshot:
        """Append one chunk and return the complete current snapshot."""
        ...

    def infer_full(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
    ) -> EntitySnapshot:
        """Run a separate cold full-text inference call."""
        ...

    def clear_session(self, session_id: str) -> None:
        """Release all state owned by one streaming session."""
        ...


__all__ = ["EntityRecord", "EntitySnapshot", "StreamingNERBackend"]
