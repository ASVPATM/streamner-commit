"""Public StreamingSpan trace orchestration.

This module deliberately depends only on the structural model protocol below.
Importing it in the main MLX environment never imports GLiNER or PyTorch.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from streamner_commit.chunking import chunk_text_by_words, count_model_words
from streamner_commit.types import ColdFullResult, PublicEntity, SnapshotStep


class PublicInferenceModel(Protocol):
    """The small portion of the public GLiNER API needed for a trace."""

    def inference(
        self,
        texts: str | list[str],
        labels: list[str],
        **kwargs: Any,
    ) -> object:
        """Run stateless or session inference."""

    def clear_session(self, session_id: str | list[str]) -> None:
        """Clear one or more streaming sessions."""


@dataclass(frozen=True, slots=True)
class PublicTraceResult:
    """All public observations for one example and its independent cold run."""

    run_id: str
    example_id: str
    session_id: str
    full_text: str
    labels: tuple[str, ...]
    words_per_chunk: int
    threshold: float
    snapshots: tuple[SnapshotStep, ...]
    cold_full: ColdFullResult
    cold_elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace document."""
        return {
            "run_id": self.run_id,
            "example_id": self.example_id,
            "session_id": self.session_id,
            "full_text": self.full_text,
            "labels": list(self.labels),
            "words_per_chunk": self.words_per_chunk,
            "threshold": self.threshold,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "cold_full": self.cold_full.to_dict(),
            "cold_elapsed_ms": self.cold_elapsed_ms,
        }


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _labels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("labels must be a sequence of strings")
    labels = tuple(values)
    if not labels:
        raise ValueError("labels must not be empty")
    for label in labels:
        _identifier("label", label)
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique and ordered")
    return labels


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("threshold must be a real number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    return threshold


def _single_model_output(output: object) -> Sequence[object]:
    if isinstance(output, str | bytes) or not isinstance(output, Sequence):
        raise TypeError("model inference must return a batch sequence")
    if len(output) != 1:
        raise ValueError("single-example inference must return exactly one batch row")
    entities = output[0]
    if isinstance(entities, str | bytes) or not isinstance(entities, Sequence):
        raise TypeError("model inference batch row must be an entity sequence")
    return entities


def _mapping_field(
    entity: Mapping[str, object],
    primary: str,
    alternate: str | None = None,
) -> object:
    if primary in entity:
        return entity[primary]
    if alternate is not None and alternate in entity:
        return entity[alternate]
    names = primary if alternate is None else f"{primary}/{alternate}"
    raise ValueError(f"public entity is missing {names}")


def _public_entity(value: object) -> PublicEntity:
    if isinstance(value, PublicEntity):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("public entities must be PublicEntity values or mappings")
    return PublicEntity(
        start_char=_mapping_field(value, "start_char", "start"),  # type: ignore[arg-type]
        end_char=_mapping_field(value, "end_char", "end"),  # type: ignore[arg-type]
        label=_mapping_field(value, "label"),  # type: ignore[arg-type]
        text=_mapping_field(value, "text"),  # type: ignore[arg-type]
        score=_mapping_field(value, "score"),  # type: ignore[arg-type]
    )


def _decode_public_output(output: object, visible_text: str) -> tuple[PublicEntity, ...]:
    entities = tuple(_public_entity(value) for value in _single_model_output(output))
    for entity in entities:
        if entity.end_char > len(visible_text):
            raise ValueError("public entity extends beyond accumulated text")
        if visible_text[entity.start_char : entity.end_char] != entity.text:
            raise ValueError("public entity text does not match accumulated-text offsets")
    return entities


def run_public_trace(
    model: PublicInferenceModel,
    *,
    text: str,
    labels: Sequence[str],
    example_id: str,
    run_id: str | None = None,
    words_per_chunk: int = 1,
    threshold: float = 0.5,
    session_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PublicTraceResult:
    """Trace public session snapshots, clear the session, then run cold inference.

    Each non-whitespace chunk is one append. Chunks are passed to the model
    verbatim, and every snapshot is validated against the exact reconstructed
    prefix. The independent full-text call happens only after session cleanup.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    labels_tuple = _labels(labels)
    example_id = _identifier("example_id", example_id)
    run_id = _identifier("run_id", run_id or f"public-{uuid.uuid4().hex}")
    session_id = _identifier(
        "session_id",
        session_id or f"{run_id}:{example_id}:{uuid.uuid4().hex}",
    )
    threshold = _threshold(threshold)

    chunks = chunk_text_by_words(text, words_per_chunk)
    if "".join(chunks) != text:
        raise AssertionError("chunking did not reconstruct the source text")

    snapshots: list[SnapshotStep] = []
    accumulated_text = ""
    source_cursor = 0
    append_step = 0
    try:
        for chunk in chunks:
            next_cursor = source_cursor + len(chunk)
            if text[source_cursor:next_cursor] != chunk:
                raise AssertionError("chunk is not the next exact source prefix")
            source_cursor = next_cursor

            # Whitespace-only calls do not advance a GLiNER session. The primary
            # chunker attaches whitespace to word-bearing chunks, except when the
            # whole input itself is blank.
            if not chunk.strip():
                continue

            append_step += 1
            accumulated_text += chunk
            if accumulated_text != text[:source_cursor]:
                raise AssertionError("accumulated session text differs from the source prefix")

            started = clock()
            output = model.inference(
                [chunk],
                list(labels_tuple),
                threshold=threshold,
                session_id=[session_id],
            )
            elapsed_ms = (clock() - started) * 1000.0
            public_entities = _decode_public_output(output, accumulated_text)
            snapshots.append(
                SnapshotStep(
                    run_id=run_id,
                    example_id=example_id,
                    step=append_step,
                    chunk=chunk,
                    accumulated_text=accumulated_text,
                    visible_char_count=len(accumulated_text),
                    visible_word_count=count_model_words(accumulated_text),
                    elapsed_ms=elapsed_ms,
                    public_entities=public_entities,
                )
            )
        if source_cursor != len(text):
            raise AssertionError("not every source character was consumed")
    finally:
        model.clear_session(session_id)

    cold_started = clock()
    cold_output = model.inference(
        [text],
        list(labels_tuple),
        threshold=threshold,
    )
    cold_elapsed_ms = (clock() - cold_started) * 1000.0
    cold_entities = _decode_public_output(cold_output, text)
    cold_full = ColdFullResult(
        example_id=example_id,
        full_text=text,
        public_entities=cold_entities,
        raw_final_span_state={},
    )
    return PublicTraceResult(
        run_id=run_id,
        example_id=example_id,
        session_id=session_id,
        full_text=text,
        labels=labels_tuple,
        words_per_chunk=words_per_chunk,
        threshold=threshold,
        snapshots=tuple(snapshots),
        cold_full=cold_full,
        cold_elapsed_ms=cold_elapsed_ms,
    )


def format_public_trace(trace: PublicTraceResult) -> str:
    """Format one trace as plain, human-readable terminal output."""
    lines = [f"== {trace.example_id} =="]
    for snapshot in trace.snapshots:
        lines.extend(
            [
                f"[step {snapshot.step:02d}] + "
                f"{json.dumps(snapshot.chunk, ensure_ascii=False)} "
                f"({snapshot.elapsed_ms:.2f} ms)",
                f"text: {json.dumps(snapshot.accumulated_text, ensure_ascii=False)}",
            ]
        )
        if snapshot.public_entities:
            for entity in snapshot.public_entities:
                lines.append(
                    f"  {entity.label:<20} {entity.score:.4f}  "
                    f"[{entity.start_char}:{entity.end_char}]  "
                    f"{json.dumps(entity.text, ensure_ascii=False)}"
                )
        else:
            lines.append("  (no entities)")
        lines.append("")

    lines.append(f"[cold full] ({trace.cold_elapsed_ms:.2f} ms)")
    if trace.cold_full.public_entities:
        for entity in trace.cold_full.public_entities:
            lines.append(
                f"  {entity.label:<20} {entity.score:.4f}  "
                f"[{entity.start_char}:{entity.end_char}]  "
                f"{json.dumps(entity.text, ensure_ascii=False)}"
            )
    else:
        lines.append("  (no entities)")
    return "\n".join(lines)


__all__ = [
    "PublicInferenceModel",
    "PublicTraceResult",
    "format_public_trace",
    "run_public_trace",
]
