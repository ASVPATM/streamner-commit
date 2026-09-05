"""One-pass MLX trace generation for offline commitment-policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from streamner_commit.chunking import chunk_text_by_words
from streamner_commit.mlx.decoder import sigmoid
from streamner_commit.mlx.preprocessing import normalize_labels
from streamner_commit.streaming.replay import assert_span_states_close, replay_span_updates
from streamner_commit.types import (
    ColdFullResult,
    GoldEntity,
    SnapshotStep,
    SpanBoundary,
    SpanScoreUpdate,
)

if TYPE_CHECKING:
    from streamner_commit.backends.mlx_streaming import (
        MLXStreamingBackend,
        StreamingAppendResult,
        StreamingSpanUpdate,
        StreamingStateMetadata,
    )


TRACE_CHUNK_WORDS = (1, 2, 4, 8)


class TraceGenerationError(RuntimeError):
    """A live MLX result cannot form a trustworthy replayable trace."""


def _chunk_condition(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("chunk_words must be an integer")
    if value not in TRACE_CHUNK_WORDS:
        raise ValueError(f"chunk_words must be one of {TRACE_CHUNK_WORDS}")
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonblank")
    return value


def _freeze_json(value: object, *, path: str) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[raw_key] = _freeze_json(item, path=f"{path}.{raw_key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TraceInputExample:
    """One complete trace input with exact text, labels, metadata, and gold."""

    example_id: str
    text: str
    labels: tuple[str, ...]
    gold_entities: tuple[GoldEntity, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _identifier(self.example_id, name="example_id")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        labels = normalize_labels(self.labels)
        object.__setattr__(self, "labels", labels)

        if (
            isinstance(self.gold_entities, str | bytes)
            or not isinstance(self.gold_entities, Sequence)
        ):
            raise TypeError("gold_entities must be a sequence of GoldEntity values")
        gold = tuple(self.gold_entities)
        for index, entity in enumerate(gold):
            if not isinstance(entity, GoldEntity):
                raise TypeError(f"gold_entities[{index}] must be a GoldEntity")
            if entity.example_id != self.example_id:
                raise ValueError(f"gold_entities[{index}] belongs to a different example")
            if entity.end_char > len(self.text):
                raise ValueError(f"gold_entities[{index}] extends beyond source text")
            if self.text[entity.start_char : entity.end_char] != entity.text:
                raise ValueError(f"gold_entities[{index}] offsets do not preserve the source slice")
        object.__setattr__(self, "gold_entities", gold)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        frozen_metadata = _freeze_json(self.metadata, path="metadata")
        assert isinstance(frozen_metadata, Mapping)
        object.__setattr__(self, "metadata", frozen_metadata)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe example data for a persistence adapter."""

        return {
            "example_id": self.example_id,
            "text": self.text,
            "labels": list(self.labels),
            "gold_entities": [entity.to_dict() for entity in self.gold_entities],
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GeneratedExampleTrace:
    """A replayable warm event log plus an independent cold silver target."""

    run_id: str
    chunk_words: int
    example: TraceInputExample
    snapshots: tuple[SnapshotStep, ...]
    span_updates: tuple[SpanScoreUpdate, ...]
    cold_full: ColdFullResult
    final_state_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.run_id, name="run_id")
        _chunk_condition(self.chunk_words)
        if not isinstance(self.example, TraceInputExample):
            raise TypeError("example must be a TraceInputExample")
        snapshots = tuple(self.snapshots)
        updates = tuple(self.span_updates)
        if not all(isinstance(snapshot, SnapshotStep) for snapshot in snapshots):
            raise TypeError("snapshots must contain only SnapshotStep values")
        if not all(isinstance(update, SpanScoreUpdate) for update in updates):
            raise TypeError("span_updates must contain only SpanScoreUpdate values")
        if any(
            snapshot.run_id != self.run_id
            or snapshot.example_id != self.example.example_id
            for snapshot in snapshots
        ):
            raise ValueError("snapshot identity differs from its generated trace")
        if any(
            update.run_id != self.run_id or update.example_id != self.example.example_id
            for update in updates
        ):
            raise ValueError("span-update identity differs from its generated trace")
        if not snapshots or tuple(snapshot.step for snapshot in snapshots) != tuple(
            range(1, len(snapshots) + 1)
        ):
            raise ValueError("snapshots must contain consecutive one-based append steps")
        if snapshots[-1].accumulated_text != self.example.text:
            raise ValueError("final warm snapshot does not contain the complete example text")
        snapshots_by_step = {snapshot.step: snapshot for snapshot in snapshots}
        for update in updates:
            snapshot = snapshots_by_step.get(update.step)
            if snapshot is None:
                raise ValueError("span update refers to a step without a public snapshot")
            if (
                update.chunk != snapshot.chunk
                or update.visible_char_count != snapshot.visible_char_count
                or update.visible_word_count != snapshot.visible_word_count
            ):
                raise ValueError("span update metadata differs from its public snapshot")
        if not isinstance(self.cold_full, ColdFullResult):
            raise TypeError("cold_full must be a ColdFullResult")
        if (
            self.cold_full.example_id != self.example.example_id
            or self.cold_full.full_text != self.example.text
        ):
            raise ValueError("cold result identity/text differs from its generated trace")
        if (
            not isinstance(self.final_state_sha256, str)
            or len(self.final_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.final_state_sha256)
        ):
            raise ValueError("final_state_sha256 must be a lowercase SHA-256 hex digest")
        cold_widths = {len(vector) for vector in self.cold_full.raw_final_span_state.values()}
        if cold_widths and cold_widths != {len(self.example.labels)}:
            raise ValueError("cold raw score width differs from the example label count")
        if span_state_sha256(replay_span_updates(updates)) != self.final_state_sha256:
            raise ValueError("final_state_sha256 differs from replayed warm score state")
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "span_updates", updates)

    @property
    def example_id(self) -> str:
        """Return the source example identifier."""

        return self.example.example_id

    @property
    def text(self) -> str:
        """Return the exact full source text."""

        return self.example.text

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the fixed ordered label vocabulary."""

        return self.example.labels

    @property
    def gold_entities(self) -> tuple[GoldEntity, ...]:
        """Return the unmodified gold annotations."""

        return self.example.gold_entities

    @property
    def metadata(self) -> Mapping[str, object]:
        """Return immutable source metadata."""

        return self.example.metadata

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe nested debug representation."""

        return {
            "run_id": self.run_id,
            "chunk_words": self.chunk_words,
            "example": self.example.to_dict(),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "span_updates": [update.to_dict() for update in self.span_updates],
            "cold_full": self.cold_full.to_dict(),
            "final_state_sha256": self.final_state_sha256,
        }


def span_state_sha256(state: Mapping[SpanBoundary, Sequence[float]]) -> str:
    """Hash one raw state in a deterministic boundary/vector representation."""

    rows: list[dict[str, object]] = []
    for boundary, raw_vector in sorted(state.items()):
        if not isinstance(boundary, SpanBoundary):
            raise TypeError("state keys must be SpanBoundary values")
        vector = tuple(float(value) for value in raw_vector)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("state vectors must be nonempty and finite")
        rows.append(
            {
                "start_word": boundary.start_word,
                "end_word": boundary.end_word,
                "logits": vector,
            }
        )
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_word_coordinates(
    state: StreamingStateMetadata,
    *,
    expected_text: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if state.accumulated_text != expected_text:
        raise TraceGenerationError("backend accumulated text differs from the exact chunk prefix")
    starts = tuple(state.word_char_starts)
    ends = tuple(state.word_char_ends)
    tokens = tuple(state.word_tokens)
    if not (len(tokens) == len(starts) == len(ends) == state.word_count):
        raise TraceGenerationError("backend model-word coordinates are not aligned")
    previous_end = 0
    for index, (token, start, end) in enumerate(zip(tokens, starts, ends, strict=True)):
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not previous_end <= start < end <= len(expected_text)
            or expected_text[start:end] != token
        ):
            raise TraceGenerationError(
                f"backend model-word coordinate {index} does not round-trip"
            )
        previous_end = end
    return starts, ends


def _trace_update(
    update: StreamingSpanUpdate,
    *,
    run_id: str,
    example_id: str,
    step: int,
    chunk: str,
    visible_text: str,
    labels: tuple[str, ...],
    starts: tuple[int, ...],
    ends: tuple[int, ...],
    previous_probabilities: Mapping[SpanBoundary, tuple[float, ...]],
) -> SpanScoreUpdate:
    boundary = SpanBoundary(update.start_word, update.end_word)
    if boundary.end_word >= len(starts):
        raise TraceGenerationError(
            f"updated boundary {boundary.to_tuple()} exceeds authoritative model words"
        )
    logits = tuple(float(value) for value in update.logits)
    probabilities = tuple(float(value) for value in update.probs)
    if len(logits) != len(labels) or len(probabilities) != len(labels):
        raise TraceGenerationError("updated class vector differs from the fixed label count")
    expected_probabilities = tuple(float(value) for value in sigmoid(logits))
    if not np.allclose(probabilities, expected_probabilities, rtol=1e-12, atol=1e-12):
        raise TraceGenerationError("updated probabilities are not sigmoid(raw logits)")

    top_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    top_probability = probabilities[top_index]
    remaining = probabilities[:top_index] + probabilities[top_index + 1 :]
    second_probability = max(remaining, default=0.0)
    previous = previous_probabilities.get(boundary)
    previous_top = None if previous is None else max(previous)
    start_char = starts[boundary.start_word]
    end_char = ends[boundary.end_word]
    return SpanScoreUpdate(
        run_id=run_id,
        example_id=example_id,
        step=step,
        chunk=chunk,
        visible_char_count=len(visible_text),
        visible_word_count=len(starts),
        start_word=boundary.start_word,
        end_word=boundary.end_word,
        start_char=start_char,
        end_char=end_char,
        span_text=visible_text[start_char:end_char],
        logits=logits,
        probs=probabilities,
        top_label_index=top_index,
        top_label=labels[top_index],
        top_probability=top_probability,
        second_probability=second_probability,
        label_margin=top_probability - second_probability,
        previous_top_probability=previous_top,
        top_probability_delta=(
            None if previous_top is None else top_probability - previous_top
        ),
        update_kind=update.update_kind,
        tail_distance_words=(len(starts) - 1) - boundary.end_word,
    )


def _copy_historical_state(
    state: Mapping[Any, object],
) -> dict[SpanBoundary, tuple[float, ...]]:
    copied: dict[SpanBoundary, tuple[float, ...]] = {}
    for raw_boundary, raw_vector in state.items():
        if isinstance(raw_boundary, SpanBoundary):
            boundary = raw_boundary
        elif isinstance(raw_boundary, tuple) and len(raw_boundary) == 2:
            boundary = SpanBoundary(raw_boundary[0], raw_boundary[1])
        else:
            raise TraceGenerationError("historical score state has an invalid boundary key")
        array = np.asarray(raw_vector, dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise TraceGenerationError(
                f"historical score at {boundary.to_tuple()} is not a finite vector"
            )
        copied[boundary] = tuple(float(value) for value in array)
    return copied


def generate_example_trace(
    backend: MLXStreamingBackend,
    example: TraceInputExample,
    *,
    run_id: str,
    chunk_words: int,
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
) -> GeneratedExampleTrace:
    """Run one warm stream once, verify replay, clear it, then run cold once."""

    _identifier(run_id, name="run_id")
    if not isinstance(example, TraceInputExample):
        raise TypeError("example must be a TraceInputExample")
    chunk_words = _chunk_condition(chunk_words)
    chunks = tuple(chunk_text_by_words(example.text, chunk_words))
    if not chunks or "".join(chunks) != example.text:
        raise TraceGenerationError("chunk schedule did not reconstruct source text exactly")

    session_id = f"{run_id}:{example.example_id}:chunk-{chunk_words}"
    session = backend.start_session(
        example.labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        session_id=session_id,
    )
    snapshots: list[SnapshotStep] = []
    updates: list[SpanScoreUpdate] = []
    previous_probabilities: dict[SpanBoundary, tuple[float, ...]] = {}
    visible_text = ""
    final_state: dict[SpanBoundary, tuple[float, ...]]
    try:
        for step, chunk in enumerate(chunks, start=1):
            visible_text += chunk
            result: StreamingAppendResult = session.append(chunk)
            if result.is_noop:
                raise TraceGenerationError("a nonblank scheduled chunk became a backend no-op")
            if result.state.labels != example.labels:
                raise TraceGenerationError("backend changed the fixed session label order")
            starts, ends = _validated_word_coordinates(
                result.state,
                expected_text=visible_text,
            )
            seen_this_step: set[SpanBoundary] = set()
            for raw_update in result.span_updates:
                boundary = SpanBoundary(raw_update.start_word, raw_update.end_word)
                if boundary in seen_this_step:
                    raise TraceGenerationError(
                        f"duplicate boundary {boundary} in append candidate set"
                    )
                seen_this_step.add(boundary)
                expected_kind = "rescore" if boundary in previous_probabilities else "new"
                if raw_update.update_kind != expected_kind:
                    raise TraceGenerationError(
                        f"backend update kind for {boundary} is {raw_update.update_kind}, "
                        f"expected {expected_kind}"
                    )
                converted = _trace_update(
                    raw_update,
                    run_id=run_id,
                    example_id=example.example_id,
                    step=step,
                    chunk=chunk,
                    visible_text=visible_text,
                    labels=example.labels,
                    starts=starts,
                    ends=ends,
                    previous_probabilities=previous_probabilities,
                )
                updates.append(converted)
                previous_probabilities[boundary] = converted.probs

            snapshots.append(
                SnapshotStep(
                    run_id=run_id,
                    example_id=example.example_id,
                    step=step,
                    chunk=chunk,
                    accumulated_text=visible_text,
                    visible_char_count=len(visible_text),
                    visible_word_count=len(starts),
                    elapsed_ms=result.elapsed_ms,
                    public_entities=result.public_entities,
                )
            )

        if visible_text != example.text:
            raise TraceGenerationError("streaming trace did not consume the complete source text")
        final_state = _copy_historical_state(session.historical_logits)
        if snapshots[-1].visible_word_count == 0:
            raise TraceGenerationError("completed nonblank trace has no model words")
        replayed = replay_span_updates(updates)
        try:
            assert_span_states_close(replayed, final_state, rel_tol=0.0, abs_tol=0.0)
        except AssertionError as error:
            raise TraceGenerationError(
                f"warm event replay differs from backend historical state: {error}"
            ) from error
        final_state_sha256 = span_state_sha256(replayed)
    finally:
        session.clear()

    cold_full = backend.infer_full_trace(
        example.text,
        example.labels,
        example_id=example.example_id,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )
    if cold_full.full_text != example.text:
        raise TraceGenerationError("cold result text differs from exact source text")
    return GeneratedExampleTrace(
        run_id=run_id,
        chunk_words=chunk_words,
        example=example,
        snapshots=tuple(snapshots),
        span_updates=tuple(updates),
        cold_full=cold_full,
        final_state_sha256=final_state_sha256,
    )


def generate_condition_traces(
    backend: MLXStreamingBackend,
    examples: Sequence[TraceInputExample],
    *,
    run_id: str,
    chunk_words: int,
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
) -> tuple[GeneratedExampleTrace, ...]:
    """Trace each example independently for one approved chunk condition."""

    if isinstance(examples, str | bytes) or not isinstance(examples, Sequence):
        raise TypeError("examples must be a sequence of TraceInputExample values")
    prepared = tuple(examples)
    if not all(isinstance(example, TraceInputExample) for example in prepared):
        raise TypeError("examples must contain only TraceInputExample values")
    identifiers: set[str] = set()
    for example in prepared:
        if example.example_id in identifiers:
            raise ValueError(f"duplicate example_id {example.example_id!r}")
        identifiers.add(example.example_id)

    generated: list[GeneratedExampleTrace] = []
    for example in prepared:
        generated.append(
            generate_example_trace(
                backend,
                example,
                run_id=run_id,
                chunk_words=chunk_words,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            )
        )
    return tuple(generated)


__all__ = [
    "GeneratedExampleTrace",
    "TRACE_CHUNK_WORDS",
    "TraceGenerationError",
    "TraceInputExample",
    "generate_condition_traces",
    "generate_example_trace",
    "span_state_sha256",
]
