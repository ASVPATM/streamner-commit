"""Small, immutable records shared by tracing, replay, and evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

UpdateKind = Literal["new", "rescore", "full"]

_PROBABILITY_TOLERANCE = 1e-6


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_string(name: str, value: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")


def _require_identifier(name: str, value: str) -> None:
    _require_string(name, value)
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _probability(name: str, value: float) -> float:
    converted = _finite_float(name, value)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _float_tuple(name: str, values: Sequence[float], *, probabilities: bool) -> tuple[float, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers")
    converter = _probability if probabilities else _finite_float
    converted = tuple(converter(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


def _validate_char_span(start_char: int, end_char: int, text: str) -> None:
    _require_int("start_char", start_char)
    _require_int("end_char", end_char)
    if end_char <= start_char:
        raise ValueError("end_char must be greater than start_char")
    _require_string("text", text)
    if end_char - start_char != len(text):
        raise ValueError("character span length must equal len(text)")


def _is_close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    )


@dataclass(frozen=True, slots=True)
class GoldEntity:
    """A gold entity using inclusive-start, exclusive-end character offsets."""

    example_id: str
    start_char: int
    end_char: int
    label: str
    text: str

    def __post_init__(self) -> None:
        _require_identifier("example_id", self.example_id)
        _require_identifier("label", self.label)
        _validate_char_span(self.start_char, self.end_char, self.text)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "example_id": self.example_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "label": self.label,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class PublicEntity:
    """An entity exposed by a backend's public decoder."""

    start_char: int
    end_char: int
    label: str
    text: str
    score: float

    def __post_init__(self) -> None:
        _require_identifier("label", self.label)
        _validate_char_span(self.start_char, self.end_char, self.text)
        object.__setattr__(self, "score", _probability("score", self.score))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "label": self.label,
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True, order=True)
class SpanBoundary:
    """An inclusive word-span boundary."""

    start_word: int
    end_word: int

    def __post_init__(self) -> None:
        _require_int("start_word", self.start_word)
        _require_int("end_word", self.end_word)
        if self.end_word < self.start_word:
            raise ValueError("end_word must be greater than or equal to start_word")

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe representation."""
        return {"start_word": self.start_word, "end_word": self.end_word}

    def to_tuple(self) -> tuple[int, int]:
        """Return the boundary in the reference cache's key format."""
        return self.start_word, self.end_word


@dataclass(frozen=True, slots=True)
class SpanScoreUpdate:
    """One complete class-score-vector update for one word span."""

    run_id: str
    example_id: str
    step: int
    chunk: str
    visible_char_count: int
    visible_word_count: int
    start_word: int
    end_word: int
    start_char: int
    end_char: int
    span_text: str
    logits: tuple[float, ...]
    probs: tuple[float, ...]
    top_label_index: int
    top_label: str
    top_probability: float
    second_probability: float
    label_margin: float
    previous_top_probability: float | None
    top_probability_delta: float | None
    update_kind: UpdateKind
    tail_distance_words: int

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        _require_identifier("example_id", self.example_id)
        _require_int("step", self.step)
        _require_string("chunk", self.chunk, allow_empty=True)
        _require_int("visible_char_count", self.visible_char_count)
        _require_int("visible_word_count", self.visible_word_count)

        boundary = SpanBoundary(self.start_word, self.end_word)
        _validate_char_span(self.start_char, self.end_char, self.span_text)
        if self.end_char > self.visible_char_count:
            raise ValueError("end_char cannot exceed visible_char_count")
        if boundary.end_word >= self.visible_word_count:
            raise ValueError("end_word must refer to a visible word")

        logits = _float_tuple("logits", self.logits, probabilities=False)
        probabilities = _float_tuple("probs", self.probs, probabilities=True)
        if len(logits) != len(probabilities):
            raise ValueError("logits and probs must have equal lengths")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "probs", probabilities)

        _require_int("top_label_index", self.top_label_index)
        if self.top_label_index >= len(probabilities):
            raise ValueError("top_label_index is outside the class score vector")
        _require_identifier("top_label", self.top_label)

        top_probability = _probability("top_probability", self.top_probability)
        second_probability = _probability("second_probability", self.second_probability)
        label_margin = _finite_float("label_margin", self.label_margin)
        object.__setattr__(self, "top_probability", top_probability)
        object.__setattr__(self, "second_probability", second_probability)
        object.__setattr__(self, "label_margin", label_margin)

        if not _is_close(top_probability, probabilities[self.top_label_index]):
            raise ValueError("top_probability must equal probs[top_label_index]")
        if not _is_close(top_probability, max(probabilities)):
            raise ValueError("top_probability must be the largest class probability")
        remaining = (
            probabilities[: self.top_label_index] + probabilities[self.top_label_index + 1 :]
        )
        expected_second = max(remaining, default=0.0)
        if not _is_close(second_probability, expected_second):
            raise ValueError("second_probability must be the next-largest class probability")
        if not _is_close(label_margin, top_probability - second_probability):
            raise ValueError("label_margin must equal top_probability - second_probability")

        if (self.previous_top_probability is None) != (self.top_probability_delta is None):
            raise ValueError(
                "previous_top_probability and top_probability_delta must both be set "
                "or both be null"
            )
        if self.previous_top_probability is not None:
            previous = _probability("previous_top_probability", self.previous_top_probability)
            delta_value = self.top_probability_delta
            assert delta_value is not None
            delta = _finite_float("top_probability_delta", delta_value)
            if not _is_close(delta, top_probability - previous):
                raise ValueError(
                    "top_probability_delta must equal top_probability - previous_top_probability"
                )
            object.__setattr__(self, "previous_top_probability", previous)
            object.__setattr__(self, "top_probability_delta", delta)

        if self.update_kind not in {"new", "rescore", "full"}:
            raise ValueError("update_kind must be one of: new, rescore, full")
        _require_int("tail_distance_words", self.tail_distance_words)
        expected_tail_distance = (self.visible_word_count - 1) - self.end_word
        if self.tail_distance_words != expected_tail_distance:
            raise ValueError("tail_distance_words must equal (visible_word_count - 1) - end_word")

    @property
    def boundary(self) -> SpanBoundary:
        """Return this update's word boundary."""
        return SpanBoundary(self.start_word, self.end_word)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for a trace row."""
        return {
            "run_id": self.run_id,
            "example_id": self.example_id,
            "step": self.step,
            "chunk": self.chunk,
            "visible_char_count": self.visible_char_count,
            "visible_word_count": self.visible_word_count,
            "start_word": self.start_word,
            "end_word": self.end_word,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "span_text": self.span_text,
            "logits": list(self.logits),
            "probs": list(self.probs),
            "top_label_index": self.top_label_index,
            "top_label": self.top_label,
            "top_probability": self.top_probability,
            "second_probability": self.second_probability,
            "label_margin": self.label_margin,
            "previous_top_probability": self.previous_top_probability,
            "top_probability_delta": self.top_probability_delta,
            "update_kind": self.update_kind,
            "tail_distance_words": self.tail_distance_words,
        }


def _public_entity_tuple(values: Sequence[PublicEntity]) -> tuple[PublicEntity, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("public_entities must be a sequence")
    entities = tuple(values)
    if not all(isinstance(entity, PublicEntity) for entity in entities):
        raise TypeError("public_entities must contain only PublicEntity values")
    return entities


def _validate_public_entities(entities: tuple[PublicEntity, ...], text: str) -> None:
    for entity in entities:
        if entity.end_char > len(text):
            raise ValueError("public entity extends beyond the visible text")
        if text[entity.start_char : entity.end_char] != entity.text:
            raise ValueError("public entity text does not match its character offsets")


@dataclass(frozen=True, slots=True)
class SnapshotStep:
    """The complete public entity snapshot after one streaming append."""

    run_id: str
    example_id: str
    step: int
    chunk: str
    accumulated_text: str
    visible_char_count: int
    visible_word_count: int
    elapsed_ms: float
    public_entities: tuple[PublicEntity, ...]

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        _require_identifier("example_id", self.example_id)
        _require_int("step", self.step)
        _require_string("chunk", self.chunk, allow_empty=True)
        _require_string("accumulated_text", self.accumulated_text, allow_empty=True)
        _require_int("visible_char_count", self.visible_char_count)
        _require_int("visible_word_count", self.visible_word_count)
        if self.visible_char_count != len(self.accumulated_text):
            raise ValueError("visible_char_count must equal len(accumulated_text)")
        if not self.accumulated_text.endswith(self.chunk):
            raise ValueError("accumulated_text must end with chunk")
        elapsed_ms = _finite_float("elapsed_ms", self.elapsed_ms)
        if elapsed_ms < 0.0:
            raise ValueError("elapsed_ms must not be negative")
        object.__setattr__(self, "elapsed_ms", elapsed_ms)

        entities = _public_entity_tuple(self.public_entities)
        _validate_public_entities(entities, self.accumulated_text)
        object.__setattr__(self, "public_entities", entities)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "run_id": self.run_id,
            "example_id": self.example_id,
            "step": self.step,
            "chunk": self.chunk,
            "accumulated_text": self.accumulated_text,
            "visible_char_count": self.visible_char_count,
            "visible_word_count": self.visible_word_count,
            "elapsed_ms": self.elapsed_ms,
            "public_entities": [entity.to_dict() for entity in self.public_entities],
        }


def _immutable_span_state(
    state: Mapping[SpanBoundary, Sequence[float]],
) -> Mapping[SpanBoundary, tuple[float, ...]]:
    if not isinstance(state, Mapping):
        raise TypeError("raw_final_span_state must be a mapping")
    normalized: dict[SpanBoundary, tuple[float, ...]] = {}
    vector_length: int | None = None
    for key, values in state.items():
        if isinstance(key, SpanBoundary):
            boundary = key
        elif isinstance(key, tuple) and len(key) == 2:
            boundary = SpanBoundary(key[0], key[1])
        else:
            raise TypeError("raw_final_span_state keys must be SpanBoundary or integer pairs")
        if boundary in normalized:
            raise ValueError(f"duplicate span boundary: {boundary.to_tuple()}")
        logits = _float_tuple(
            f"raw_final_span_state[{boundary.to_tuple()}]",
            values,
            probabilities=False,
        )
        if vector_length is None:
            vector_length = len(logits)
        elif len(logits) != vector_length:
            raise ValueError("all raw span score vectors must have equal lengths")
        normalized[boundary] = logits
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ColdFullResult:
    """A cold full-text prediction and its final raw span-score map."""

    example_id: str
    full_text: str
    public_entities: tuple[PublicEntity, ...]
    raw_final_span_state: Mapping[SpanBoundary, tuple[float, ...]] = field(hash=False)

    def __post_init__(self) -> None:
        _require_identifier("example_id", self.example_id)
        _require_string("full_text", self.full_text, allow_empty=True)
        entities = _public_entity_tuple(self.public_entities)
        _validate_public_entities(entities, self.full_text)
        object.__setattr__(self, "public_entities", entities)
        object.__setattr__(
            self,
            "raw_final_span_state",
            _immutable_span_state(self.raw_final_span_state),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data, encoding tuple-keyed raw state as sorted rows."""
        raw_state = [
            {
                "start_word": boundary.start_word,
                "end_word": boundary.end_word,
                "logits": list(logits),
            }
            for boundary, logits in sorted(self.raw_final_span_state.items())
        ]
        return {
            "example_id": self.example_id,
            "full_text": self.full_text,
            "public_entities": [entity.to_dict() for entity in self.public_entities],
            "raw_final_span_state": raw_state,
        }


__all__ = [
    "ColdFullResult",
    "GoldEntity",
    "PublicEntity",
    "SnapshotStep",
    "SpanBoundary",
    "SpanScoreUpdate",
    "UpdateKind",
]
