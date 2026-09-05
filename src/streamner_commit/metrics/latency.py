"""Streaming visibility and commitment-delay metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

type Record = Mapping[str, Any] | object


@dataclass(frozen=True, slots=True)
class VisibilityPoint:
    """The first update at which a complete gold span is observable."""

    start_char: int
    end_char: int
    label: str
    gold_visible_step: int
    gold_end_word: int
    example_id: str | None = None


@dataclass(frozen=True, slots=True)
class CommitmentDelay:
    """Comparable word-context and update-granularity delay components."""

    gold_visible_step: int
    gold_end_word: int
    commit_step: int
    visible_word_count_at_commit: int
    first_detection_step: int | None
    commit_context_words: int
    update_delay_steps: int
    model_detection_delay_steps: int | None
    policy_added_delay_steps: int | None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "gold_visible_step": self.gold_visible_step,
            "gold_end_word": self.gold_end_word,
            "commit_step": self.commit_step,
            "visible_word_count_at_commit": self.visible_word_count_at_commit,
            "first_detection_step": self.first_detection_step,
            "commit_context_words": self.commit_context_words,
            "update_delay_steps": self.update_delay_steps,
            "model_detection_delay_steps": self.model_detection_delay_steps,
            "policy_added_delay_steps": self.policy_added_delay_steps,
        }


def gold_end_word(gold: Record, word_offsets: Sequence[Record | tuple[int, int]]) -> int:
    """Resolve a gold end offset to the backend's authoritative word index.

    A span-update table may be passed directly: all rows ending at the gold
    character boundary must agree on ``end_word``.  A one-row-per-word offset
    table may instead expose ``word_index``.  Bare ``(start, end)`` offsets are
    indexed by their sequence position.
    """

    _, gold_end, _ = _entity_identity(gold)
    exact: set[int] = set()
    single_word_containing: set[int] = set()
    for position, row in enumerate(word_offsets):
        if isinstance(row, tuple):
            if len(row) != 2:
                raise ValueError("word offset tuples must contain two integers")
            start_char, end_char = row
            index = position
            is_single_word = True
        else:
            start_char = _integer_field(row, "start_char")
            end_char = _integer_field(row, "end_char")
            index_value = _optional_field(row, "word_index")
            if index_value is None:
                index_value = _field(row, "end_word")
            index = _integer(index_value, "word index")
            start_word = _optional_field(row, "start_word")
            is_single_word = start_word is None or _integer(start_word, "start_word") == index
        _validate_interval(start_char, end_char)
        if end_char == gold_end:
            exact.add(index)
        if is_single_word and start_char < gold_end <= end_char:
            single_word_containing.add(index)

    candidates = exact or single_word_containing
    if not candidates:
        raise ValueError("gold end offset does not map to a model word")
    if len(candidates) != 1:
        raise ValueError("gold end offset maps to inconsistent model word indices")
    return next(iter(candidates))


def gold_visibility_point(
    gold: Record,
    steps: Sequence[Record],
    word_offsets: Sequence[Record | tuple[int, int]],
) -> VisibilityPoint:
    """Find the first step whose accumulated text includes ``gold.end_char``."""

    start_char, end_char, label = _entity_identity(gold)
    ordered_steps = sorted(
        ((_integer_field(step, "step"), _visible_char_count(step)) for step in steps),
        key=lambda item: item[0],
    )
    visible_steps = [step for step, visible_chars in ordered_steps if visible_chars >= end_char]
    if not visible_steps:
        raise ValueError("gold entity is not fully visible in the supplied steps")
    example_value = _optional_field(gold, "example_id")
    if example_value is not None and not isinstance(example_value, str):
        raise TypeError("example_id must be a string when present")
    return VisibilityPoint(
        start_char=start_char,
        end_char=end_char,
        label=label,
        gold_visible_step=visible_steps[0],
        gold_end_word=gold_end_word(gold, word_offsets),
        example_id=example_value,
    )


def gold_visibility_points(
    gold_entities: Iterable[Record],
    steps: Sequence[Record],
    word_offsets: Sequence[Record | tuple[int, int]],
) -> tuple[VisibilityPoint, ...]:
    """Resolve all gold entities in deterministic offset/label order."""

    points = [gold_visibility_point(gold, steps, word_offsets) for gold in gold_entities]
    return tuple(sorted(points, key=lambda item: (item.start_char, item.end_char, item.label)))


def first_exact_detection_step(
    gold: Record,
    snapshots: Sequence[Record],
    *,
    threshold: float = 0.5,
    entities_field: str = "public_entities",
) -> int | None:
    """Return the first exact internal detection at a canonical score threshold."""

    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("threshold must be numeric")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    gold_identity = _entity_identity(gold)
    for snapshot in sorted(snapshots, key=lambda item: _integer_field(item, "step")):
        entities = _field(snapshot, entities_field)
        if isinstance(entities, str | bytes) or not isinstance(entities, Sequence):
            raise TypeError(f"{entities_field} must be a sequence")
        for entity in entities:
            if _entity_identity(entity) != gold_identity:
                continue
            score_value = _optional_field(entity, "score")
            if score_value is None:
                score_value = _optional_field(entity, "top_probability")
            if score_value is None:
                raise ValueError("detected entity must provide score or top_probability")
            score = _finite_float(score_value, "entity score")
            if score >= threshold:
                return _integer_field(snapshot, "step")
    return None


def commitment_delay_record(
    visibility: VisibilityPoint,
    *,
    commit_step: int,
    visible_word_count_at_commit: int,
    first_detection_step: int | None,
) -> CommitmentDelay:
    """Separate base-model delay from policy-added delay for a correct commit."""

    if not isinstance(visibility, VisibilityPoint):
        raise TypeError("visibility must be a VisibilityPoint")
    commit_step = _integer(commit_step, "commit_step")
    visible_word_count_at_commit = _integer(
        visible_word_count_at_commit, "visible_word_count_at_commit"
    )
    if commit_step < visibility.gold_visible_step:
        raise ValueError("a correct commitment cannot precede gold visibility")
    commit_context_words = visible_word_count_at_commit - (visibility.gold_end_word + 1)
    if commit_context_words < 0:
        raise ValueError("commitment cannot precede visibility of the gold end word")

    model_delay: int | None = None
    policy_delay: int | None = None
    if first_detection_step is not None:
        first_detection_step = _integer(first_detection_step, "first_detection_step")
        if first_detection_step < visibility.gold_visible_step:
            raise ValueError("exact detection cannot precede gold visibility")
        model_delay = first_detection_step - visibility.gold_visible_step
        policy_delay = commit_step - first_detection_step

    return CommitmentDelay(
        gold_visible_step=visibility.gold_visible_step,
        gold_end_word=visibility.gold_end_word,
        commit_step=commit_step,
        visible_word_count_at_commit=visible_word_count_at_commit,
        first_detection_step=first_detection_step,
        commit_context_words=commit_context_words,
        update_delay_steps=commit_step - visibility.gold_visible_step,
        model_detection_delay_steps=model_delay,
        policy_added_delay_steps=policy_delay,
    )


def _entity_identity(item: Record) -> tuple[int, int, str]:
    start = _integer_field(item, "start_char")
    end = _integer_field(item, "end_char")
    _validate_interval(start, end)
    label = _field(item, "label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-blank string")
    return start, end, label


def _visible_char_count(step: Record) -> int:
    value = _optional_field(step, "visible_char_count")
    if value is not None:
        return _integer(value, "visible_char_count")
    text = _field(step, "accumulated_text")
    if not isinstance(text, str):
        raise TypeError("accumulated_text must be a string")
    return len(text)


def _validate_interval(start: object, end: object) -> None:
    start_value = _integer(start, "start_char")
    end_value = _integer(end, "end_char")
    if end_value <= start_value:
        raise ValueError("end_char must be greater than start_char")


def _integer_field(item: Record, name: str) -> int:
    return _integer(_field(item, name), name)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _field(item: Record, name: str) -> Any:
    if isinstance(item, Mapping):
        if name not in item:
            raise ValueError(f"record is missing {name}")
        return item[name]
    try:
        return getattr(item, name)
    except AttributeError as error:
        raise ValueError(f"record is missing {name}") from error


def _optional_field(item: Record, name: str) -> Any | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


__all__ = [
    "CommitmentDelay",
    "VisibilityPoint",
    "commitment_delay_record",
    "first_exact_detection_step",
    "gold_end_word",
    "gold_visibility_point",
    "gold_visibility_points",
]
