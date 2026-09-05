"""Exact NER and PIIMB-style character-coverage metrics.

The functions in this module deliberately operate on small mappings or objects
with matching attributes.  Evaluation therefore does not depend on a policy
implementation (or on its serialization classes).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

type Record = Mapping[str, Any] | object


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a benchmark-safe ratio (zero when the denominator is zero)."""

    return float(numerator / denominator) if denominator else 0.0


def _f_beta(precision: float, recall: float, *, beta: float) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return _ratio((1.0 + beta_squared) * precision * recall, denominator)


@dataclass(frozen=True, slots=True)
class StrictNERMetrics:
    """Additive exact-match counts with derived precision, recall, and F1."""

    true_positives: int
    false_positives: int
    false_negatives: int

    def __post_init__(self) -> None:
        for name, value in (
            ("true_positives", self.true_positives),
            ("false_positives", self.false_positives),
            ("false_negatives", self.false_negatives),
        ):
            _require_nonnegative_int(name, value)

    @property
    def precision(self) -> float:
        return _ratio(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        return _f_beta(self.precision, self.recall, beta=1.0)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True, slots=True)
class MaskingMetrics:
    """Additive label-agnostic character counts and PIIMB-style rates."""

    true_positive_chars: int
    predicted_chars: int
    gold_chars: int
    non_pii_chars: int

    def __post_init__(self) -> None:
        for name, value in (
            ("true_positive_chars", self.true_positive_chars),
            ("predicted_chars", self.predicted_chars),
            ("gold_chars", self.gold_chars),
            ("non_pii_chars", self.non_pii_chars),
        ):
            _require_nonnegative_int(name, value)
        if self.true_positive_chars > self.predicted_chars:
            raise ValueError("true_positive_chars cannot exceed predicted_chars")
        if self.true_positive_chars > self.gold_chars:
            raise ValueError("true_positive_chars cannot exceed gold_chars")
        if self.false_positive_chars > self.non_pii_chars:
            raise ValueError("false-positive coverage cannot exceed non-PII characters")

    @property
    def false_positive_chars(self) -> int:
        return self.predicted_chars - self.true_positive_chars

    @property
    def precision(self) -> float:
        return _ratio(self.true_positive_chars, self.predicted_chars)

    @property
    def recall(self) -> float:
        return _ratio(self.true_positive_chars, self.gold_chars)

    @property
    def f1(self) -> float:
        return _f_beta(self.precision, self.recall, beta=1.0)

    @property
    def f2(self) -> float:
        return _f_beta(self.precision, self.recall, beta=2.0)

    @property
    def false_positive_rate(self) -> float:
        return _ratio(self.false_positive_chars, self.non_pii_chars)

    @property
    def fpr(self) -> float:
        """Short alias used in experiment tables."""

        return self.false_positive_rate

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positive_chars": self.true_positive_chars,
            "false_positive_chars": self.false_positive_chars,
            "predicted_chars": self.predicted_chars,
            "gold_chars": self.gold_chars,
            "non_pii_chars": self.non_pii_chars,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "fpr": self.fpr,
        }


@dataclass(frozen=True, slots=True)
class NERScoreAverage:
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1}


@dataclass(frozen=True, slots=True)
class MaskingScoreAverage:
    precision: float
    recall: float
    f1: float
    f2: float
    fpr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
            "fpr": self.fpr,
        }


@dataclass(frozen=True, slots=True)
class TaskNERAggregate:
    """Per-task micro counts, pooled micro, and unweighted task macro."""

    per_task: Mapping[str, StrictNERMetrics]
    pooled_micro: StrictNERMetrics
    task_macro: NERScoreAverage


@dataclass(frozen=True, slots=True)
class TaskMaskingAggregate:
    """PIIMB per-task micro counts and their unweighted task macro."""

    per_task: Mapping[str, MaskingMetrics]
    pooled_micro: MaskingMetrics
    task_macro: MaskingScoreAverage


def strict_ner_metrics(
    predicted: Iterable[Record],
    gold: Iterable[Record],
    *,
    scope_field: str | None = None,
) -> StrictNERMetrics:
    """Calculate strict entity metrics for ``(start_char, end_char, label)``.

    Call this once per example by default.  For pooled records, pass a field
    such as ``example_id`` as ``scope_field`` so equal offsets in different
    examples remain distinct.  Duplicate identical entities are collapsed.
    """

    predicted_ids = {_entity_identity(item, scope_field=scope_field) for item in predicted}
    gold_ids = {_entity_identity(item, scope_field=scope_field) for item in gold}
    true_positives = len(predicted_ids & gold_ids)
    return StrictNERMetrics(
        true_positives=true_positives,
        false_positives=len(predicted_ids - gold_ids),
        false_negatives=len(gold_ids - predicted_ids),
    )


def merge_half_open_intervals(
    intervals: Iterable[Record | tuple[int, int]],
    *,
    text_length: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return the union of overlapping *or adjacent* half-open intervals."""

    if text_length is not None:
        _require_nonnegative_int("text_length", text_length)
    normalized = sorted(_interval(item, text_length=text_length) for item in intervals)
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = previous_start, max(previous_end, end)
    return tuple(merged)


def piimb_masking_metrics(
    predicted: Iterable[Record | tuple[int, int]],
    gold: Iterable[Record | tuple[int, int]],
    *,
    text_length: int,
) -> MaskingMetrics:
    """Calculate label-agnostic character coverage with additive counts.

    Both gold and predicted coverage are unioned, merging adjacent intervals.
    This prevents overlaps from double-counting characters.  FPR uses the
    number of non-PII characters in the source text as its denominator.
    """

    _require_nonnegative_int("text_length", text_length)
    predicted_intervals = merge_half_open_intervals(predicted, text_length=text_length)
    gold_intervals = merge_half_open_intervals(gold, text_length=text_length)
    predicted_chars = _covered_length(predicted_intervals)
    gold_chars = _covered_length(gold_intervals)
    true_positive_chars = _intersection_length(predicted_intervals, gold_intervals)
    return MaskingMetrics(
        true_positive_chars=true_positive_chars,
        predicted_chars=predicted_chars,
        gold_chars=gold_chars,
        non_pii_chars=text_length - gold_chars,
    )


def aggregate_ner_by_task(
    results: Mapping[str, Iterable[StrictNERMetrics]],
) -> TaskNERAggregate:
    """Micro-aggregate within each task, then average tasks without weighting."""

    per_task = {
        _task_name(task): _sum_ner_metrics(metrics) for task, metrics in sorted(results.items())
    }
    pooled = _sum_ner_metrics(per_task.values())
    return TaskNERAggregate(
        per_task=per_task,
        pooled_micro=pooled,
        task_macro=NERScoreAverage(
            precision=_mean(metric.precision for metric in per_task.values()),
            recall=_mean(metric.recall for metric in per_task.values()),
            f1=_mean(metric.f1 for metric in per_task.values()),
        ),
    )


def aggregate_masking_by_task(
    results: Mapping[str, Iterable[MaskingMetrics]],
) -> TaskMaskingAggregate:
    """Apply PIIMB's per-task micro then unweighted task-macro methodology."""

    per_task = {
        _task_name(task): _sum_masking_metrics(metrics) for task, metrics in sorted(results.items())
    }
    pooled = _sum_masking_metrics(per_task.values())
    return TaskMaskingAggregate(
        per_task=per_task,
        pooled_micro=pooled,
        task_macro=MaskingScoreAverage(
            precision=_mean(metric.precision for metric in per_task.values()),
            recall=_mean(metric.recall for metric in per_task.values()),
            f1=_mean(metric.f1 for metric in per_task.values()),
            f2=_mean(metric.f2 for metric in per_task.values()),
            fpr=_mean(metric.fpr for metric in per_task.values()),
        ),
    )


def _sum_ner_metrics(metrics: Iterable[StrictNERMetrics]) -> StrictNERMetrics:
    rows = tuple(metrics)
    if not all(isinstance(row, StrictNERMetrics) for row in rows):
        raise TypeError("NER aggregation accepts StrictNERMetrics values")
    return StrictNERMetrics(
        true_positives=sum(row.true_positives for row in rows),
        false_positives=sum(row.false_positives for row in rows),
        false_negatives=sum(row.false_negatives for row in rows),
    )


def _sum_masking_metrics(metrics: Iterable[MaskingMetrics]) -> MaskingMetrics:
    rows = tuple(metrics)
    if not all(isinstance(row, MaskingMetrics) for row in rows):
        raise TypeError("masking aggregation accepts MaskingMetrics values")
    return MaskingMetrics(
        true_positive_chars=sum(row.true_positive_chars for row in rows),
        predicted_chars=sum(row.predicted_chars for row in rows),
        gold_chars=sum(row.gold_chars for row in rows),
        non_pii_chars=sum(row.non_pii_chars for row in rows),
    )


def _mean(values: Iterable[float]) -> float:
    rows = tuple(values)
    return statistics.fmean(rows) if rows else 0.0


def _task_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task names must be non-blank strings")
    return value


def _entity_identity(item: Record, *, scope_field: str | None) -> tuple[Any, int, int, str]:
    start, end = _interval(item, text_length=None)
    label = _record_field(item, "label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("entity label must be a non-blank string")
    scope = _record_field(item, scope_field) if scope_field is not None else None
    try:
        hash(scope)
    except TypeError as error:
        raise TypeError(f"{scope_field} must be hashable") from error
    return scope, start, end, label


def _interval(item: Record | tuple[int, int], *, text_length: int | None) -> tuple[int, int]:
    if isinstance(item, tuple):
        if len(item) != 2:
            raise ValueError("interval tuples must contain exactly two integers")
        start, end = item
    else:
        start = _record_field(item, "start_char")
        end = _record_field(item, "end_char")
    _require_nonnegative_int("start_char", start)
    _require_nonnegative_int("end_char", end)
    if end <= start:
        raise ValueError("end_char must be greater than start_char")
    if text_length is not None and end > text_length:
        raise ValueError("character interval extends beyond text_length")
    return start, end


def _record_field(item: Record, name: str | None) -> Any:
    if name is None:
        return None
    if isinstance(item, Mapping):
        if name not in item:
            raise ValueError(f"record is missing {name}")
        return item[name]
    try:
        return getattr(item, name)
    except AttributeError as error:
        raise ValueError(f"record is missing {name}") from error


def _covered_length(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_length(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> int:
    total = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


__all__ = [
    "MaskingMetrics",
    "MaskingScoreAverage",
    "NERScoreAverage",
    "StrictNERMetrics",
    "TaskMaskingAggregate",
    "TaskNERAggregate",
    "aggregate_masking_by_task",
    "aggregate_ner_by_task",
    "merge_half_open_intervals",
    "piimb_masking_metrics",
    "strict_ner_metrics",
]
