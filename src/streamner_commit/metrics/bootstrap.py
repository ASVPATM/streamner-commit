"""Deterministic document bootstrap and multi-objective Pareto helpers."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

type Record = Mapping[str, Any]
type RecordStatistic = Callable[[Sequence[Record]], float]

DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 20_260_901


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A paired comparison estimate and percentile confidence interval."""

    baseline_estimate: float
    comparison_estimate: float
    difference: float
    lower: float
    upper: float
    confidence: float
    replicates: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "baseline_estimate": self.baseline_estimate,
            "comparison_estimate": self.comparison_estimate,
            "difference": self.difference,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "replicates": self.replicates,
            "seed": self.seed,
        }


def paired_stratified_bootstrap(
    records: Sequence[Record],
    *,
    baseline_statistic: RecordStatistic,
    comparison_statistic: RecordStatistic,
    parent_field: str = "parent_id",
    stratum_field: str = "task",
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Bootstrap a paired difference by parent/document, stratified by task.

    Each callback receives the same resampled records, preserving pairing.  A
    parent and all of its rows are sampled as one cluster.  Parent identifiers
    are scoped by stratum, which matches PIIMB's task-level aggregation.
    """

    rows = _records(records)
    if not rows:
        raise ValueError("bootstrap requires at least one record")
    _positive_integer(replicates, "replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    confidence = _finite_float(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if not callable(baseline_statistic) or not callable(comparison_statistic):
        raise TypeError("bootstrap statistics must be callable")

    grouped: dict[tuple[Any, Any], list[Record]] = defaultdict(list)
    for row in rows:
        parent = _identifier(row, parent_field)
        stratum = _identifier(row, stratum_field)
        grouped[(stratum, parent)].append(row)

    strata: dict[Any, list[tuple[Record, ...]]] = defaultdict(list)
    for (stratum, _parent), group_rows in grouped.items():
        strata[stratum].append(tuple(sorted(group_rows, key=_canonical_row)))
    ordered_strata = [
        (
            stratum,
            sorted(groups, key=lambda group: tuple(_canonical_row(row) for row in group)),
        )
        for stratum, groups in sorted(strata.items(), key=lambda item: _stable_key(item[0]))
    ]
    ordered_rows = tuple(
        row for _stratum, groups in ordered_strata for group in groups for row in group
    )

    baseline_estimate = _statistic(baseline_statistic, ordered_rows, "baseline_statistic")
    comparison_estimate = _statistic(comparison_statistic, ordered_rows, "comparison_statistic")
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(replicates):
        sample: list[Record] = []
        for _stratum, groups in ordered_strata:
            for _ in range(len(groups)):
                sample.extend(rng.choice(groups))
        baseline = _statistic(baseline_statistic, sample, "baseline_statistic")
        comparison = _statistic(comparison_statistic, sample, "comparison_statistic")
        differences.append(comparison - baseline)

    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        baseline_estimate=baseline_estimate,
        comparison_estimate=comparison_estimate,
        difference=comparison_estimate - baseline_estimate,
        lower=_percentile(differences, tail),
        upper=_percentile(differences, 1.0 - tail),
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )


def paired_mean_bootstrap(
    records: Sequence[Record],
    *,
    baseline_field: str,
    comparison_field: str,
    parent_field: str = "parent_id",
    stratum_field: str = "task",
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> BootstrapInterval:
    """Convenience paired bootstrap for two numeric per-row fields."""

    def field_mean(rows: Sequence[Record], field: str) -> float:
        return statistics.fmean(_numeric_field(row, field) for row in rows)

    return paired_stratified_bootstrap(
        records,
        baseline_statistic=lambda rows: field_mean(rows, baseline_field),
        comparison_statistic=lambda rows: field_mean(rows, comparison_field),
        parent_field=parent_field,
        stratum_field=stratum_field,
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )


def pareto_front(
    records: Sequence[Record],
    *,
    minimize: Sequence[str] = (),
    maximize: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return deterministic nondominated rows for mixed-direction objectives."""

    rows = _records(records)
    minimize_fields = _objective_fields(minimize, name="minimize")
    maximize_fields = _objective_fields(maximize, name="maximize")
    if not minimize_fields and not maximize_fields:
        raise ValueError("at least one Pareto objective is required")
    overlap = set(minimize_fields) & set(maximize_fields)
    if overlap:
        raise ValueError(f"objectives cannot be both minimized and maximized: {sorted(overlap)}")

    unique_rows: dict[str, Record] = {}
    for row in rows:
        canonical = _canonical_row(row)
        unique_rows.setdefault(canonical, row)
        for field in (*minimize_fields, *maximize_fields):
            _numeric_field(row, field)
    candidates = tuple(unique_rows.values())

    def objective(row: Record) -> tuple[float, ...]:
        return tuple(_numeric_field(row, field) for field in minimize_fields) + tuple(
            -_numeric_field(row, field) for field in maximize_fields
        )

    front: list[Record] = []
    for candidate in candidates:
        candidate_values = objective(candidate)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_values = objective(other)
            if all(
                left <= right for left, right in zip(other_values, candidate_values, strict=True)
            ) and any(
                left < right for left, right in zip(other_values, candidate_values, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(
        dict(row) for row in sorted(front, key=lambda row: (objective(row), _canonical_row(row)))
    )


def _records(values: Sequence[Record]) -> tuple[Record, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("records must be a sequence")
    rows = tuple(values)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("records must contain only mappings")
    return rows


def _identifier(row: Record, field: str) -> Any:
    if field not in row:
        raise ValueError(f"record is missing {field}")
    value = row[field]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{field} must identify a non-empty bootstrap unit")
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{field} must be hashable") from error
    return value


def _statistic(function: RecordStatistic, rows: Sequence[Record], name: str) -> float:
    return _finite_float(function(rows), name)


def _numeric_field(row: Record, field: str) -> float:
    if field not in row:
        raise ValueError(f"record is missing {field}")
    return _finite_float(row[field], field)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = quantile * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _objective_fields(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{name} objectives must be a sequence")
    fields = tuple(values)
    if any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError(f"{name} objectives must be non-blank strings")
    if len(fields) != len(set(fields)):
        raise ValueError(f"{name} objectives must not contain duplicates")
    return fields


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _canonical_row(row: Record) -> str:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=repr)


__all__ = [
    "DEFAULT_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "BootstrapInterval",
    "paired_mean_bootstrap",
    "paired_stratified_bootstrap",
    "pareto_front",
]
