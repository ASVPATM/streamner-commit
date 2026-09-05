"""Offline descriptive metrics for validating the streaming-revision premise."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

type JSONMapping = Mapping[str, Any]


def raw_revision_metrics(span_updates: Sequence[JSONMapping]) -> dict[str, Any]:
    """Summarize raw update and top-probability revision behavior."""

    updates = _mapping_rows(span_updates, name="span_updates")
    _validate_step_order(updates, name="span_updates")
    boundaries = {_word_boundary(update) for update in updates}
    rescores = [update for update in updates if update.get("update_kind") == "rescore"]
    rescored_boundaries = {_word_boundary(update) for update in rescores}

    movements: list[float] = []
    rescores_without_previous = 0
    for update in updates:
        kind = update.get("update_kind")
        if kind not in {"new", "rescore", "full"}:
            raise ValueError("update_kind must be one of: new, rescore, full")
        previous = _optional_finite_number(update.get("previous_top_probability"))
        delta = _optional_finite_number(update.get("top_probability_delta"))
        current = _optional_finite_number(update.get("top_probability"))
        if previous is not None:
            if current is None:
                raise ValueError("top_probability is required when previous_top_probability is set")
            derived_delta = current - previous
            if delta is not None and not math.isclose(delta, derived_delta, abs_tol=1e-6):
                raise ValueError(
                    "top_probability_delta is inconsistent with the probability values"
                )
            movements.append(abs(derived_delta))
        elif delta is not None:
            movements.append(abs(delta))
        elif kind == "rescore":
            rescores_without_previous += 1

    return {
        "span_update_count": len(updates),
        "rescore_update_count": len(rescores),
        "rescore_update_rate": _ratio(len(rescores), len(updates)),
        "observed_boundary_count": len(boundaries),
        "rescored_boundary_count": len(rescored_boundaries),
        "rescored_boundary_rate": _ratio(len(rescored_boundaries), len(boundaries)),
        "rescores_without_previous_probability": rescores_without_previous,
        "probability_movement": _distribution(movements, values_key="absolute_values"),
    }


def decoded_snapshot_churn(snapshots: Sequence[JSONMapping]) -> dict[str, Any]:
    """Measure entity-identity symmetric differences between adjacent snapshots."""

    rows = _snapshot_rows(snapshots)
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:], strict=False):
        previous_ids = _entity_identities(previous)
        current_ids = _entity_identities(current)
        symmetric_difference = previous_ids ^ current_ids
        union = previous_ids | current_ids
        transitions.append(
            {
                "from_step": _integer_field(previous, "step"),
                "to_step": _integer_field(current, "step"),
                "symmetric_difference_count": len(symmetric_difference),
                "union_count": len(union),
                "normalized_churn": _ratio(len(symmetric_difference), len(union)),
            }
        )
    return _churn_summary(transitions)


def boundary_extension_metrics(snapshots: Sequence[JSONMapping]) -> dict[str, Any]:
    """Detect later public entities with the same start/label and a larger end."""

    rows = _snapshot_rows(snapshots)
    best_by_family: dict[tuple[int, str], dict[str, Any]] = {}
    cases: list[dict[str, Any]] = []

    for snapshot in rows:
        step = _integer_field(snapshot, "step")
        current_by_family: dict[tuple[int, str], dict[str, Any]] = {}
        for entity in _entity_rows(snapshot):
            identity = _entity_identity(entity)
            start_char, end_char, label = identity
            family = start_char, label
            candidate = {
                "step": step,
                "start_char": start_char,
                "end_char": end_char,
                "label": label,
                "text": _entity_text(entity, snapshot),
            }
            existing = current_by_family.get(family)
            if existing is None or end_char > existing["end_char"]:
                current_by_family[family] = candidate

        for family in sorted(current_by_family):
            candidate = current_by_family[family]
            previous = best_by_family.get(family)
            if previous is not None and candidate["end_char"] > previous["end_char"]:
                cases.append(
                    {
                        "from_step": previous["step"],
                        "to_step": candidate["step"],
                        "start_char": candidate["start_char"],
                        "from_end_char": previous["end_char"],
                        "to_end_char": candidate["end_char"],
                        "label": candidate["label"],
                        "from_text": previous["text"],
                        "to_text": candidate["text"],
                    }
                )
            if previous is None or candidate["end_char"] > previous["end_char"]:
                best_by_family[family] = candidate

    return {"count": len(cases), "cases": cases}


def revision_horizon_metrics(span_updates: Sequence[JSONMapping]) -> dict[str, Any]:
    """Report each exact boundary's horizon at its last actual rescore event.

    Model-visible word counts are read directly from observer records. They are
    intentionally not recomputed with the benchmark's whitespace-run chunker.
    """

    updates = _mapping_rows(span_updates, name="span_updates")
    _validate_step_order(updates, name="span_updates")
    last_rescore: dict[tuple[int, int], dict[str, Any]] = {}

    for update in updates:
        if update.get("update_kind") != "rescore":
            continue
        start_word, end_word = _word_boundary(update)
        visible_word_count = _integer_field(update, "visible_word_count")
        if visible_word_count <= 0 or end_word >= visible_word_count:
            raise ValueError("rescore boundary must refer to a visible model word")
        latest_visible_word = visible_word_count - 1
        horizon = latest_visible_word - end_word
        recorded_horizon = update.get("tail_distance_words")
        if (
            recorded_horizon is not None
            and _integer(recorded_horizon, "tail_distance_words") != horizon
        ):
            raise ValueError("tail_distance_words disagrees with raw model word coordinates")
        last_rescore[(start_word, end_word)] = {
            "start_word": start_word,
            "end_word": end_word,
            "last_rescore_step": _integer_field(update, "step"),
            "last_rescore_visible_word": latest_visible_word,
            "revision_horizon_words": horizon,
        }

    records = [last_rescore[boundary] for boundary in sorted(last_rescore)]
    return _horizon_summary(records)


def analyze_premise_trace(trace: JSONMapping) -> dict[str, Any]:
    """Compute all Phase 4 descriptive metrics for one serialized trace."""

    if not isinstance(trace, Mapping):
        raise TypeError("trace must be a mapping")
    example_id = trace.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("trace example_id must be a non-blank string")
    updates = trace.get("span_updates")
    snapshots = trace.get("steps", trace.get("snapshots"))
    if isinstance(updates, str | bytes) or not isinstance(updates, Sequence):
        raise TypeError("trace span_updates must be a sequence")
    if isinstance(snapshots, str | bytes) or not isinstance(snapshots, Sequence):
        raise TypeError("trace steps/snapshots must be a sequence")

    extensions = boundary_extension_metrics(snapshots)
    extensions["cases"] = [{"example_id": example_id, **case} for case in extensions["cases"]]
    churn = decoded_snapshot_churn(snapshots)
    churn["transitions"] = [
        {"example_id": example_id, **transition} for transition in churn["transitions"]
    ]
    horizon = revision_horizon_metrics(updates)
    horizon["per_boundary"] = [
        {"example_id": example_id, **record} for record in horizon["per_boundary"]
    ]
    return {
        "example_id": example_id,
        "raw_revisions": raw_revision_metrics(updates),
        "decoded_snapshot_churn": churn,
        "boundary_extensions": extensions,
        "revision_horizon": horizon,
    }


def aggregate_premise_reports(reports: Sequence[JSONMapping]) -> dict[str, Any]:
    """Aggregate per-trace reports without crossing example boundaries."""

    rows = _mapping_rows(reports, name="reports")
    raw_rows = [_required_mapping(row, "raw_revisions") for row in rows]
    churn_rows = [_required_mapping(row, "decoded_snapshot_churn") for row in rows]
    extension_rows = [_required_mapping(row, "boundary_extensions") for row in rows]
    horizon_rows = [_required_mapping(row, "revision_horizon") for row in rows]

    update_count = sum(_integer_field(row, "span_update_count") for row in raw_rows)
    rescore_count = sum(_integer_field(row, "rescore_update_count") for row in raw_rows)
    boundary_count = sum(_integer_field(row, "observed_boundary_count") for row in raw_rows)
    rescored_boundary_count = sum(
        _integer_field(row, "rescored_boundary_count") for row in raw_rows
    )
    missing_previous = sum(
        _integer_field(row, "rescores_without_previous_probability") for row in raw_rows
    )
    movement_values = [
        _finite_number(value, "absolute movement")
        for row in raw_rows
        for value in _sequence_field(
            _required_mapping(row, "probability_movement"), "absolute_values"
        )
    ]
    transitions = [
        dict(transition)
        for row in churn_rows
        for transition in _mapping_rows(
            _sequence_field(row, "transitions"), name="churn transitions"
        )
    ]
    extension_cases = [
        dict(case)
        for row in extension_rows
        for case in _mapping_rows(_sequence_field(row, "cases"), name="extension cases")
    ]
    horizon_records = [
        dict(record)
        for row in horizon_rows
        for record in _mapping_rows(_sequence_field(row, "per_boundary"), name="horizon records")
    ]

    return {
        "trace_count": len(rows),
        "raw_revisions": {
            "span_update_count": update_count,
            "rescore_update_count": rescore_count,
            "rescore_update_rate": _ratio(rescore_count, update_count),
            "observed_boundary_count": boundary_count,
            "rescored_boundary_count": rescored_boundary_count,
            "rescored_boundary_rate": _ratio(rescored_boundary_count, boundary_count),
            "rescores_without_previous_probability": missing_previous,
            "probability_movement": _distribution(movement_values, values_key="absolute_values"),
        },
        "decoded_snapshot_churn": _churn_summary(transitions),
        "boundary_extensions": {"count": len(extension_cases), "cases": extension_cases},
        "revision_horizon": _horizon_summary(horizon_records),
    }


@dataclass(frozen=True, slots=True)
class RateMetric:
    """An explicit numerator/denominator rate with zero-safe division."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        _integer(self.numerator, "numerator")
        _integer(self.denominator, "denominator")
        if self.numerator > self.denominator:
            raise ValueError("numerator cannot exceed denominator")

    @property
    def rate(self) -> float:
        return _ratio(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class PrematurePrefixComparison:
    """Gold-relative and cold-model-relative prefix rates kept separate."""

    gold_relative: RateMetric
    silver_relative: RateMetric


@dataclass(frozen=True, slots=True)
class CommitmentOutcomes:
    """Irreversible commitment errors plus gold entities missed by stream end."""

    commitments: int
    correct_commitments: int
    wrong_commitments: int
    gold_entities: int
    missed_entities: int

    @property
    def precision(self) -> float:
        return _ratio(self.correct_commitments, self.commitments)

    @property
    def wrong_commitment_rate(self) -> float:
        return _ratio(self.wrong_commitments, self.commitments)

    @property
    def missed_entity_rate(self) -> float:
        return _ratio(self.missed_entities, self.gold_entities)


@dataclass(frozen=True, slots=True)
class BlockedRevisionMetrics:
    """Blocked conflict counts with the three required normalizations."""

    blocked_revisions: int
    example_count: int
    commitment_count: int

    @property
    def per_example(self) -> float:
        return _ratio(self.blocked_revisions, self.example_count)

    @property
    def per_committed_entity(self) -> float:
        return _ratio(self.blocked_revisions, self.commitment_count)

    @property
    def per_100_commitments(self) -> float:
        return 100.0 * self.per_committed_entity


@dataclass(frozen=True, slots=True)
class TrajectoryStabilityStats:
    """Descriptive stability statistics for one exact candidate trajectory."""

    observation_count: int
    variance: float
    max_absolute_delta: float
    mean_absolute_delta: float
    delta_sign_change_count: int
    rank_change_count: int


def premature_prefix_rate(
    commitments: Sequence[JSONMapping],
    reference_entities: Sequence[JSONMapping],
) -> RateMetric:
    """Count strict same-start/same-label truncations against one reference."""

    committed_rows = _mapping_rows(commitments, name="commitments")
    reference_rows = _mapping_rows(reference_entities, name="reference_entities")
    count = sum(
        any(_is_premature_prefix(commitment, reference) for reference in reference_rows)
        for commitment in committed_rows
    )
    return RateMetric(numerator=count, denominator=len(committed_rows))


def premature_prefix_rates(
    commitments: Sequence[JSONMapping],
    *,
    gold_entities: Sequence[JSONMapping],
    silver_entities: Sequence[JSONMapping],
) -> PrematurePrefixComparison:
    """Report gold and cold-final-model prefix definitions independently."""

    return PrematurePrefixComparison(
        gold_relative=premature_prefix_rate(commitments, gold_entities),
        silver_relative=premature_prefix_rate(commitments, silver_entities),
    )


def commitment_outcomes(
    commitments: Sequence[JSONMapping],
    gold_entities: Sequence[JSONMapping],
) -> CommitmentOutcomes:
    """Calculate wrong immutable commitments and gold entities never committed."""

    committed_rows = _mapping_rows(commitments, name="commitments")
    gold_rows = _mapping_rows(gold_entities, name="gold_entities")
    correct = sum(
        any(_same_scoped_entity(commitment, gold) for gold in gold_rows)
        for commitment in committed_rows
    )
    missed = sum(
        not any(_same_scoped_entity(commitment, gold) for commitment in committed_rows)
        for gold in gold_rows
    )
    return CommitmentOutcomes(
        commitments=len(committed_rows),
        correct_commitments=correct,
        wrong_commitments=len(committed_rows) - correct,
        gold_entities=len(gold_rows),
        missed_entities=missed,
    )


def blocked_revision_metrics(
    blocked_revisions: Sequence[JSONMapping],
    *,
    example_count: int,
    commitment_count: int,
) -> BlockedRevisionMetrics:
    """Normalize logged blocked conflicts without inferring hidden events."""

    events = _mapping_rows(blocked_revisions, name="blocked_revisions")
    _integer(example_count, "example_count")
    _integer(commitment_count, "commitment_count")
    return BlockedRevisionMetrics(
        blocked_revisions=len(events),
        example_count=example_count,
        commitment_count=commitment_count,
    )


def boundary_revision_metrics(
    snapshots: Sequence[JSONMapping],
    *,
    entities_field: str = "public_entities",
) -> dict[str, Any]:
    """Track candidate families whose favored end boundary changes.

    Families use ``(example_id, start_char, label)`` when ``example_id`` is
    present and ``(start_char, label)`` otherwise.  Within a snapshot the
    favored boundary is selected deterministically by score, then longer end.
    """

    rows = _mapping_rows(snapshots, name="snapshots")
    _validate_step_order(rows, name="snapshots")
    previous: dict[tuple[Any, int, str], int] = {}
    all_families: set[tuple[Any, int, str]] = set()
    revised_families: set[tuple[Any, int, str]] = set()
    revision_count = 0
    comparison_count = 0
    transitions: list[dict[str, Any]] = []

    for snapshot in rows:
        entity_values = snapshot.get(entities_field)
        entities = _mapping_rows(_as_sequence(entity_values, entities_field), name=entities_field)
        favored: dict[tuple[Any, int, str], tuple[float, int]] = {}
        snapshot_scope = snapshot.get("example_id")
        for entity in entities:
            start, end, label = _entity_identity(entity)
            scope = entity.get("example_id", snapshot_scope)
            family = scope, start, label
            score_value = entity.get("score", entity.get("top_probability", 0.0))
            score = _finite_number(score_value, "entity score")
            current = favored.get(family)
            candidate = score, end
            if current is None or candidate > current:
                favored[family] = candidate
        current_ends = {family: score_end[1] for family, score_end in favored.items()}
        all_families.update(current_ends)
        for family in sorted(previous.keys() & current_ends.keys(), key=repr):
            comparison_count += 1
            from_end = previous[family]
            to_end = current_ends[family]
            if from_end == to_end:
                continue
            revision_count += 1
            revised_families.add(family)
            transitions.append(
                {
                    "step": _integer_field(snapshot, "step"),
                    "example_id": family[0],
                    "start_char": family[1],
                    "label": family[2],
                    "from_end_char": from_end,
                    "to_end_char": to_end,
                }
            )
        previous = current_ends

    return {
        "family_count": len(all_families),
        "revised_family_count": len(revised_families),
        "revised_family_rate": _ratio(len(revised_families), len(all_families)),
        "comparable_transition_count": comparison_count,
        "boundary_revision_count": revision_count,
        "boundary_revision_rate": _ratio(revision_count, comparison_count),
        "transitions": transitions,
    }


def material_revision_horizon_metrics(
    span_updates: Sequence[JSONMapping],
    *,
    minimum_probability_delta: float = 0.0,
) -> dict[str, Any]:
    """Revision horizons using only observed, materially changed rescoring events."""

    minimum = _finite_number(minimum_probability_delta, "minimum_probability_delta")
    if minimum < 0.0:
        raise ValueError("minimum_probability_delta must be nonnegative")
    updates = _mapping_rows(span_updates, name="span_updates")
    _validate_step_order(updates, name="span_updates")
    previous_label: dict[tuple[int, int], str] = {}
    last_material: dict[tuple[int, int], dict[str, Any]] = {}
    for update in updates:
        boundary = _word_boundary(update)
        label_value = update.get("top_label")
        label = label_value if isinstance(label_value, str) else ""
        prior_label = previous_label.get(boundary)
        previous_label[boundary] = label
        if update.get("update_kind") != "rescore":
            continue
        delta_value = update.get("top_probability_delta")
        delta = abs(_finite_number(delta_value, "top_probability_delta"))
        label_changed = bool(prior_label is not None and label and label != prior_label)
        if delta < minimum and not label_changed:
            continue
        visible_word_count = _integer_field(update, "visible_word_count")
        end_word = boundary[1]
        if end_word >= visible_word_count:
            raise ValueError("rescore boundary must refer to a visible model word")
        last_material[boundary] = {
            "start_word": boundary[0],
            "end_word": end_word,
            "last_material_step": _integer_field(update, "step"),
            "last_material_visible_word": visible_word_count - 1,
            "revision_horizon_words": (visible_word_count - 1) - end_word,
        }
    records = [last_material[boundary] for boundary in sorted(last_material)]
    return _horizon_summary(records)


def trajectory_stability_metrics(
    trajectory: Sequence[JSONMapping] | Sequence[float],
    *,
    score_field: str = "top_probability",
    rank_field: str = "rank",
) -> TrajectoryStabilityStats:
    """Compute exact-trajectory variance, movement, direction, and rank churn."""

    if isinstance(trajectory, str | bytes) or not isinstance(trajectory, Sequence):
        raise TypeError("trajectory must be a sequence")
    scores: list[float] = []
    ranks: list[int | None] = []
    for point in trajectory:
        if isinstance(point, Mapping):
            scores.append(_finite_number(point.get(score_field), score_field))
            rank = point.get(rank_field)
            ranks.append(None if rank is None else _integer(rank, rank_field))
        else:
            scores.append(_finite_number(point, "trajectory score"))
            ranks.append(None)
    if any(rank is not None for rank in ranks) and any(rank is None for rank in ranks):
        raise ValueError("rank must be supplied for every trajectory point or none")
    deltas = [current - previous for previous, current in zip(scores, scores[1:], strict=False)]
    absolute_deltas = [abs(delta) for delta in deltas]
    nonzero_signs = [1 if delta > 0 else -1 for delta in deltas if delta != 0.0]
    sign_changes = sum(
        current != previous
        for previous, current in zip(nonzero_signs, nonzero_signs[1:], strict=False)
    )
    rank_values = [rank for rank in ranks if rank is not None]
    rank_changes = sum(
        current != previous for previous, current in zip(rank_values, rank_values[1:], strict=False)
    )
    return TrajectoryStabilityStats(
        observation_count=len(scores),
        variance=statistics.pvariance(scores) if scores else 0.0,
        max_absolute_delta=max(absolute_deltas, default=0.0),
        mean_absolute_delta=statistics.fmean(absolute_deltas) if absolute_deltas else 0.0,
        delta_sign_change_count=sign_changes,
        rank_change_count=rank_changes,
    )


def _is_premature_prefix(commitment: JSONMapping, reference: JSONMapping) -> bool:
    commit_start, commit_end, commit_label = _entity_identity(commitment)
    reference_start, reference_end, reference_label = _entity_identity(reference)
    return (
        _same_scope(commitment, reference)
        and commit_start == reference_start
        and commit_label == reference_label
        and commit_end < reference_end
    )


def _same_scoped_entity(left: JSONMapping, right: JSONMapping) -> bool:
    return _same_scope(left, right) and _entity_identity(left) == _entity_identity(right)


def _same_scope(left: JSONMapping, right: JSONMapping) -> bool:
    left_scope = left.get("example_id")
    right_scope = right.get("example_id")
    return left_scope is None or right_scope is None or left_scope == right_scope


def _as_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _churn_summary(transitions: Sequence[JSONMapping]) -> dict[str, Any]:
    difference_counts = [
        _integer_field(transition, "symmetric_difference_count") for transition in transitions
    ]
    union_counts = [_integer_field(transition, "union_count") for transition in transitions]
    normalized_values = [
        _finite_number(transition.get("normalized_churn"), "normalized_churn")
        for transition in transitions
    ]
    total_difference = sum(difference_counts)
    total_union = sum(union_counts)
    return {
        "transition_count": len(transitions),
        "total_symmetric_difference": total_difference,
        "mean_symmetric_difference": (
            statistics.fmean(difference_counts) if difference_counts else None
        ),
        "micro_normalized_churn": _ratio(total_difference, total_union),
        "mean_normalized_churn": (
            statistics.fmean(normalized_values) if normalized_values else None
        ),
        "transitions": [dict(transition) for transition in transitions],
    }


def _horizon_summary(records: Sequence[JSONMapping]) -> dict[str, Any]:
    horizons = [_integer_field(record, "revision_horizon_words") for record in records]
    summary = _distribution(horizons, values_key="values_words")
    summary["boundary_count"] = summary.pop("count")
    summary["per_boundary"] = [dict(record) for record in records]
    return summary


def _distribution(values: Sequence[float], *, values_key: str) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "maximum": max(normalized) if normalized else None,
        values_key: normalized,
    }


def _snapshot_rows(snapshots: Sequence[JSONMapping]) -> tuple[JSONMapping, ...]:
    rows = _mapping_rows(snapshots, name="snapshots")
    _validate_step_order(rows, name="snapshots")
    return rows


def _validate_step_order(rows: Sequence[JSONMapping], *, name: str) -> None:
    previous = -1
    for row in rows:
        step = _integer_field(row, "step")
        if step < previous:
            raise ValueError(f"{name} steps must be nondecreasing")
        previous = step


def _entity_rows(snapshot: JSONMapping) -> tuple[JSONMapping, ...]:
    return _mapping_rows(_sequence_field(snapshot, "public_entities"), name="public_entities")


def _entity_identities(snapshot: JSONMapping) -> set[tuple[int, int, str]]:
    return {_entity_identity(entity) for entity in _entity_rows(snapshot)}


def _entity_identity(entity: JSONMapping) -> tuple[int, int, str]:
    start = _alternate_integer_field(entity, "start_char", "start")
    end = _alternate_integer_field(entity, "end_char", "end")
    label = entity.get("label")
    if end <= start:
        raise ValueError("public entity end must be greater than start")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("public entity label must be non-blank")
    return start, end, label


def _entity_text(entity: JSONMapping, snapshot: JSONMapping) -> str:
    text = entity.get("text")
    if isinstance(text, str):
        return text
    accumulated = snapshot.get("accumulated_text")
    if not isinstance(accumulated, str):
        return ""
    start, end, _ = _entity_identity(entity)
    return accumulated[start:end]


def _word_boundary(update: JSONMapping) -> tuple[int, int]:
    start = _integer_field(update, "start_word")
    end = _integer_field(update, "end_word")
    if end < start:
        raise ValueError("end_word must be greater than or equal to start_word")
    return start, end


def _mapping_rows(values: Sequence[JSONMapping], *, name: str) -> tuple[JSONMapping, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    rows = tuple(values)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{name} must contain only mappings")
    return rows


def _required_mapping(row: JSONMapping, field: str) -> JSONMapping:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence_field(row: JSONMapping, field: str) -> Sequence[Any]:
    value = row.get(field)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _alternate_integer_field(row: JSONMapping, primary: str, alternate: str) -> int:
    value = row.get(primary, row.get(alternate))
    return _integer(value, primary)


def _integer_field(row: JSONMapping, field: str) -> int:
    return _integer(row.get(field), field)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _optional_finite_number(value: Any) -> float | None:
    return None if value is None else _finite_number(value, "probability value")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "BlockedRevisionMetrics",
    "CommitmentOutcomes",
    "PrematurePrefixComparison",
    "RateMetric",
    "TrajectoryStabilityStats",
    "aggregate_premise_reports",
    "analyze_premise_trace",
    "blocked_revision_metrics",
    "boundary_extension_metrics",
    "boundary_revision_metrics",
    "commitment_outcomes",
    "decoded_snapshot_churn",
    "material_revision_horizon_metrics",
    "premature_prefix_rate",
    "premature_prefix_rates",
    "raw_revision_metrics",
    "revision_horizon_metrics",
    "trajectory_stability_metrics",
]
