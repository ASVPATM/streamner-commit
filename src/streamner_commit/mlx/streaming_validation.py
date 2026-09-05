"""Deterministic synthetic streaming parity validation for the native MLX backend.

The input is the timing-free JSON oracle emitted by
``export_streaming_parity_suite.py``.  Runtime objects are consumed structurally so
unit tests need neither MLX nor checkpoint weights.  The report never records
backend timing and is therefore byte-deterministic for equal oracle/model outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STREAMING_VALIDATION_SCHEMA_VERSION = 1
ORACLE_SCHEMA_VERSION = 1
SUPPORTED_CHUNK_UNITS = frozenset({1, 2, 4})


class StreamingValidationError(ValueError):
    """The oracle or MLX result violated the validation contract."""


@dataclass(frozen=True, slots=True)
class StreamingValidationTolerances:
    """Numerical gates and narrowly defined non-material exception bands."""

    minimum_logit_cosine: float = 0.9999
    minimum_logit_l2_norm_for_cosine: float = 1e-3
    maximum_probability_error: float = 5e-3
    maximum_margin_error: float = 5e-3
    maximum_delta_error: float = 5e-3
    top_tie_maximum_logit_margin: float = 0.1
    top_tie_maximum_probability: float = 1e-6
    near_threshold_distance: float = 5e-3

    def __post_init__(self) -> None:
        for name in (
            "minimum_logit_cosine",
            "minimum_logit_l2_norm_for_cosine",
            "maximum_probability_error",
            "maximum_margin_error",
            "maximum_delta_error",
            "top_tie_maximum_logit_margin",
            "top_tie_maximum_probability",
            "near_threshold_distance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, normalized)
        if self.minimum_logit_cosine > 1.0:
            raise ValueError("minimum_logit_cosine must not exceed one")
        for name in (
            "maximum_probability_error",
            "maximum_margin_error",
            "maximum_delta_error",
            "top_tie_maximum_probability",
            "near_threshold_distance",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_logit_cosine": self.minimum_logit_cosine,
            "minimum_logit_l2_norm_for_cosine": self.minimum_logit_l2_norm_for_cosine,
            "maximum_probability_error": self.maximum_probability_error,
            "maximum_margin_error": self.maximum_margin_error,
            "maximum_delta_error": self.maximum_delta_error,
            "top_tie_maximum_logit_margin": self.top_tie_maximum_logit_margin,
            "top_tie_maximum_probability": self.top_tie_maximum_probability,
            "near_threshold_distance": self.near_threshold_distance,
        }


def _field(value: object, name: str, *, description: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise StreamingValidationError(f"{description} is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise StreamingValidationError(f"{description} is missing {name}")
    return getattr(value, name)


def _optional_field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonblank(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StreamingValidationError(f"{name} must be a nonblank string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StreamingValidationError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StreamingValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise StreamingValidationError(f"{name} must be finite")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise StreamingValidationError(f"{name} must be between zero and one")
    return result


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise StreamingValidationError(f"{name} must be a sequence")
    return value


def _string_tuple(value: object, *, name: str, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(
        _nonblank(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if nonempty and not result:
        raise StreamingValidationError(f"{name} must not be empty")
    return result


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )


def _vector(value: object, *, name: str, probability: bool) -> tuple[float, ...]:
    converter = _probability if probability else _number
    result = tuple(
        converter(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if not result:
        raise StreamingValidationError(f"{name} must not be empty")
    return result


def _boundary(value: object, *, name: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start = value.get("start_word")
        end = value.get("end_word")
    elif hasattr(value, "start_word") and hasattr(value, "end_word"):
        start = _field(value, "start_word", description=name)
        end = _field(value, "end_word", description=name)
    else:
        values = _sequence(value, name=name)
        if len(values) != 2:
            raise StreamingValidationError(f"{name} must contain start and end")
        start, end = values
    normalized_start = _integer(start, name=f"{name}.start")
    normalized_end = _integer(end, name=f"{name}.end")
    if normalized_end < normalized_start:
        raise StreamingValidationError(f"{name} end must not precede start")
    return normalized_start, normalized_end


def _updated_boundary(update: object, *, name: str) -> tuple[int, int]:
    direct = _optional_field(update, "boundary")
    if direct is not None:
        return _boundary(direct, name=f"{name}.boundary")
    return _boundary(
        {
            "start_word": _field(update, "start_word", description=name),
            "end_word": _field(update, "end_word", description=name),
        },
        name=f"{name}.boundary",
    )


def _normalize_schedules(chunk_units: Sequence[int]) -> tuple[int, ...]:
    if isinstance(chunk_units, str | bytes) or not isinstance(chunk_units, Sequence):
        raise TypeError("chunk_units must be a sequence of integers")
    result: set[int] = set()
    for value in chunk_units:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("chunk_units must contain integers")
        if value not in SUPPORTED_CHUNK_UNITS:
            raise StreamingValidationError("chunk_units must contain only 1, 2, or 4")
        result.add(value)
    if not result:
        raise StreamingValidationError("at least one chunk-unit schedule is required")
    return tuple(sorted(result))


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _second_index(values: Sequence[float], top_index: int) -> int | None:
    remaining = [index for index in range(len(values)) if index != top_index]
    return max(remaining, key=values.__getitem__) if remaining else None


@dataclass(slots=True)
class _Numerics:
    vector_count: int = 0
    scalar_count: int = 0
    logit_absolute_sum: float = 0.0
    logit_maximum_error: float = 0.0
    probability_absolute_sum: float = 0.0
    probability_maximum_error: float = 0.0
    margin_absolute_sum: float = 0.0
    margin_maximum_error: float = 0.0
    delta_count: int = 0
    delta_absolute_sum: float = 0.0
    delta_maximum_error: float = 0.0
    previous_count: int = 0
    previous_absolute_sum: float = 0.0
    previous_maximum_error: float = 0.0
    logit_dot: float = 0.0
    reference_logit_square_sum: float = 0.0
    candidate_logit_square_sum: float = 0.0
    top_label_total: int = 0
    top_label_matches: int = 0

    def add_vectors(
        self,
        reference_logits: Sequence[float],
        candidate_logits: Sequence[float],
        reference_probs: Sequence[float],
        candidate_probs: Sequence[float],
    ) -> None:
        if not (
            len(reference_logits)
            == len(candidate_logits)
            == len(reference_probs)
            == len(candidate_probs)
        ):
            raise StreamingValidationError("score vectors have different class widths")
        self.vector_count += 1
        self.scalar_count += len(reference_logits)
        for reference, candidate in zip(reference_logits, candidate_logits, strict=True):
            error = abs(reference - candidate)
            self.logit_absolute_sum += error
            self.logit_maximum_error = max(self.logit_maximum_error, error)
            self.logit_dot += reference * candidate
            self.reference_logit_square_sum += reference * reference
            self.candidate_logit_square_sum += candidate * candidate
        for reference, candidate in zip(reference_probs, candidate_probs, strict=True):
            error = abs(reference - candidate)
            self.probability_absolute_sum += error
            self.probability_maximum_error = max(self.probability_maximum_error, error)

    def add_margin(self, reference: float, candidate: float) -> None:
        error = abs(reference - candidate)
        self.margin_absolute_sum += error
        self.margin_maximum_error = max(self.margin_maximum_error, error)

    def add_delta(self, reference: float, candidate: float) -> None:
        error = abs(reference - candidate)
        self.delta_count += 1
        self.delta_absolute_sum += error
        self.delta_maximum_error = max(self.delta_maximum_error, error)

    def add_previous(self, reference: float, candidate: float) -> None:
        error = abs(reference - candidate)
        self.previous_count += 1
        self.previous_absolute_sum += error
        self.previous_maximum_error = max(self.previous_maximum_error, error)

    def merge(self, other: _Numerics) -> None:
        for name in (
            "vector_count",
            "scalar_count",
            "logit_absolute_sum",
            "probability_absolute_sum",
            "margin_absolute_sum",
            "delta_count",
            "delta_absolute_sum",
            "previous_count",
            "previous_absolute_sum",
            "logit_dot",
            "reference_logit_square_sum",
            "candidate_logit_square_sum",
            "top_label_total",
            "top_label_matches",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.logit_maximum_error = max(self.logit_maximum_error, other.logit_maximum_error)
        self.probability_maximum_error = max(
            self.probability_maximum_error, other.probability_maximum_error
        )
        self.margin_maximum_error = max(self.margin_maximum_error, other.margin_maximum_error)
        self.delta_maximum_error = max(self.delta_maximum_error, other.delta_maximum_error)
        self.previous_maximum_error = max(self.previous_maximum_error, other.previous_maximum_error)

    @property
    def cosine_similarity(self) -> float | None:
        denominator = math.sqrt(self.reference_logit_square_sum * self.candidate_logit_square_sum)
        if denominator == 0.0:
            return (
                1.0 if self.reference_logit_square_sum == self.candidate_logit_square_sum else None
            )
        return max(-1.0, min(1.0, self.logit_dot / denominator))

    def passes(self, tolerances: StreamingValidationTolerances) -> bool:
        reference_norm = math.sqrt(self.reference_logit_square_sum)
        cosine = self.cosine_similarity
        cosine_pass = reference_norm < tolerances.minimum_logit_l2_norm_for_cosine or (
            cosine is not None and cosine >= tolerances.minimum_logit_cosine
        )
        return (
            cosine_pass
            and self.probability_maximum_error <= tolerances.maximum_probability_error
            and self.margin_maximum_error <= tolerances.maximum_margin_error
            and self.delta_maximum_error <= tolerances.maximum_delta_error
        )

    def to_dict(self, tolerances: StreamingValidationTolerances) -> dict[str, Any]:
        return {
            "vector_count": self.vector_count,
            "scalar_count": self.scalar_count,
            "logit_mean_absolute_error": (
                self.logit_absolute_sum / self.scalar_count if self.scalar_count else 0.0
            ),
            "logit_maximum_absolute_error": self.logit_maximum_error,
            "logit_cosine_similarity": self.cosine_similarity,
            "reference_logit_l2_norm": math.sqrt(self.reference_logit_square_sum),
            "probability_mean_absolute_error": (
                self.probability_absolute_sum / self.scalar_count if self.scalar_count else 0.0
            ),
            "probability_maximum_absolute_error": self.probability_maximum_error,
            "margin_mean_absolute_error": (
                self.margin_absolute_sum / self.vector_count if self.vector_count else 0.0
            ),
            "margin_maximum_absolute_error": self.margin_maximum_error,
            "previous_probability_mean_absolute_error": (
                self.previous_absolute_sum / self.previous_count if self.previous_count else 0.0
            ),
            "previous_probability_maximum_absolute_error": self.previous_maximum_error,
            "delta_mean_absolute_error": (
                self.delta_absolute_sum / self.delta_count if self.delta_count else 0.0
            ),
            "delta_maximum_absolute_error": self.delta_maximum_error,
            "top_label_matches": self.top_label_matches,
            "top_label_total": self.top_label_total,
            "top_label_agreement": (
                self.top_label_matches / self.top_label_total if self.top_label_total else 1.0
            ),
            "within_tolerances": self.passes(tolerances),
        }


type _History = dict[tuple[int, int], tuple[tuple[float, ...], tuple[float, ...]]]


def _near_top_tie(
    logits: Sequence[float],
    probs: Sequence[float],
    tolerances: StreamingValidationTolerances,
) -> tuple[bool, float, float]:
    # The pinned observer ranks sigmoid probabilities.  At extreme negative logits
    # several values can underflow to an exact zero, in which case stable argmax picks
    # the first class even if a different raw logit is mathematically larger.
    top = _argmax(probs)
    second = _second_index(logits, top)
    logit_margin = logits[top] - (logits[second] if second is not None else logits[top])
    top_probability = probs[top]
    return (
        logit_margin <= tolerances.top_tie_maximum_logit_margin
        and top_probability <= tolerances.top_tie_maximum_probability,
        logit_margin,
        top_probability,
    )


def _compare_update(
    expected: object,
    candidate: object,
    *,
    labels: tuple[str, ...],
    reference_history: _History,
    candidate_history: _History,
    numerics: _Numerics,
    tolerances: StreamingValidationTolerances,
) -> tuple[list[str], list[dict[str, Any]]]:
    boundary = _updated_boundary(expected, name="reference update")
    if _updated_boundary(candidate, name="MLX update") != boundary:
        raise StreamingValidationError("matched updates disagree on their boundary")
    reference_logits = _vector(
        _field(expected, "logits", description="reference update"),
        name="reference logits",
        probability=False,
    )
    reference_probs = _vector(
        _field(expected, "probs", description="reference update"),
        name="reference probabilities",
        probability=True,
    )
    candidate_logits = _vector(
        _field(candidate, "logits", description="MLX update"),
        name="MLX logits",
        probability=False,
    )
    candidate_probs = _vector(
        _field(candidate, "probs", description="MLX update"),
        name="MLX probabilities",
        probability=True,
    )
    if len(reference_logits) != len(labels):
        raise StreamingValidationError("reference update class width differs from ordered labels")
    numerics.add_vectors(reference_logits, candidate_logits, reference_probs, candidate_probs)

    reasons: list[str] = []
    categories: list[dict[str, Any]] = []
    expected_kind = _nonblank(
        _field(expected, "update_kind", description="reference update"),
        name="reference update kind",
    )
    candidate_kind_value = _optional_field(candidate, "update_kind")
    candidate_kind = (
        _nonblank(candidate_kind_value, name="MLX update kind")
        if candidate_kind_value is not None
        else ("rescore" if boundary in candidate_history else "new")
    )
    if candidate_kind != expected_kind:
        reasons.append(f"update_kind:{boundary[0]}:{boundary[1]}")

    reference_top = _argmax(reference_probs)
    candidate_top = _argmax(candidate_probs)
    expected_top_index = _integer(
        _field(expected, "top_label_index", description="reference update"),
        name="reference top label index",
    )
    expected_top_label = _nonblank(
        _field(expected, "top_label", description="reference update"),
        name="reference top label",
    )
    if expected_top_index != reference_top or expected_top_label != labels[reference_top]:
        raise StreamingValidationError("reference update top-label metadata is inconsistent")
    numerics.top_label_total += 1
    if candidate_top == reference_top:
        numerics.top_label_matches += 1
    else:
        near_tie, reference_logit_margin, reference_top_probability = _near_top_tie(
            reference_logits,
            reference_probs,
            tolerances,
        )
        detail = {
            "category": "near_top_tie" if near_tie else "material_top_label_mismatch",
            "boundary": list(boundary),
            "reference_label": labels[reference_top],
            "mlx_label": labels[candidate_top],
            "reference_logit_margin": reference_logit_margin,
            "reference_top_probability": reference_top_probability,
        }
        if near_tie:
            categories.append(detail)
        else:
            reasons.append(f"top_label:{boundary[0]}:{boundary[1]}")
            categories.append(detail)

    reference_second = _second_index(reference_probs, reference_top)
    candidate_second = _second_index(candidate_probs, candidate_top)
    reference_margin = reference_probs[reference_top] - (
        reference_probs[reference_second] if reference_second is not None else 0.0
    )
    candidate_margin = candidate_probs[candidate_top] - (
        candidate_probs[candidate_second] if candidate_second is not None else 0.0
    )
    expected_margin = _number(
        _field(expected, "label_margin", description="reference update"),
        name="reference label margin",
    )
    if not math.isclose(expected_margin, reference_margin, rel_tol=1e-6, abs_tol=1e-12):
        raise StreamingValidationError("reference update label margin is inconsistent")
    numerics.add_margin(reference_margin, candidate_margin)

    reference_previous = reference_history.get(boundary)
    candidate_previous = candidate_history.get(boundary)
    expected_previous_value = _field(
        expected,
        "previous_top_probability",
        description="reference update",
    )
    expected_delta_value = _field(
        expected,
        "top_probability_delta",
        description="reference update",
    )
    if reference_previous is None:
        if expected_previous_value is not None or expected_delta_value is not None:
            raise StreamingValidationError("new reference boundary unexpectedly has delta metadata")
        if candidate_previous is not None:
            reasons.append(f"history_presence:{boundary[0]}:{boundary[1]}")
    else:
        if expected_previous_value is None or expected_delta_value is None:
            raise StreamingValidationError("rescored reference boundary lacks delta metadata")
        reference_previous_top = max(reference_previous[1])
        candidate_previous_top = max(candidate_previous[1]) if candidate_previous else None
        expected_previous = _probability(
            expected_previous_value,
            name="reference previous top probability",
        )
        expected_delta = _number(expected_delta_value, name="reference top probability delta")
        reference_delta = reference_probs[reference_top] - reference_previous_top
        if not math.isclose(
            expected_previous, reference_previous_top, rel_tol=1e-6, abs_tol=1e-12
        ) or not math.isclose(expected_delta, reference_delta, rel_tol=1e-6, abs_tol=1e-12):
            raise StreamingValidationError("reference update delta metadata is inconsistent")
        if candidate_previous_top is None:
            reasons.append(f"history_presence:{boundary[0]}:{boundary[1]}")
        else:
            candidate_delta = candidate_probs[candidate_top] - candidate_previous_top
            numerics.add_previous(reference_previous_top, candidate_previous_top)
            numerics.add_delta(reference_delta, candidate_delta)

    reference_history[boundary] = (reference_logits, reference_probs)
    candidate_history[boundary] = (candidate_logits, candidate_probs)
    return reasons, categories


def _entity_record(value: object, *, name: str) -> dict[str, Any]:
    start = _integer(_field(value, "start_char", description=name), name=f"{name}.start_char")
    end = _integer(_field(value, "end_char", description=name), name=f"{name}.end_char")
    if end <= start:
        raise StreamingValidationError(f"{name} has an invalid character span")
    label = _nonblank(_field(value, "label", description=name), name=f"{name}.label")
    text = _field(value, "text", description=name)
    if not isinstance(text, str) or len(text) != end - start:
        raise StreamingValidationError(f"{name}.text does not match its character span length")
    score = _probability(_field(value, "score", description=name), name=f"{name}.score")
    return {"start_char": start, "end_char": end, "label": label, "text": text, "score": score}


def _entity_map(values: object, *, name: str) -> dict[tuple[int, int, str, str], float]:
    result: dict[tuple[int, int, str, str], float] = {}
    for index, value in enumerate(_sequence(values, name=name)):
        record = _entity_record(value, name=f"{name}[{index}]")
        identity = (
            record["start_char"],
            record["end_char"],
            record["label"],
            record["text"],
        )
        if identity in result:
            raise StreamingValidationError(f"{name} contains a duplicate entity identity")
        result[identity] = record["score"]
    return result


def _identity_row(identity: tuple[int, int, str, str]) -> list[Any]:
    return [identity[0], identity[1], identity[2], identity[3]]


def _compare_entities(
    reference_values: object,
    candidate_values: object,
    *,
    threshold: float,
    tolerances: StreamingValidationTolerances,
) -> dict[str, Any]:
    reference = _entity_map(reference_values, name="reference entities")
    candidate = _entity_map(candidate_values, name="MLX entities")
    reference_identities = set(reference)
    candidate_identities = set(candidate)
    unmatched_reference = sorted(reference_identities - candidate_identities)
    unmatched_candidate = sorted(candidate_identities - reference_identities)
    near_threshold: list[dict[str, Any]] = []
    material: list[dict[str, Any]] = []
    for source, identities, scores in (
        ("reference_only", unmatched_reference, reference),
        ("mlx_only", unmatched_candidate, candidate),
    ):
        for identity in identities:
            score = scores[identity]
            detail = {
                "category": "near_threshold"
                if abs(score - threshold) <= tolerances.near_threshold_distance
                else "material_decoded_identity_mismatch",
                "source": source,
                "identity": _identity_row(identity),
                "score": score,
                "distance_from_threshold": abs(score - threshold),
            }
            (near_threshold if detail["category"] == "near_threshold" else material).append(detail)
    common_errors = [
        abs(reference[identity] - candidate[identity])
        for identity in sorted(reference_identities & candidate_identities)
    ]
    return {
        "identity_match": reference_identities == candidate_identities,
        "reference": [_identity_row(identity) for identity in sorted(reference_identities)],
        "mlx": [_identity_row(identity) for identity in sorted(candidate_identities)],
        "matched_score_maximum_error": max(common_errors, default=0.0),
        "near_threshold": near_threshold,
        "material": material,
    }


def _candidate_updates(result: object) -> tuple[object, ...]:
    return tuple(
        _sequence(
            _field(result, "span_updates", description="MLX append result"),
            name="MLX span updates",
        )
    )


def _compare_step(
    expected: Mapping[str, Any],
    result: object,
    *,
    labels: tuple[str, ...],
    threshold: float,
    reference_history: _History,
    candidate_history: _History,
    tolerances: StreamingValidationTolerances,
) -> tuple[dict[str, Any], _Numerics, list[str], list[dict[str, Any]]]:
    step = _integer(expected.get("step"), name="reference step", minimum=1)
    chunk = expected.get("chunk")
    if not isinstance(chunk, str):
        raise StreamingValidationError("reference step chunk must be a string")
    state = _field(result, "state", description="MLX append result")
    expected_words = _string_tuple(expected.get("model_words"), name="reference model words")
    expected_starts = _integer_tuple(expected.get("word_char_starts"), name="reference word starts")
    expected_ends = _integer_tuple(expected.get("word_char_ends"), name="reference word ends")
    expected_labels = _string_tuple(expected.get("labels"), name="reference labels", nonempty=True)
    expected_text = expected.get("accumulated_text")
    if not isinstance(expected_text, str):
        raise StreamingValidationError("reference accumulated text must be a string")

    actual_text = _field(state, "accumulated_text", description="MLX state")
    if not isinstance(actual_text, str):
        raise StreamingValidationError("MLX accumulated text must be a string")
    actual_words = _string_tuple(
        _field(state, "word_tokens", description="MLX state"), name="MLX model words"
    )
    actual_starts = _integer_tuple(
        _field(state, "word_char_starts", description="MLX state"), name="MLX word starts"
    )
    actual_ends = _integer_tuple(
        _field(state, "word_char_ends", description="MLX state"), name="MLX word ends"
    )
    actual_labels = _string_tuple(
        _field(state, "labels", description="MLX state"), name="MLX labels", nonempty=True
    )
    exact = {
        "accumulated_text": actual_text == expected_text,
        "model_words": actual_words == expected_words,
        "word_char_starts": actual_starts == expected_starts,
        "word_char_ends": actual_ends == expected_ends,
        "ordered_labels": actual_labels == expected_labels == labels,
        "visible_char_count": _integer(
            _optional_field(state, "visible_char_count", len(actual_text)),
            name="MLX visible char count",
        )
        == _integer(expected.get("visible_char_count"), name="reference visible char count"),
        "visible_word_count": _integer(
            _optional_field(state, "word_count", len(actual_words)),
            name="MLX visible word count",
        )
        == _integer(expected.get("visible_word_count"), name="reference visible word count"),
    }

    expected_updates_raw = _sequence(expected.get("span_updates"), name="reference span updates")
    expected_updates: dict[tuple[int, int], object] = {}
    for index, update in enumerate(expected_updates_raw):
        boundary = _updated_boundary(update, name=f"reference update {index}")
        if boundary in expected_updates:
            raise StreamingValidationError("reference step contains duplicate updated boundaries")
        expected_updates[boundary] = update
    candidate_updates: dict[tuple[int, int], object] = {}
    for index, update in enumerate(_candidate_updates(result)):
        boundary = _updated_boundary(update, name=f"MLX update {index}")
        if boundary in candidate_updates:
            raise StreamingValidationError("MLX step contains duplicate updated boundaries")
        candidate_updates[boundary] = update
    expected_boundary_list = tuple(
        _boundary(value, name="reference updated boundary")
        for value in _sequence(expected.get("updated_boundaries"), name="updated boundaries")
    )
    if expected_boundary_list != tuple(sorted(expected_updates)):
        raise StreamingValidationError("reference updated-boundary summary is inconsistent")
    exact["updated_boundaries"] = tuple(sorted(candidate_updates)) == expected_boundary_list

    reasons = [f"exact:{name}" for name, matched in exact.items() if not matched]
    categories: list[dict[str, Any]] = []
    numerics = _Numerics()
    for boundary in sorted(set(expected_updates) & set(candidate_updates)):
        update_reasons, update_categories = _compare_update(
            expected_updates[boundary],
            candidate_updates[boundary],
            labels=labels,
            reference_history=reference_history,
            candidate_history=candidate_history,
            numerics=numerics,
            tolerances=tolerances,
        )
        reasons.extend(update_reasons)
        categories.extend(update_categories)
    for boundary in sorted(set(expected_updates) - set(candidate_updates)):
        reasons.append(f"missing_update:{boundary[0]}:{boundary[1]}")
        expected_update = expected_updates[boundary]
        reference_history[boundary] = (
            _vector(
                _field(expected_update, "logits", description="reference update"),
                name="reference logits",
                probability=False,
            ),
            _vector(
                _field(expected_update, "probs", description="reference update"),
                name="reference probabilities",
                probability=True,
            ),
        )
    for boundary in sorted(set(candidate_updates) - set(expected_updates)):
        reasons.append(f"unexpected_update:{boundary[0]}:{boundary[1]}")
        candidate_update = candidate_updates[boundary]
        candidate_history[boundary] = (
            _vector(
                _field(candidate_update, "logits", description="MLX update"),
                name="MLX logits",
                probability=False,
            ),
            _vector(
                _field(candidate_update, "probs", description="MLX update"),
                name="MLX probabilities",
                probability=True,
            ),
        )

    entity_comparison = _compare_entities(
        expected.get("public_entities"),
        _field(result, "public_entities", description="MLX append result"),
        threshold=threshold,
        tolerances=tolerances,
    )
    categories.extend(entity_comparison["near_threshold"])
    if entity_comparison["material"]:
        reasons.append("decoded_identity")
        categories.extend(entity_comparison["material"])
    if not numerics.passes(tolerances):
        reasons.append("numerical_tolerance")

    row = {
        "step": step,
        "chunk": chunk,
        "pass": not reasons,
        "exact": exact,
        "reference_update_count": len(expected_updates),
        "mlx_update_count": len(candidate_updates),
        "numerical": numerics.to_dict(tolerances),
        "decoded": entity_comparison,
        "categorized_disagreements": categories,
        "material_reasons": sorted(set(reasons)),
    }
    return row, numerics, reasons, categories


def _final_score_map(value: object, *, name: str) -> _History:
    result: _History = {}
    for index, row in enumerate(_sequence(value, name=name)):
        boundary = _boundary(row, name=f"{name}[{index}]")
        if boundary in result:
            raise StreamingValidationError(f"{name} contains a duplicate boundary")
        result[boundary] = (
            _vector(
                _field(row, "logits", description=f"{name}[{index}]"),
                name=f"{name}[{index}].logits",
                probability=False,
            ),
            _vector(
                _field(row, "probs", description=f"{name}[{index}]"),
                name=f"{name}[{index}].probs",
                probability=True,
            ),
        )
    return result


def _compare_final(
    expected: Mapping[str, Any],
    last_result: object,
    *,
    labels: tuple[str, ...],
    threshold: float,
    reference_history: _History,
    candidate_history: _History,
    tolerances: StreamingValidationTolerances,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    state = _field(last_result, "state", description="final MLX append result")
    exact = {
        "accumulated_text": _field(state, "accumulated_text", description="MLX state")
        == expected.get("accumulated_text"),
        "model_words": _string_tuple(
            _field(state, "word_tokens", description="MLX state"), name="MLX final words"
        )
        == _string_tuple(expected.get("model_words"), name="reference final words"),
        "word_char_starts": _integer_tuple(
            _field(state, "word_char_starts", description="MLX state"),
            name="MLX final word starts",
        )
        == _integer_tuple(expected.get("word_char_starts"), name="reference final word starts"),
        "word_char_ends": _integer_tuple(
            _field(state, "word_char_ends", description="MLX state"),
            name="MLX final word ends",
        )
        == _integer_tuple(expected.get("word_char_ends"), name="reference final word ends"),
        "ordered_labels": _string_tuple(
            _field(state, "labels", description="MLX state"),
            name="MLX final labels",
            nonempty=True,
        )
        == _string_tuple(expected.get("labels"), name="reference final labels", nonempty=True)
        == labels,
    }
    expected_scores = _final_score_map(expected.get("span_scores"), name="final span scores")
    if expected_scores != reference_history:
        raise StreamingValidationError("reference final score state differs from its updates")
    exact["historical_boundaries"] = tuple(sorted(candidate_history)) == tuple(
        sorted(expected_scores)
    )
    expected_count = _integer(expected.get("span_count"), name="reference final span count")
    actual_count = _integer(
        _optional_field(state, "historical_span_count", len(candidate_history)),
        name="MLX historical span count",
    )
    exact["historical_span_count"] = (
        expected_count == len(expected_scores) == len(candidate_history) == actual_count
    )
    reasons = [f"final_exact:{name}" for name, matched in exact.items() if not matched]
    decoded = _compare_entities(
        expected.get("public_entities"),
        _field(last_result, "public_entities", description="final MLX append result"),
        threshold=threshold,
        tolerances=tolerances,
    )
    categories = list(decoded["near_threshold"])
    if decoded["material"]:
        reasons.append("final_decoded_identity")
        categories.extend(decoded["material"])
    return (
        {
            "pass": not reasons,
            "exact": exact,
            "decoded": decoded,
            "categorized_disagreements": categories,
            "material_reasons": sorted(set(reasons)),
        },
        reasons,
        categories,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamingValidationError(f"{name} must be an object")
    return value


def _validate_oracle(payload: Mapping[str, Any]) -> tuple[str, str, float, tuple[str, ...]]:
    if payload.get("schema_version") != ORACLE_SCHEMA_VERSION:
        raise StreamingValidationError("unsupported streaming oracle schema version")
    if payload.get("kind") != "synthetic_streaming_reference_parity_suite":
        raise StreamingValidationError("unexpected streaming oracle kind")
    model_id = _nonblank(payload.get("model_id"), name="oracle model_id")
    revision = _nonblank(payload.get("model_revision_sha"), name="oracle revision")
    if len(revision) != 40:
        raise StreamingValidationError("oracle revision must be a 40-character SHA")
    threshold = _probability(payload.get("threshold"), name="oracle threshold")
    labels = _string_tuple(payload.get("labels"), name="oracle labels", nonempty=True)
    if len(labels) != len(set(labels)):
        raise StreamingValidationError("oracle labels must be unique and ordered")
    return model_id, revision, threshold, labels


def run_streaming_validation(
    backend: object,
    oracle: Mapping[str, Any],
    *,
    chunk_units: Sequence[int] = (1,),
    tolerances: StreamingValidationTolerances | None = None,
) -> dict[str, Any]:
    """Run selected oracle schedules and return a deterministic parity report."""

    if not isinstance(oracle, Mapping):
        raise TypeError("oracle must be a mapping")
    start_session = getattr(backend, "start_session", None)
    if not callable(start_session):
        raise TypeError("backend must expose start_session")
    schedules = _normalize_schedules(chunk_units)
    limits = tolerances or StreamingValidationTolerances()
    model_id, revision, threshold, labels = _validate_oracle(oracle)
    conditions_raw = _sequence(oracle.get("conditions"), name="oracle conditions")
    by_schedule: dict[int, Mapping[str, Any]] = {}
    for index, raw_condition in enumerate(conditions_raw):
        condition = _mapping(raw_condition, name=f"oracle condition {index}")
        schedule = _integer(condition.get("chunk_units"), name="condition chunk_units", minimum=1)
        if schedule in by_schedule:
            raise StreamingValidationError("oracle contains a duplicate chunk schedule")
        by_schedule[schedule] = condition
    missing = [schedule for schedule in schedules if schedule not in by_schedule]
    if missing:
        raise StreamingValidationError(f"oracle lacks requested chunk schedules: {missing}")

    report_conditions: list[dict[str, Any]] = []
    global_numerics = _Numerics()
    total_cases = 0
    total_steps = 0
    exact_step_count = 0
    material_reason_count = 0
    near_top_tie_count = 0
    near_threshold_count = 0

    for schedule in schedules:
        condition = by_schedule[schedule]
        cases_raw = _sequence(condition.get("cases"), name=f"schedule {schedule} cases")
        condition_cases: list[dict[str, Any]] = []
        condition_pass = True
        for case_index, raw_case in enumerate(cases_raw):
            case = _mapping(raw_case, name=f"schedule {schedule} case {case_index}")
            case_id = _nonblank(case.get("id"), name="case id")
            case_labels = _string_tuple(case.get("labels"), name=f"{case_id} labels", nonempty=True)
            if case_labels != labels:
                raise StreamingValidationError(f"case {case_id!r} labels differ from suite labels")
            steps_raw = _sequence(case.get("steps"), name=f"{case_id} steps")
            expected_chunks = tuple(
                step.get("chunk")
                for step in (_mapping(item, name=f"{case_id} step") for item in steps_raw)
            )
            if tuple(_sequence(case.get("chunks"), name=f"{case_id} chunks")) != expected_chunks:
                raise StreamingValidationError(f"case {case_id!r} chunks differ from step chunks")
            if _integer(case.get("step_count"), name=f"{case_id} step_count") != len(steps_raw):
                raise StreamingValidationError(f"case {case_id!r} step_count is inconsistent")

            session = start_session(
                list(labels),
                threshold=threshold,
                flat_ner=True,
                multi_label=False,
            )
            append = getattr(session, "append", None)
            clear = getattr(session, "clear", None)
            if not callable(append) or not callable(clear):
                raise StreamingValidationError("streaming session must expose append and clear")
            reference_history: _History = {}
            candidate_history: _History = {}
            case_numerics = _Numerics()
            case_steps: list[dict[str, Any]] = []
            case_reasons: list[str] = []
            case_categories: list[dict[str, Any]] = []
            last_result: object | None = None
            try:
                for raw_step in steps_raw:
                    expected_step = _mapping(raw_step, name=f"{case_id} step")
                    chunk = expected_step.get("chunk")
                    if not isinstance(chunk, str):
                        raise StreamingValidationError("reference chunk must be a string")
                    last_result = append(chunk)
                    step_row, step_numerics, step_reasons, step_categories = _compare_step(
                        expected_step,
                        last_result,
                        labels=labels,
                        threshold=threshold,
                        reference_history=reference_history,
                        candidate_history=candidate_history,
                        tolerances=limits,
                    )
                    case_steps.append(step_row)
                    case_numerics.merge(step_numerics)
                    case_reasons.extend(step_reasons)
                    case_categories.extend(step_categories)
                    total_steps += 1
                    exact_step_count += int(all(step_row["exact"].values()))
            finally:
                clear()
            if last_result is None:
                raise StreamingValidationError(f"case {case_id!r} produced no append result")

            expected_final = _mapping(case.get("final_state"), name=f"{case_id} final state")
            final_row, final_reasons, final_categories = _compare_final(
                expected_final,
                last_result,
                labels=labels,
                threshold=threshold,
                reference_history=reference_history,
                candidate_history=candidate_history,
                tolerances=limits,
            )
            case_reasons.extend(final_reasons)
            case_categories.extend(final_categories)
            case_pass = not case_reasons and case_numerics.passes(limits)
            condition_pass &= case_pass
            material_reason_count += len(case_reasons)
            near_top_tie_count += sum(
                item.get("category") == "near_top_tie" for item in case_categories
            )
            near_threshold_count += sum(
                item.get("category") == "near_threshold" for item in case_categories
            )
            global_numerics.merge(case_numerics)
            total_cases += 1
            condition_cases.append(
                {
                    "id": case_id,
                    "pass": case_pass,
                    "step_count": len(case_steps),
                    "numerical": case_numerics.to_dict(limits),
                    "material_reasons": sorted(set(case_reasons)),
                    "categorized_disagreements": case_categories,
                    "steps": case_steps,
                    "final": final_row,
                }
            )
        report_conditions.append(
            {
                "chunk_units": schedule,
                "pass": condition_pass,
                "case_count": len(condition_cases),
                "cases": condition_cases,
            }
        )

    report_pass = all(condition["pass"] for condition in report_conditions)
    return {
        "schema_version": STREAMING_VALIDATION_SCHEMA_VERSION,
        "kind": "mlx_streaming_parity_report",
        "model_id": model_id,
        "model_revision_sha": revision,
        "oracle_schema_version": ORACLE_SCHEMA_VERSION,
        "canonical_threshold": threshold,
        "chunk_unit_schedules": list(schedules),
        "tolerances": limits.to_dict(),
        "pass": report_pass,
        "totals": {
            "condition_count": len(report_conditions),
            "case_count": total_cases,
            "step_count": total_steps,
            "exact_step_count": exact_step_count,
            "material_reason_count": material_reason_count,
            "near_top_tie_count": near_top_tie_count,
            "near_threshold_count": near_threshold_count,
        },
        "numerical": global_numerics.to_dict(limits),
        "conditions": report_conditions,
    }


def deterministic_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical report bytes and reject non-finite output values."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_streaming_validation_report(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one deterministic report and return content-addressed metadata."""

    path = Path(output_path)
    data = deterministic_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


__all__ = [
    "ORACLE_SCHEMA_VERSION",
    "STREAMING_VALIDATION_SCHEMA_VERSION",
    "SUPPORTED_CHUNK_UNITS",
    "StreamingValidationError",
    "StreamingValidationTolerances",
    "deterministic_json_bytes",
    "run_streaming_validation",
    "write_streaming_validation_report",
]
