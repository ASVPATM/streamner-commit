"""Sanitized real-data parity gate for the 100-case PIIMB MLX smoke."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_LICENSE,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SPLIT,
    PIIMB_SUBSET,
    PRIMARY_TASKS,
)

PIIMB_VALIDATION_SCHEMA_VERSION = 1
PIIMB_ORACLE_SCHEMA_VERSION = 1
PIIMB_ORACLE_KIND = "piimb_reference_parity_smoke"
PIIMB_REPORT_KIND = "mlx_piimb_parity_smoke_report"

_NONMATERIAL_CATEGORIES = frozenset(
    {"saturated_near_top_tie", "near_threshold_decoded_identity_mismatch"}
)
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "text",
        "full_text",
        "chunk",
        "chunks",
        "entities",
        "public_entities",
        "annotations",
        "model_words",
        "word_char_starts",
        "word_char_ends",
        "span_updates",
        "span_scores",
        "logits",
        "probs",
    }
)


class PIIMBValidationError(ValueError):
    """The PIIMB oracle, backend result, or report violates its locked contract."""


@dataclass(frozen=True, slots=True)
class PIIMBValidationTolerances:
    """Numerical, near-tie, and acceptance thresholds for the smoke gate."""

    minimum_logit_cosine: float = 0.9999
    minimum_logit_l2_norm_for_cosine: float = 1e-3
    maximum_probability_error: float = 5e-3
    saturated_tie_maximum_logit_margin: float = 0.1
    saturated_tie_maximum_probability: float = 1e-6
    near_threshold_distance: float = 5e-3
    minimum_candidate_top_label_agreement: float = 0.995
    minimum_decoded_identity_agreement: float = 0.99

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
            object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PIIMBValidationError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise PIIMBValidationError(f"{name} must be a sequence")
    return value


def _field(value: object, name: str, *, description: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise PIIMBValidationError(f"{description} is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise PIIMBValidationError(f"{description} is missing {name}")
    return getattr(value, name)


def _nonblank(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PIIMBValidationError(f"{name} must be a nonblank string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PIIMBValidationError(f"{name} must be an integer of at least {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PIIMBValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise PIIMBValidationError(f"{name} must be finite")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise PIIMBValidationError(f"{name} must be between zero and one")
    return result


def _string_tuple(value: object, *, name: str, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(
        _nonblank(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if nonempty and not result:
        raise PIIMBValidationError(f"{name} must not be empty")
    return result


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )


def _vector(value: object, *, name: str, probabilities: bool) -> tuple[float, ...]:
    converter = _probability if probabilities else _number
    result = tuple(
        converter(item, name=f"{name}[{index}]")
        for index, item in enumerate(_sequence(value, name=name))
    )
    if not result:
        raise PIIMBValidationError(f"{name} must not be empty")
    return result


def _boundary(value: object, *, name: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start, end = value.get("start_word"), value.get("end_word")
    elif hasattr(value, "start_word") and hasattr(value, "end_word"):
        start = _field(value, "start_word", description=name)
        end = _field(value, "end_word", description=name)
    else:
        values = _sequence(value, name=name)
        if len(values) != 2:
            raise PIIMBValidationError(f"{name} must contain start and end")
        start, end = values
    normalized = (
        _integer(start, name=f"{name}.start"),
        _integer(end, name=f"{name}.end"),
    )
    if normalized[1] < normalized[0]:
        raise PIIMBValidationError(f"{name} end precedes start")
    return normalized


def _update_boundary(value: object, *, name: str) -> tuple[int, int]:
    if isinstance(value, Mapping) and "boundary" in value:
        return _boundary(value["boundary"], name=f"{name}.boundary")
    if not isinstance(value, Mapping) and hasattr(value, "boundary"):
        return _boundary(value.boundary, name=f"{name}.boundary")
    return _boundary(value, name=name)


def _update_map(values: object, *, name: str) -> dict[tuple[int, int], object]:
    result: dict[tuple[int, int], object] = {}
    for index, value in enumerate(_sequence(values, name=name)):
        boundary = _update_boundary(value, name=f"{name}[{index}]")
        if boundary in result:
            raise PIIMBValidationError(f"{name} contains duplicate boundary {boundary}")
        result[boundary] = value
    return result


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _is_saturated_near_tie(
    logits: Sequence[float],
    probs: Sequence[float],
    limits: PIIMBValidationTolerances,
) -> bool:
    if len(logits) < 2:
        return False
    top = _argmax(probs)
    second = max((index for index in range(len(logits)) if index != top), key=logits.__getitem__)
    return (
        logits[top] - logits[second] <= limits.saturated_tie_maximum_logit_margin
        and probs[top] <= limits.saturated_tie_maximum_probability
    )


@dataclass(slots=True)
class _Stats:
    cases: int = 0
    structural_exact_cases: int = 0
    decoded_identity_exact_cases: int = 0
    decoded_identity_adjusted_cases: int = 0
    candidate_vectors: int = 0
    top_label_matches: int = 0
    adjusted_top_label_matches: int = 0
    scalar_scores: int = 0
    logit_absolute_sum: float = 0.0
    logit_maximum_error: float = 0.0
    probability_absolute_sum: float = 0.0
    probability_maximum_error: float = 0.0
    logit_dot: float = 0.0
    reference_logit_square_sum: float = 0.0
    candidate_logit_square_sum: float = 0.0
    categories: Counter[str] = field(default_factory=Counter)
    category_cases: Counter[str] = field(default_factory=Counter)

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
            raise PIIMBValidationError("reference and MLX score-vector widths differ")
        self.candidate_vectors += 1
        self.scalar_scores += len(reference_logits)
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

    def add_category(self, category: str, *, count: int = 1) -> None:
        self.categories[category] += count

    def finish_case(self, categories: set[str]) -> None:
        self.cases += 1
        self.category_cases.update(categories)

    def merge(self, other: _Stats) -> None:
        for name in (
            "cases",
            "structural_exact_cases",
            "decoded_identity_exact_cases",
            "decoded_identity_adjusted_cases",
            "candidate_vectors",
            "top_label_matches",
            "adjusted_top_label_matches",
            "scalar_scores",
            "logit_absolute_sum",
            "probability_absolute_sum",
            "logit_dot",
            "reference_logit_square_sum",
            "candidate_logit_square_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.logit_maximum_error = max(self.logit_maximum_error, other.logit_maximum_error)
        self.probability_maximum_error = max(
            self.probability_maximum_error, other.probability_maximum_error
        )
        self.categories.update(other.categories)
        self.category_cases.update(other.category_cases)

    @property
    def logit_cosine(self) -> float | None:
        denominator = math.sqrt(self.reference_logit_square_sum * self.candidate_logit_square_sum)
        if denominator == 0.0:
            return (
                1.0 if self.reference_logit_square_sum == self.candidate_logit_square_sum else None
            )
        return max(-1.0, min(1.0, self.logit_dot / denominator))

    @property
    def top_agreement(self) -> float:
        return self.top_label_matches / self.candidate_vectors if self.candidate_vectors else 1.0

    @property
    def adjusted_top_agreement(self) -> float:
        return (
            self.adjusted_top_label_matches / self.candidate_vectors
            if self.candidate_vectors
            else 1.0
        )

    @property
    def decoded_agreement(self) -> float:
        return self.decoded_identity_exact_cases / self.cases if self.cases else 1.0

    def numeric_pass(self, limits: PIIMBValidationTolerances) -> bool:
        reference_norm = math.sqrt(self.reference_logit_square_sum)
        cosine_pass = reference_norm < limits.minimum_logit_l2_norm_for_cosine or (
            self.logit_cosine is not None and self.logit_cosine >= limits.minimum_logit_cosine
        )
        return cosine_pass and self.probability_maximum_error <= limits.maximum_probability_error

    def metrics(self, limits: PIIMBValidationTolerances) -> dict[str, object]:
        return {
            "structural_case_agreement": (
                self.structural_exact_cases / self.cases if self.cases else 1.0
            ),
            "candidate_top_label_agreement": self.top_agreement,
            "candidate_top_label_agreement_adjusted": self.adjusted_top_agreement,
            "decoded_identity_case_agreement": self.decoded_agreement,
            "decoded_identity_case_agreement_adjusted": (
                self.decoded_identity_adjusted_cases / self.cases if self.cases else 1.0
            ),
            "logit_mean_absolute_error": (
                self.logit_absolute_sum / self.scalar_scores if self.scalar_scores else 0.0
            ),
            "logit_maximum_absolute_error": self.logit_maximum_error,
            "logit_cosine_similarity": self.logit_cosine,
            "probability_mean_absolute_error": (
                self.probability_absolute_sum / self.scalar_scores if self.scalar_scores else 0.0
            ),
            "probability_maximum_absolute_error": self.probability_maximum_error,
            "within_numerical_tolerances": self.numeric_pass(limits),
        }

    def counts(self) -> dict[str, int]:
        return {
            "cases": self.cases,
            "structural_exact_cases": self.structural_exact_cases,
            "candidate_vectors": self.candidate_vectors,
            "candidate_top_label_matches": self.top_label_matches,
            "candidate_top_label_matches_adjusted": self.adjusted_top_label_matches,
            "decoded_identity_exact_cases": self.decoded_identity_exact_cases,
            "decoded_identity_adjusted_cases": self.decoded_identity_adjusted_cases,
            "scalar_scores": self.scalar_scores,
        }


def _metadata_checksum(selection: Mapping[str, Any]) -> str:
    fields = {
        name: selection.get(name)
        for name in (
            "uid",
            "source_row_index",
            "task_name",
            "source_dataset",
            "source_uid",
            "parent_id",
            "sentence_index",
            "language",
        )
    }
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _entity_map(values: object, *, name: str) -> dict[tuple[int, int, str], float]:
    result: dict[tuple[int, int, str], float] = {}
    for index, value in enumerate(_sequence(values, name=name)):
        description = f"{name}[{index}]"
        start = _integer(_field(value, "start_char", description=description), name="start_char")
        end = _integer(_field(value, "end_char", description=description), name="end_char")
        if end <= start:
            raise PIIMBValidationError(f"{description} has invalid character offsets")
        label = _nonblank(_field(value, "label", description=description), name="entity label")
        entity_text = _field(value, "text", description=description)
        if not isinstance(entity_text, str) or len(entity_text) != end - start:
            raise PIIMBValidationError(f"{description} text length disagrees with its offsets")
        score = _probability(_field(value, "score", description=description), name="entity score")
        identity = start, end, label
        if identity in result:
            raise PIIMBValidationError(f"{name} contains a duplicate entity identity")
        result[identity] = score
    return result


def _compare_entities(
    reference_values: object,
    candidate_values: object,
    *,
    threshold: float,
    limits: PIIMBValidationTolerances,
    stats: _Stats,
) -> tuple[bool, bool, set[str]]:
    reference = _entity_map(reference_values, name="reference final entities")
    candidate = _entity_map(candidate_values, name="MLX final entities")
    unmatched_reference = set(reference) - set(candidate)
    unmatched_candidate = set(candidate) - set(reference)
    categories: set[str] = set()
    near_threshold_only = True
    for identity in unmatched_reference:
        category = (
            "near_threshold_decoded_identity_mismatch"
            if abs(reference[identity] - threshold) <= limits.near_threshold_distance
            else "material_decoded_identity_mismatch"
        )
        stats.add_category(category)
        categories.add(category)
        near_threshold_only &= category in _NONMATERIAL_CATEGORIES
    for identity in unmatched_candidate:
        category = (
            "near_threshold_decoded_identity_mismatch"
            if abs(candidate[identity] - threshold) <= limits.near_threshold_distance
            else "material_decoded_identity_mismatch"
        )
        stats.add_category(category)
        categories.add(category)
        near_threshold_only &= category in _NONMATERIAL_CATEGORIES
    exact = not unmatched_reference and not unmatched_candidate
    return exact, exact or near_threshold_only, categories


def _validate_dataset(dataset: object) -> dict[str, object]:
    value = _mapping(dataset, name="oracle dataset")
    expected = {
        "id": PIIMB_DATASET_ID,
        "subset": PIIMB_SUBSET,
        "revision": PIIMB_REVISION,
        "split": PIIMB_SPLIT,
        "license": PIIMB_LICENSE,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise PIIMBValidationError(f"oracle dataset {name} differs from the PIIMB lock")
    source = _mapping(value.get("source_file"), name="oracle dataset source_file")
    expected_source = {
        "path": PIIMB_SOURCE_FILE,
        "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
        "sha256": PIIMB_SOURCE_SHA256,
        "rows": PIIMB_SOURCE_ROW_COUNT,
    }
    if dict(source) != expected_source:
        raise PIIMBValidationError("oracle dataset source file differs from the PIIMB lock")
    return {**expected, "source_file": expected_source}


def _validate_reference_final(step: Mapping[str, Any], final: Mapping[str, Any]) -> None:
    for name in (
        "accumulated_text",
        "model_words",
        "word_char_starts",
        "word_char_ends",
        "labels",
    ):
        if final.get(name) != step.get(name):
            raise PIIMBValidationError(f"oracle final state disagrees with its step for {name}")
    step_updates = _update_map(step.get("span_updates"), name="oracle step updates")
    final_updates = _update_map(final.get("span_scores"), name="oracle final scores")
    if set(step_updates) != set(final_updates):
        raise PIIMBValidationError("oracle final score boundaries differ from its sole step")
    for boundary in step_updates:
        for name in ("logits", "probs"):
            if _field(step_updates[boundary], name, description="oracle step update") != _field(
                final_updates[boundary], name, description="oracle final score"
            ):
                raise PIIMBValidationError(f"oracle final {name} differ from its sole step")


def _compare_case(
    backend: object,
    case: Mapping[str, Any],
    *,
    task_labels: Mapping[str, tuple[str, ...]],
    threshold: float,
    limits: PIIMBValidationTolerances,
) -> tuple[dict[str, object], _Stats]:
    selection = _mapping(case.get("selection"), name="oracle case selection")
    selection_index = _integer(selection.get("selection_index"), name="selection_index")
    benchmark_split = _nonblank(selection.get("benchmark_split"), name="benchmark_split")
    if benchmark_split not in {"dev", "test"}:
        raise PIIMBValidationError("benchmark_split must be dev or test")
    uid = _nonblank(selection.get("uid"), name="selection uid")
    source_row_index = _integer(selection.get("source_row_index"), name="source_row_index")
    task = _nonblank(selection.get("task_name"), name="selection task_name")
    if task not in task_labels:
        raise PIIMBValidationError(f"oracle case uses unknown task {task!r}")
    metadata_sha = _nonblank(selection.get("metadata_sha256"), name="metadata_sha256")
    if metadata_sha != _metadata_checksum(selection):
        raise PIIMBValidationError("oracle case selection metadata checksum is invalid")

    trace = _mapping(case.get("trace"), name="oracle case trace")
    labels = _string_tuple(trace.get("labels"), name="trace labels", nonempty=True)
    if labels != task_labels[task]:
        raise PIIMBValidationError("oracle case labels differ from its full task vocabulary")
    chunks = _sequence(trace.get("chunks"), name="trace chunks")
    steps = _sequence(trace.get("steps"), name="trace steps")
    if len(chunks) != 1 or len(steps) != 1 or not isinstance(chunks[0], str):
        raise PIIMBValidationError("PIIMB smoke cases must contain one full-sentence append")
    step = _mapping(steps[0], name="oracle step")
    if step.get("chunk") != chunks[0]:
        raise PIIMBValidationError("oracle chunk summary differs from its sole step")
    final = _mapping(trace.get("final_state"), name="oracle final state")
    _validate_reference_final(step, final)

    start_session = getattr(backend, "start_session", None)
    if not callable(start_session):
        raise PIIMBValidationError("MLX backend must expose start_session")
    session = start_session(list(labels), threshold=threshold, flat_ner=True, multi_label=False)
    append = getattr(session, "append", None)
    clear = getattr(session, "clear", None)
    if not callable(append) or not callable(clear):
        raise PIIMBValidationError("MLX session must expose append and clear")
    try:
        result = append(chunks[0])
    finally:
        clear()

    stats = _Stats()
    categories: set[str] = set()
    state = _field(result, "state", description="MLX append result")
    structural = {
        "accumulated_text_mismatch": _field(state, "accumulated_text", description="MLX state")
        != step.get("accumulated_text"),
        "model_words_mismatch": _string_tuple(
            _field(state, "word_tokens", description="MLX state"), name="MLX words"
        )
        != _string_tuple(step.get("model_words"), name="oracle words"),
        "word_char_starts_mismatch": _integer_tuple(
            _field(state, "word_char_starts", description="MLX state"), name="MLX starts"
        )
        != _integer_tuple(step.get("word_char_starts"), name="oracle starts"),
        "word_char_ends_mismatch": _integer_tuple(
            _field(state, "word_char_ends", description="MLX state"), name="MLX ends"
        )
        != _integer_tuple(step.get("word_char_ends"), name="oracle ends"),
        "ordered_labels_mismatch": _string_tuple(
            _field(state, "labels", description="MLX state"),
            name="MLX labels",
            nonempty=True,
        )
        != labels,
    }
    for category, mismatch in structural.items():
        if mismatch:
            stats.add_category(category)
            categories.add(category)

    reference_updates = _update_map(step.get("span_updates"), name="oracle span updates")
    candidate_updates = _update_map(
        _field(result, "span_updates", description="MLX append result"),
        name="MLX span updates",
    )
    boundary_summary = tuple(
        _boundary(value, name="oracle updated boundary")
        for value in _sequence(step.get("updated_boundaries"), name="updated boundaries")
    )
    if len(set(boundary_summary)) != len(boundary_summary) or set(boundary_summary) != set(
        reference_updates
    ):
        raise PIIMBValidationError("oracle updated-boundary summary is inconsistent")
    if set(candidate_updates) != set(reference_updates):
        stats.add_category("updated_candidate_boundaries_mismatch")
        categories.add("updated_candidate_boundaries_mismatch")

    for boundary in sorted(set(reference_updates) & set(candidate_updates)):
        reference = reference_updates[boundary]
        candidate = candidate_updates[boundary]
        reference_logits = _vector(
            _field(reference, "logits", description="oracle update"),
            name="oracle logits",
            probabilities=False,
        )
        reference_probs = _vector(
            _field(reference, "probs", description="oracle update"),
            name="oracle probabilities",
            probabilities=True,
        )
        candidate_logits = _vector(
            _field(candidate, "logits", description="MLX update"),
            name="MLX logits",
            probabilities=False,
        )
        candidate_probs = _vector(
            _field(candidate, "probs", description="MLX update"),
            name="MLX probabilities",
            probabilities=True,
        )
        if len(reference_logits) != len(labels):
            raise PIIMBValidationError("oracle update width differs from ordered task labels")
        stats.add_vectors(reference_logits, candidate_logits, reference_probs, candidate_probs)
        reference_top = _argmax(reference_probs)
        candidate_top = _argmax(candidate_probs)
        if (
            _integer(
                _field(reference, "top_label_index", description="oracle update"),
                name="oracle top_label_index",
            )
            != reference_top
            or _nonblank(
                _field(reference, "top_label", description="oracle update"),
                name="oracle top_label",
            )
            != labels[reference_top]
        ):
            raise PIIMBValidationError("oracle top-label metadata is inconsistent")
        if reference_top == candidate_top:
            stats.top_label_matches += 1
            stats.adjusted_top_label_matches += 1
        elif _is_saturated_near_tie(reference_logits, reference_probs, limits):
            stats.adjusted_top_label_matches += 1
            stats.add_category("saturated_near_top_tie")
            categories.add("saturated_near_top_tie")
        else:
            stats.add_category("material_top_label_mismatch")
            categories.add("material_top_label_mismatch")
        reference_kind = _field(reference, "update_kind", description="oracle update")
        candidate_kind = _field(candidate, "update_kind", description="MLX update")
        if reference_kind != candidate_kind:
            stats.add_category("update_kind_mismatch")
            categories.add("update_kind_mismatch")

    if not stats.numeric_pass(limits):
        if stats.probability_maximum_error > limits.maximum_probability_error:
            stats.add_category("probability_tolerance_exceeded")
            categories.add("probability_tolerance_exceeded")
        reference_norm = math.sqrt(stats.reference_logit_square_sum)
        if reference_norm >= limits.minimum_logit_l2_norm_for_cosine and (
            stats.logit_cosine is None or stats.logit_cosine < limits.minimum_logit_cosine
        ):
            stats.add_category("logit_cosine_tolerance_exceeded")
            categories.add("logit_cosine_tolerance_exceeded")

    entity_exact, entity_adjusted, entity_categories = _compare_entities(
        final.get("public_entities"),
        _field(result, "public_entities", description="MLX append result"),
        threshold=threshold,
        limits=limits,
        stats=stats,
    )
    categories.update(entity_categories)
    stats.decoded_identity_exact_cases += int(entity_exact)
    stats.decoded_identity_adjusted_cases += int(entity_adjusted)
    structural_exact = not any(structural.values()) and set(candidate_updates) == set(
        reference_updates
    )
    stats.structural_exact_cases += int(structural_exact)
    stats.finish_case(categories)
    material = sorted(categories - _NONMATERIAL_CATEGORIES)
    case_pass = structural_exact and stats.numeric_pass(limits) and not material
    row = {
        "selection_index": selection_index,
        "benchmark_split": benchmark_split,
        "uid": uid,
        "source_row_index": source_row_index,
        "task_name": task,
        "pass": case_pass,
        "candidate_count": len(reference_updates),
        "top_label_matches": stats.top_label_matches,
        "top_label_matches_adjusted": stats.adjusted_top_label_matches,
        "decoded_identity_exact": entity_exact,
        "disagreement_categories": dict(sorted(stats.categories.items())),
    }
    return row, stats


def run_piimb_validation(
    backend: object,
    oracle: Mapping[str, Any],
    *,
    expected_case_count: int = 100,
    expected_tasks: Sequence[str] = PRIMARY_TASKS,
    tolerances: PIIMBValidationTolerances | None = None,
) -> dict[str, Any]:
    """Compare the MLX backend with the single-append PIIMB reference smoke."""

    if not isinstance(oracle, Mapping):
        raise TypeError("oracle must be a mapping")
    count = _integer(expected_case_count, name="expected_case_count", minimum=1)
    tasks = tuple(expected_tasks)
    if not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise TypeError("expected_tasks must contain nonblank strings")
    if len(set(tasks)) != len(tasks):
        raise PIIMBValidationError("expected_tasks must be unique")
    limits = tolerances or PIIMBValidationTolerances()
    if oracle.get("schema_version") != PIIMB_ORACLE_SCHEMA_VERSION:
        raise PIIMBValidationError("unsupported PIIMB oracle schema version")
    if oracle.get("kind") != PIIMB_ORACLE_KIND:
        raise PIIMBValidationError("oracle kind is not the PIIMB reference smoke")
    model_id = _nonblank(oracle.get("model_id"), name="oracle model_id")
    revision = _nonblank(oracle.get("model_revision_sha"), name="oracle model revision")
    dataset = _validate_dataset(oracle.get("dataset"))
    threshold = _probability(oracle.get("threshold"), name="oracle threshold")
    if threshold != 0.5:
        raise PIIMBValidationError("PIIMB smoke oracle must use threshold 0.5")
    capture = _mapping(oracle.get("capture"), name="oracle capture")
    if capture.get("chunk_mode") != "single" or capture.get("appends_per_case") != 1:
        raise PIIMBValidationError("PIIMB smoke oracle must use one single-chunk append")

    labels_raw = _mapping(oracle.get("task_labels"), name="oracle task_labels")
    if tuple(labels_raw) != tasks:
        raise PIIMBValidationError("oracle task-label mapping differs from expected task order")
    task_labels = {
        task: _string_tuple(labels_raw[task], name=f"{task} labels", nonempty=True)
        for task in tasks
    }
    selection = _mapping(oracle.get("selection"), name="oracle selection")
    if selection.get("preset") != "smoke":
        raise PIIMBValidationError("oracle selection preset must be smoke")
    if _integer(selection.get("case_count"), name="selection case_count") != count:
        raise PIIMBValidationError("oracle selection case count differs from the smoke gate")
    manifest_sha = _nonblank(selection.get("manifest_sha256"), name="manifest_sha256")
    task_labels_sha = _nonblank(selection.get("task_labels_sha256"), name="task_labels_sha256")
    cases = _sequence(oracle.get("cases"), name="oracle cases")
    if len(cases) != count:
        raise PIIMBValidationError("oracle case array differs from its declared count")
    task_counts = _mapping(selection.get("task_counts"), name="selection task_counts")
    split_counts = _mapping(selection.get("split_counts"), name="selection split_counts")
    if tuple(task_counts) != tasks:
        raise PIIMBValidationError("selection task count order differs from expected tasks")
    if count % len(tasks) or any(task_counts.get(task) != count // len(tasks) for task in tasks):
        raise PIIMBValidationError("selection is not task-balanced")

    global_stats = _Stats()
    per_task_stats = {task: _Stats() for task in tasks}
    sanitized_cases: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    observed_split_counts: Counter[str] = Counter()
    for case_index, raw_case in enumerate(cases):
        case = _mapping(raw_case, name=f"oracle case {case_index}")
        case_selection = _mapping(case.get("selection"), name="oracle case selection")
        if case_selection.get("selection_index") != case_index:
            raise PIIMBValidationError("oracle selection indices must be contiguous and ordered")
        identity = (
            _nonblank(case_selection.get("uid"), name="selection uid"),
            _integer(case_selection.get("source_row_index"), name="source_row_index"),
        )
        if identity in identities:
            raise PIIMBValidationError("oracle repeats a (uid, source_row_index) identity")
        identities.add(identity)
        row, stats = _compare_case(
            backend,
            case,
            task_labels=task_labels,
            threshold=threshold,
            limits=limits,
        )
        task = str(row["task_name"])
        observed_split_counts[str(row["benchmark_split"])] += 1
        global_stats.merge(stats)
        per_task_stats[task].merge(stats)
        sanitized_cases.append(row)
    declared_splits = {
        split: _integer(split_counts.get(split), name=f"selection {split} count")
        for split in ("dev", "test")
    }
    if set(split_counts) != {"dev", "test"} or declared_splits != {
        split: observed_split_counts[split] for split in ("dev", "test")
    }:
        raise PIIMBValidationError("selection split counts differ from its cases")

    material_categories = sorted(
        category for category in global_stats.categories if category not in _NONMATERIAL_CATEGORIES
    )
    systematic_categories = sorted(
        category
        for category, case_count in global_stats.category_cases.items()
        if category not in _NONMATERIAL_CATEGORIES and case_count > 1
    )
    exact_structural_gate = global_stats.structural_exact_cases == count
    numerical_gate = global_stats.numeric_pass(limits) and all(
        stats.numeric_pass(limits) for stats in per_task_stats.values()
    )
    top_label_gate = (
        global_stats.adjusted_top_agreement >= limits.minimum_candidate_top_label_agreement
        and all(
            stats.adjusted_top_agreement >= limits.minimum_candidate_top_label_agreement
            for stats in per_task_stats.values()
        )
    )
    decoded_gate = global_stats.decoded_agreement >= limits.minimum_decoded_identity_agreement
    material_gate = not material_categories and not systematic_categories
    gates = {
        "exact_structure": exact_structural_gate,
        "numerical_tolerances": numerical_gate,
        "candidate_top_label_agreement": top_label_gate,
        "decoded_identity_agreement": decoded_gate,
        "no_material_or_systematic_category": material_gate,
    }
    per_task = {
        task: {
            "counts": per_task_stats[task].counts(),
            "metrics": per_task_stats[task].metrics(limits),
            "disagreement_categories": dict(sorted(per_task_stats[task].categories.items())),
        }
        for task in tasks
    }
    return {
        "schema_version": PIIMB_VALIDATION_SCHEMA_VERSION,
        "kind": PIIMB_REPORT_KIND,
        "model_id": model_id,
        "model_revision_sha": revision,
        "oracle_schema_version": PIIMB_ORACLE_SCHEMA_VERSION,
        "dataset": dataset,
        "selection": {
            "preset": "smoke",
            "manifest_sha256": manifest_sha,
            "task_labels_sha256": task_labels_sha,
            "case_count": count,
            "split_counts": dict(split_counts),
            "task_counts": dict(task_counts),
        },
        "threshold": threshold,
        "tolerances": limits.to_dict(),
        "pass": all(gates.values()),
        "gates": gates,
        "counts": global_stats.counts(),
        "metrics": global_stats.metrics(limits),
        "disagreement_categories": dict(sorted(global_stats.categories.items())),
        "material_categories": material_categories,
        "systematic_categories": systematic_categories,
        "per_task": per_task,
        "cases": sanitized_cases,
    }


def validate_sanitized_piimb_report(value: object) -> None:
    """Reject any field capable of redistributing PIIMB source content."""

    if isinstance(value, Mapping):
        forbidden = set(value) & _FORBIDDEN_REPORT_KEYS
        if forbidden:
            raise PIIMBValidationError(
                f"sanitized report contains forbidden fields: {sorted(forbidden)}"
            )
        for child in value.values():
            validate_sanitized_piimb_report(child)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            validate_sanitized_piimb_report(child)


def deterministic_piimb_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic JSON only after enforcing the no-content schema."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    validate_sanitized_piimb_report(payload)
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()


def write_piimb_validation_report(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, object]:
    """Write the sanitized report and return its content-addressed metadata."""

    path = Path(output_path)
    data = deterministic_piimb_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


__all__ = [
    "PIIMB_ORACLE_KIND",
    "PIIMB_ORACLE_SCHEMA_VERSION",
    "PIIMB_REPORT_KIND",
    "PIIMB_VALIDATION_SCHEMA_VERSION",
    "PIIMBValidationError",
    "PIIMBValidationTolerances",
    "deterministic_piimb_json_bytes",
    "run_piimb_validation",
    "validate_sanitized_piimb_report",
    "write_piimb_validation_report",
]
