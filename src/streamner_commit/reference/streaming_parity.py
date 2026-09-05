"""Deterministic streaming oracle export built on the pinned observer.

The observer remains the sole owner of the reference forward hook.  This module only
coordinates schedules, copies its already-validated session coordinates, and emits a
timing-free JSON contract suitable for PyTorch-versus-MLX differential tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.chunking import chunk_text_by_words
from streamner_commit.reference import observer as observer_module
from streamner_commit.streaming.replay import assert_span_states_close, replay_span_updates
from streamner_commit.types import PublicEntity, SpanBoundary, SpanScoreUpdate

STREAMING_PARITY_SCHEMA_VERSION = 1
SUPPORTED_CHUNK_UNITS = frozenset({1, 2, 4})


class StreamingParityError(ValueError):
    """A fixture, schedule, or observed reference state violated the oracle contract."""


def _nonblank_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StreamingParityError(f"{name} must be a nonblank string")
    return value


def _labels(value: object) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise StreamingParityError("fixture labels must be an ordered sequence")
    normalized = tuple(_nonblank_string(label, name="label") for label in value)
    if not normalized:
        raise StreamingParityError("fixture labels must not be empty")
    if len(normalized) != len(set(normalized)):
        raise StreamingParityError("fixture labels must be unique and ordered")
    return normalized


def normalize_chunk_units(chunk_units: Sequence[int]) -> tuple[int, ...]:
    """Validate and canonicalize a configurable subset of the 1/2/4 schedules."""

    if isinstance(chunk_units, str | bytes) or not isinstance(chunk_units, Sequence):
        raise TypeError("chunk_units must be a sequence of integers")
    normalized: set[int] = set()
    for value in chunk_units:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("chunk_units must contain only integers")
        if value not in SUPPORTED_CHUNK_UNITS:
            raise StreamingParityError("chunk_units values must be one of 1, 2, or 4")
        normalized.add(value)
    if not normalized:
        raise StreamingParityError("at least one chunk-unit schedule is required")
    return tuple(sorted(normalized))


def _threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("threshold must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise StreamingParityError("threshold must be finite and between zero and one")
    return normalized


def _fixture_cases(document: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_cases = document.get("cases")
    if isinstance(raw_cases, str | bytes) or not isinstance(raw_cases, Sequence):
        raise StreamingParityError("fixture document must contain a cases list")
    cases: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise StreamingParityError(f"fixture case {index} must be an object")
        case_id = _nonblank_string(raw_case.get("id"), name=f"fixture case {index} id")
        text = raw_case.get("text")
        if not isinstance(text, str) or not text.strip():
            raise StreamingParityError(f"fixture {case_id!r} text must be nonblank")
        if case_id in seen:
            raise StreamingParityError(f"duplicate fixture ID: {case_id!r}")
        seen.add(case_id)
        cases.append((case_id, text))
    if not cases:
        raise StreamingParityError("fixture document must contain at least one case")
    return tuple(cases)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _entities(values: Sequence[PublicEntity]) -> list[dict[str, Any]]:
    return [entity.to_dict() for entity in values]


def _updated_boundaries(updates: Sequence[SpanScoreUpdate]) -> list[list[int]]:
    boundaries = {update.boundary for update in updates}
    if len(boundaries) != len(updates):
        raise StreamingParityError("an observed step contains a duplicate updated boundary")
    return [[boundary.start_word, boundary.end_word] for boundary in sorted(boundaries)]


def _span_state_rows(
    state: Mapping[SpanBoundary, Sequence[float]],
    *,
    text: str,
    starts: Sequence[int],
    ends: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for boundary, raw_logits in sorted(state.items()):
        if boundary.end_word >= len(starts) or boundary.end_word >= len(ends):
            raise StreamingParityError(
                f"final boundary {boundary.to_tuple()} exceeds the model word coordinates"
            )
        logits = [float(value) for value in raw_logits]
        if not logits or not all(math.isfinite(value) for value in logits):
            raise StreamingParityError("final state contains an invalid logit vector")
        start_char = starts[boundary.start_word]
        end_char = ends[boundary.end_word]
        rows.append(
            {
                "start_word": boundary.start_word,
                "end_word": boundary.end_word,
                "start_char": start_char,
                "end_char": end_char,
                "span_text": text[start_char:end_char],
                "logits": logits,
                "probs": [_sigmoid(value) for value in logits],
            }
        )
    return rows


def _session_coordinates(
    model: object,
    session_id: str,
) -> tuple[str, tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    cache = getattr(model, "_session_cache", None)
    get_state = getattr(cache, "get", None)
    if not callable(get_state):
        raise StreamingParityError("reference model session cache is unavailable")
    state = get_state(session_id)
    if state is None:
        raise StreamingParityError("reference session state disappeared after append")
    # Reuse the observer's exact, tested cache-coordinate validation rather than
    # cloning private GLiNER assumptions into a second instrumentation path.
    return observer_module._state_coordinates(state)


def capture_streaming_case(
    model: object,
    *,
    case_id: str,
    text: str,
    labels: Sequence[str],
    chunk_units: int | None,
    run_id: str,
    threshold: float = 0.5,
    verify_reference: bool = True,
    single_chunk: bool = False,
) -> dict[str, Any]:
    """Capture one fixture under one schedule using the existing observer."""

    case_id = _nonblank_string(case_id, name="case_id")
    run_id = _nonblank_string(run_id, name="run_id")
    if not isinstance(text, str) or not text.strip():
        raise StreamingParityError("text must be nonblank")
    ordered_labels = _labels(labels)
    if not isinstance(single_chunk, bool):
        raise TypeError("single_chunk must be a boolean")
    if single_chunk:
        if chunk_units is not None:
            raise StreamingParityError("single-chunk capture requires chunk_units=None")
        schedule = None
        chunks = [text]
    else:
        if chunk_units is None:
            raise StreamingParityError("scheduled capture requires an integer chunk_units")
        schedule = normalize_chunk_units((chunk_units,))[0]
        chunks = chunk_text_by_words(text, schedule)
    threshold = _threshold(threshold)
    if "".join(chunks) != text:
        raise StreamingParityError("chunk schedule does not reconstruct the fixture text")

    session_id = f"{run_id}:{case_id}"
    step_rows: list[dict[str, Any]] = []
    all_updates: list[SpanScoreUpdate] = []
    final_observed: observer_module.ObservedAppend | None = None
    final_coordinates: tuple[str, tuple[str, ...], tuple[int, ...], tuple[int, ...]] | None = None
    with observer_module.StreamingSpanObserver(
        model,
        run_id=run_id,
        example_id=case_id,
        session_id=session_id,
        labels=ordered_labels,
        threshold=threshold,
        verify_reference=verify_reference,
    ) as observer:
        append_step = 0
        for chunk in chunks:
            if not chunk.strip():
                continue
            append_step += 1
            observed = observer.append(chunk, step=append_step)
            state_text, words, starts, ends = _session_coordinates(model, session_id)
            snapshot = observed.snapshot
            if state_text != snapshot.accumulated_text:
                raise StreamingParityError("observer snapshot and cache text disagree")
            if snapshot.visible_word_count != len(words):
                raise StreamingParityError("observer and cache model-word counts disagree")
            if tuple(update.boundary for update in observed.span_updates) != tuple(
                dict.fromkeys(update.boundary for update in observed.span_updates)
            ):
                raise StreamingParityError("observer emitted a duplicate boundary in one step")

            updates = tuple(observed.span_updates)
            all_updates.extend(updates)
            step_rows.append(
                {
                    "step": append_step,
                    "chunk": chunk,
                    "accumulated_text": state_text,
                    "visible_char_count": len(state_text),
                    "visible_word_count": len(words),
                    "model_words": list(words),
                    "word_char_starts": list(starts),
                    "word_char_ends": list(ends),
                    "labels": list(ordered_labels),
                    "updated_boundaries": _updated_boundaries(updates),
                    "span_updates": [update.to_dict() for update in updates],
                    "public_entities": _entities(snapshot.public_entities),
                    "validated_public_score_count": observed.validated_public_score_count,
                }
            )
            final_observed = observed
            final_coordinates = (state_text, words, starts, ends)

    if final_observed is None or final_coordinates is None:
        raise StreamingParityError("streaming fixture produced no nonblank append")
    assert_span_states_close(
        replay_span_updates(all_updates),
        final_observed.merged_span_logits,
    )
    state_text, words, starts, ends = final_coordinates
    if state_text != text:
        raise StreamingParityError("final accumulated text differs from the fixture text")
    final_scores = _span_state_rows(
        final_observed.merged_span_logits,
        text=state_text,
        starts=starts,
        ends=ends,
    )
    result = {
        "id": case_id,
        "full_text": text,
        "labels": list(ordered_labels),
        "chunk_units": schedule,
        "chunks": chunks,
        "step_count": len(step_rows),
        "span_update_count": len(all_updates),
        "steps": step_rows,
        "final_state": {
            "accumulated_text": state_text,
            "model_words": list(words),
            "word_char_starts": list(starts),
            "word_char_ends": list(ends),
            "labels": list(ordered_labels),
            "span_count": len(final_scores),
            "span_scores": final_scores,
            "public_entities": _entities(final_observed.snapshot.public_entities),
        },
    }
    if single_chunk:
        result["chunk_mode"] = "single"
    return result


def capture_streaming_parity_suite(
    model: object,
    *,
    fixture_document: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    chunk_units: Sequence[int] = (1,),
    threshold: float = 0.5,
    verify_reference: bool = True,
) -> dict[str, Any]:
    """Capture every synthetic case for each requested canonical schedule."""

    if not isinstance(fixture_document, Mapping):
        raise TypeError("fixture_document must be a mapping")
    model_id = _nonblank_string(model_id, name="model_id")
    model_revision = _nonblank_string(model_revision, name="model_revision")
    if len(model_revision) != 40:
        raise StreamingParityError("model_revision must be a 40-character commit SHA")
    labels = _labels(fixture_document.get("labels"))
    cases = _fixture_cases(fixture_document)
    schedules = normalize_chunk_units(chunk_units)
    threshold = _threshold(threshold)

    conditions: list[dict[str, Any]] = []
    total_steps = 0
    total_updates = 0
    total_final_spans = 0
    total_public_score_checks = 0
    for schedule in schedules:
        run_id = f"streaming-parity-{model_revision[:12]}-u{schedule}"
        captured_cases = [
            capture_streaming_case(
                model,
                case_id=case_id,
                text=text,
                labels=labels,
                chunk_units=schedule,
                run_id=run_id,
                threshold=threshold,
                verify_reference=verify_reference,
            )
            for case_id, text in cases
        ]
        for case in captured_cases:
            total_steps += int(case["step_count"])
            total_updates += int(case["span_update_count"])
            final_state = case["final_state"]
            if not isinstance(final_state, Mapping):
                raise StreamingParityError("captured final state is not an object")
            total_final_spans += int(final_state["span_count"])
            steps = case["steps"]
            if not isinstance(steps, list):
                raise StreamingParityError("captured steps are not a list")
            total_public_score_checks += sum(
                int(step["validated_public_score_count"])
                for step in steps
                if isinstance(step, Mapping)
            )
        conditions.append(
            {
                "chunk_units": schedule,
                "run_id": run_id,
                "case_count": len(captured_cases),
                "cases": captured_cases,
            }
        )

    return {
        "schema_version": STREAMING_PARITY_SCHEMA_VERSION,
        "kind": "synthetic_streaming_reference_parity_suite",
        "backend": "gliner-reference",
        "model_id": model_id,
        "model_revision_sha": model_revision,
        "gliner_version": observer_module.PINNED_GLINER_VERSION,
        "device": "cpu",
        "dtype": "float32",
        "fixture_schema_version": fixture_document.get("schema_version"),
        "threshold": threshold,
        "labels": list(labels),
        "chunk_unit_schedules": list(schedules),
        "fixture_case_count": len(cases),
        "condition_count": len(conditions),
        "totals": {
            "case_condition_count": len(cases) * len(schedules),
            "step_count": total_steps,
            "span_update_count": total_updates,
            "final_span_count": total_final_spans,
            "validated_public_score_count": total_public_score_checks,
        },
        "conditions": conditions,
    }


def deterministic_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize an oracle payload canonically and reject non-finite JSON values."""

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


def write_streaming_parity_suite(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write one deterministic JSON suite and return content-addressed metadata."""

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
    "STREAMING_PARITY_SCHEMA_VERSION",
    "SUPPORTED_CHUNK_UNITS",
    "StreamingParityError",
    "capture_streaming_case",
    "capture_streaming_parity_suite",
    "deterministic_json_bytes",
    "normalize_chunk_units",
    "write_streaming_parity_suite",
]
