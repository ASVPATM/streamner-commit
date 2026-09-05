"""Deterministic reconstruction of StreamingSpan's historical score map."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from itertools import groupby
from types import MappingProxyType

from streamner_commit.types import SpanBoundary, SpanScoreUpdate

type SpanLogitVector = tuple[float, ...]
type SpanState = Mapping[SpanBoundary, SpanLogitVector]
type SpanStateInput = Mapping[SpanBoundary, Iterable[float]]

DEFAULT_REL_TOLERANCE = 1e-6
DEFAULT_ABS_TOLERANCE = 1e-6


class TraceReplayError(ValueError):
    """Raised when an event stream cannot represent one deterministic trace."""


def replay_span_updates(
    updates: Iterable[SpanScoreUpdate],
    *,
    through_step: int | None = None,
) -> SpanState:
    """Replay updates through an inclusive step and return an immutable state.

    ``new`` events insert previously unseen boundaries, ``rescore`` events replace
    existing vectors, and ``full`` events insert or replace unconditionally.  A
    requested step before the first event yields an empty state; a requested step
    after the final event yields the final state.

    The complete supplied event stream is validated even when ``through_step`` is
    earlier than its end, so a malformed trace cannot appear valid by truncation.
    """

    if through_step is not None:
        _validate_step(through_step, name="through_step")

    prepared = _prepare_updates(updates)
    state: dict[SpanBoundary, SpanLogitVector] = {}
    selected: dict[SpanBoundary, SpanLogitVector] = {}

    for step, step_updates_iter in groupby(prepared, key=lambda update: update.step):
        for update in step_updates_iter:
            _apply_update(state, update)
        if through_step is None or step <= through_step:
            selected = state.copy()

    return _freeze_state(selected)


def replay_states_by_step(updates: Iterable[SpanScoreUpdate]) -> Mapping[int, SpanState]:
    """Return immutable post-step states for every step containing an event."""

    prepared = _prepare_updates(updates)
    state: dict[SpanBoundary, SpanLogitVector] = {}
    states: dict[int, SpanState] = {}

    for step, step_updates_iter in groupby(prepared, key=lambda update: update.step):
        for update in step_updates_iter:
            _apply_update(state, update)
        states[step] = _freeze_state(state)

    return MappingProxyType(states)


def span_states_close(
    actual: SpanStateInput,
    expected: SpanStateInput,
    *,
    rel_tol: float = DEFAULT_REL_TOLERANCE,
    abs_tol: float = DEFAULT_ABS_TOLERANCE,
) -> bool:
    """Return whether two span maps have identical keys and close vectors."""

    try:
        assert_span_states_close(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)
    except AssertionError:
        return False
    return True


def assert_span_states_close(
    actual: SpanStateInput,
    expected: SpanStateInput,
    *,
    rel_tol: float = DEFAULT_REL_TOLERANCE,
    abs_tol: float = DEFAULT_ABS_TOLERANCE,
) -> None:
    """Assert exact boundary parity and elementwise floating-point closeness."""

    rel_tol = _validate_tolerance(rel_tol, name="rel_tol")
    abs_tol = _validate_tolerance(abs_tol, name="abs_tol")
    actual_state = _normalize_external_state(actual, name="actual")
    expected_state = _normalize_external_state(expected, name="expected")

    actual_keys = set(actual_state)
    expected_keys = set(expected_state)
    if actual_keys != expected_keys:
        missing = sorted(boundary.to_tuple() for boundary in expected_keys - actual_keys)
        unexpected = sorted(boundary.to_tuple() for boundary in actual_keys - expected_keys)
        raise AssertionError(f"span boundaries differ: missing={missing}, unexpected={unexpected}")

    for boundary in sorted(actual_keys):
        actual_vector = actual_state[boundary]
        expected_vector = expected_state[boundary]
        if len(actual_vector) != len(expected_vector):
            raise AssertionError(
                f"vector width differs at {boundary.to_tuple()}: "
                f"actual={len(actual_vector)}, expected={len(expected_vector)}"
            )
        for index, (actual_value, expected_value) in enumerate(
            zip(actual_vector, expected_vector, strict=True)
        ):
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                raise AssertionError(
                    f"logit differs at {boundary.to_tuple()}[{index}]: "
                    f"actual={actual_value}, expected={expected_value}, "
                    f"rel_tol={rel_tol}, abs_tol={abs_tol}"
                )


def _prepare_updates(updates: Iterable[SpanScoreUpdate]) -> tuple[SpanScoreUpdate, ...]:
    if isinstance(updates, str | bytes) or not isinstance(updates, Iterable):
        raise TypeError("updates must be an iterable of SpanScoreUpdate records")
    prepared = tuple(updates)
    if not all(isinstance(update, SpanScoreUpdate) for update in prepared):
        raise TypeError("updates must contain only SpanScoreUpdate records")
    if not prepared:
        return prepared

    run_id = prepared[0].run_id
    example_id = prepared[0].example_id
    vector_width = len(prepared[0].logits)
    previous_step = prepared[0].step
    seen_in_step: set[SpanBoundary] = set()
    step_metadata = _step_metadata(prepared[0])

    for index, update in enumerate(prepared):
        if update.run_id != run_id or update.example_id != example_id:
            raise TraceReplayError("all updates must belong to one run_id and example_id")
        if len(update.logits) != vector_width:
            raise TraceReplayError("all update logit vectors must have equal width")
        if update.step < previous_step:
            raise TraceReplayError(
                f"step regression at event {index}: {update.step} follows {previous_step}"
            )
        if update.step != previous_step:
            seen_in_step.clear()
            step_metadata = _step_metadata(update)
        elif _step_metadata(update) != step_metadata:
            raise TraceReplayError(f"inconsistent metadata within step {update.step}")
        if update.boundary in seen_in_step:
            raise TraceReplayError(
                f"duplicate update for boundary {update.boundary.to_tuple()} at step {update.step}"
            )
        seen_in_step.add(update.boundary)
        previous_step = update.step

    return prepared


def _step_metadata(update: SpanScoreUpdate) -> tuple[str, int, int]:
    return update.chunk, update.visible_char_count, update.visible_word_count


def _apply_update(
    state: dict[SpanBoundary, SpanLogitVector],
    update: SpanScoreUpdate,
) -> None:
    boundary = update.boundary
    exists = boundary in state
    if update.update_kind == "new" and exists:
        raise TraceReplayError(
            f"new update repeats previously seen boundary {boundary.to_tuple()} "
            f"at step {update.step}"
        )
    if update.update_kind == "rescore" and not exists:
        raise TraceReplayError(
            f"rescore update references unseen boundary {boundary.to_tuple()} at step {update.step}"
        )
    state[boundary] = update.logits


def _freeze_state(state: Mapping[SpanBoundary, SpanLogitVector]) -> SpanState:
    return MappingProxyType({boundary: state[boundary] for boundary in sorted(state)})


def _normalize_external_state(
    state: SpanStateInput, *, name: str
) -> dict[SpanBoundary, SpanLogitVector]:
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} state must be a mapping")
    normalized: dict[SpanBoundary, SpanLogitVector] = {}
    for raw_boundary, raw_vector in state.items():
        boundary = _normalize_boundary(raw_boundary, name=name)
        if boundary in normalized:
            raise ValueError(f"{name} state contains duplicate boundary {boundary.to_tuple()}")
        normalized[boundary] = _normalize_vector(raw_vector, name=name, boundary=boundary)
    return normalized


def _normalize_boundary(
    boundary: SpanBoundary | tuple[int, int],
    *,
    name: str,
) -> SpanBoundary:
    if isinstance(boundary, SpanBoundary):
        return boundary
    if isinstance(boundary, tuple) and len(boundary) == 2:
        try:
            return SpanBoundary(boundary[0], boundary[1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} state contains an invalid boundary: {boundary!r}") from error
    raise TypeError(f"{name} state keys must be SpanBoundary values or integer pairs")


def _normalize_vector(
    vector: Iterable[float],
    *,
    name: str,
    boundary: SpanBoundary,
) -> SpanLogitVector:
    if isinstance(vector, str | bytes) or not isinstance(vector, Iterable):
        raise TypeError(f"{name} vector at {boundary.to_tuple()} must be iterable")
    try:
        normalized = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} vector at {boundary.to_tuple()} must contain numbers") from error
    if not normalized:
        raise ValueError(f"{name} vector at {boundary.to_tuple()} must not be empty")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} vector at {boundary.to_tuple()} must contain finite values")
    return normalized


def _validate_step(step: int, *, name: str) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"{name} must be an integer")
    if step < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_tolerance(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


__all__ = [
    "DEFAULT_ABS_TOLERANCE",
    "DEFAULT_REL_TOLERANCE",
    "SpanLogitVector",
    "SpanState",
    "TraceReplayError",
    "assert_span_states_close",
    "replay_span_updates",
    "replay_states_by_step",
    "span_states_close",
]
