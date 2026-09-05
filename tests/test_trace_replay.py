from __future__ import annotations

from collections.abc import Iterator

import pytest

from streamner_commit.streaming.replay import (
    TraceReplayError,
    assert_span_states_close,
    replay_span_updates,
    replay_states_by_step,
    span_states_close,
)
from streamner_commit.types import SpanBoundary, SpanScoreUpdate


def make_update(
    *,
    step: int,
    boundary: tuple[int, int],
    logits: tuple[float, ...],
    kind: str,
    run_id: str = "run-1",
    example_id: str = "example-1",
    chunk: str | None = None,
    visible_char_count: int | None = None,
    visible_word_count: int = 4,
) -> SpanScoreUpdate:
    start_word, end_word = boundary
    start_char = start_word * 2
    end_char = end_word * 2 + 1
    probabilities = (0.8,) + (0.1,) * (len(logits) - 1)
    second_probability = max(probabilities[1:], default=0.0)
    return SpanScoreUpdate(
        run_id=run_id,
        example_id=example_id,
        step=step,
        chunk=chunk if chunk is not None else f" chunk-{step}",
        visible_char_count=(
            visible_char_count if visible_char_count is not None else visible_word_count * 2
        ),
        visible_word_count=visible_word_count,
        start_word=start_word,
        end_word=end_word,
        start_char=start_char,
        end_char=end_char,
        span_text="x" * (end_char - start_char),
        logits=logits,
        probs=probabilities,
        top_label_index=0,
        top_label="person",
        top_probability=probabilities[0],
        second_probability=second_probability,
        label_margin=probabilities[0] - second_probability,
        previous_top_probability=None,
        top_probability_delta=None,
        update_kind=kind,  # type: ignore[arg-type]
        tail_distance_words=(visible_word_count - 1) - end_word,
    )


def sample_updates() -> list[SpanScoreUpdate]:
    return [
        make_update(step=1, boundary=(1, 1), logits=(1.0, 2.0), kind="new"),
        make_update(step=1, boundary=(0, 0), logits=(3.0, 4.0), kind="new"),
        make_update(step=3, boundary=(0, 0), logits=(5.0, 6.0), kind="rescore"),
        make_update(step=3, boundary=(2, 2), logits=(7.0, 8.0), kind="new"),
        make_update(step=4, boundary=(1, 1), logits=(9.0, 10.0), kind="rescore"),
    ]


def test_new_and_rescore_events_insert_and_replace_vectors() -> None:
    state = replay_span_updates(sample_updates())

    assert tuple(state) == (SpanBoundary(0, 0), SpanBoundary(1, 1), SpanBoundary(2, 2))
    assert state == {
        SpanBoundary(0, 0): (5.0, 6.0),
        SpanBoundary(1, 1): (9.0, 10.0),
        SpanBoundary(2, 2): (7.0, 8.0),
    }
    with pytest.raises(TypeError):
        state[SpanBoundary(3, 3)] = (0.0, 0.0)  # type: ignore[index]


@pytest.mark.parametrize(
    ("through_step", "expected"),
    [
        (0, {}),
        (1, {(0, 0): (3.0, 4.0), (1, 1): (1.0, 2.0)}),
        (2, {(0, 0): (3.0, 4.0), (1, 1): (1.0, 2.0)}),
        (3, {(0, 0): (5.0, 6.0), (1, 1): (1.0, 2.0), (2, 2): (7.0, 8.0)}),
        (99, {(0, 0): (5.0, 6.0), (1, 1): (9.0, 10.0), (2, 2): (7.0, 8.0)}),
    ],
)
def test_replay_supports_state_through_any_inclusive_step(
    through_step: int,
    expected: dict[tuple[int, int], tuple[float, ...]],
) -> None:
    assert span_states_close(
        replay_span_updates(sample_updates(), through_step=through_step),
        expected,
    )


def test_full_events_can_initialize_or_replace_a_boundary() -> None:
    updates = [
        make_update(step=0, boundary=(0, 0), logits=(1.0, 1.5), kind="full"),
        make_update(step=0, boundary=(1, 1), logits=(2.0, 2.5), kind="full"),
        make_update(step=1, boundary=(0, 0), logits=(3.0, 3.5), kind="full"),
    ]

    assert replay_span_updates(updates) == {
        SpanBoundary(0, 0): (3.0, 3.5),
        SpanBoundary(1, 1): (2.0, 2.5),
    }


def test_states_by_step_are_post_step_snapshots_and_do_not_alias() -> None:
    states = replay_states_by_step(sample_updates())

    assert tuple(states) == (1, 3, 4)
    assert states[1][SpanBoundary(0, 0)] == (3.0, 4.0)
    assert states[3][SpanBoundary(0, 0)] == (5.0, 6.0)
    assert states[1][SpanBoundary(1, 1)] == (1.0, 2.0)
    assert states[4][SpanBoundary(1, 1)] == (9.0, 10.0)
    with pytest.raises(TypeError):
        states[2] = {}  # type: ignore[index]


def test_empty_and_generator_event_streams_are_supported() -> None:
    assert replay_span_updates([]) == {}
    assert replay_states_by_step([]) == {}

    def event_generator() -> Iterator[SpanScoreUpdate]:
        yield from sample_updates()

    assert replay_span_updates(event_generator()) == replay_span_updates(sample_updates())


def test_step_regression_is_rejected() -> None:
    updates = [
        make_update(step=2, boundary=(0, 0), logits=(1.0, 2.0), kind="new"),
        make_update(step=1, boundary=(1, 1), logits=(3.0, 4.0), kind="new"),
    ]

    with pytest.raises(TraceReplayError, match="step regression"):
        replay_span_updates(updates)


def test_duplicate_boundary_within_one_step_is_rejected() -> None:
    updates = [
        make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="new"),
        make_update(step=1, boundary=(0, 0), logits=(3.0, 4.0), kind="full"),
    ]

    with pytest.raises(TraceReplayError, match="duplicate update"):
        replay_span_updates(updates)


def test_new_boundary_must_not_have_been_seen_before() -> None:
    updates = [
        make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="new"),
        make_update(step=2, boundary=(0, 0), logits=(3.0, 4.0), kind="new"),
    ]

    with pytest.raises(TraceReplayError, match="new update repeats"):
        replay_span_updates(updates)


def test_rescore_boundary_must_have_been_seen_before() -> None:
    update = make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="rescore")

    with pytest.raises(TraceReplayError, match="unseen boundary"):
        replay_span_updates([update])


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "run-2"},
        {"example_id": "example-2"},
    ],
)
def test_events_must_belong_to_one_trace(override: dict[str, str]) -> None:
    first = make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="new")
    second = make_update(
        step=2,
        boundary=(1, 1),
        logits=(3.0, 4.0),
        kind="new",
        **override,
    )

    with pytest.raises(TraceReplayError, match="one run_id and example_id"):
        replay_span_updates([first, second])


def test_class_vector_width_must_be_constant() -> None:
    updates = [
        make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="new"),
        make_update(step=2, boundary=(1, 1), logits=(3.0, 4.0, 5.0), kind="new"),
    ]

    with pytest.raises(TraceReplayError, match="equal width"):
        replay_span_updates(updates)


def test_metadata_must_be_consistent_within_a_step() -> None:
    updates = [
        make_update(step=1, boundary=(0, 0), logits=(1.0, 2.0), kind="new"),
        make_update(
            step=1,
            boundary=(1, 1),
            logits=(3.0, 4.0),
            kind="new",
            chunk="different chunk",
        ),
    ]

    with pytest.raises(TraceReplayError, match="inconsistent metadata"):
        replay_span_updates(updates)


def test_state_comparison_accepts_key_forms_and_configurable_tolerance() -> None:
    actual = {SpanBoundary(0, 0): (1.0, -2.0), SpanBoundary(1, 2): (3.0, 4.0)}
    expected = {(0, 0): [1.0 + 5e-7, -2.0], (1, 2): [3.0, 4.0 - 5e-7]}

    assert span_states_close(actual, expected)
    assert not span_states_close(actual, expected, rel_tol=0.0, abs_tol=1e-8)
    assert_span_states_close(actual, expected, rel_tol=0.0, abs_tol=1e-6)


def test_state_comparison_reports_key_width_and_value_mismatches() -> None:
    with pytest.raises(AssertionError, match=r"missing=.*\(1, 1\)"):
        assert_span_states_close({(0, 0): [1.0]}, {(0, 0): [1.0], (1, 1): [2.0]})
    with pytest.raises(AssertionError, match="vector width differs"):
        assert_span_states_close({(0, 0): [1.0]}, {(0, 0): [1.0, 2.0]})
    with pytest.raises(AssertionError, match=r"\(0, 0\)\[1\]"):
        assert_span_states_close({(0, 0): [1.0, 2.0]}, {(0, 0): [1.0, 9.0]})


@pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan")])
def test_comparison_tolerance_must_be_finite_and_nonnegative(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        span_states_close({}, {}, abs_tol=tolerance)


@pytest.mark.parametrize("through_step", [-1, 1.5, True])
def test_through_step_must_be_a_nonnegative_integer(through_step: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replay_span_updates([], through_step=through_step)  # type: ignore[arg-type]


def test_replay_rejects_non_update_records() -> None:
    with pytest.raises(TypeError, match="SpanScoreUpdate"):
        replay_span_updates(["not an update"])  # type: ignore[list-item]
