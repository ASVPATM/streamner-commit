from __future__ import annotations

from dataclasses import fields

import pytest

from streamner_commit.streaming.tracker import (
    CandidateState,
    CandidateTracker,
    StreamingObservation,
    build_streaming_observations,
)
from streamner_commit.streaming.trajectory import (
    CandidateKey,
    HypothesisFamilyKey,
    build_candidate_trajectories,
    build_hypothesis_families,
)
from streamner_commit.types import SnapshotStep, SpanScoreUpdate, UpdateKind

LABELS = ("person", "organization")
RUN_ID = "trajectory-run"
EXAMPLE_ID = "sarah-example"


def _snapshot(
    step: int,
    accumulated_text: str,
    chunk: str,
    visible_word_count: int,
) -> SnapshotStep:
    return SnapshotStep(
        run_id=RUN_ID,
        example_id=EXAMPLE_ID,
        step=step,
        chunk=chunk,
        accumulated_text=accumulated_text,
        visible_char_count=len(accumulated_text),
        visible_word_count=visible_word_count,
        elapsed_ms=1.0,
        public_entities=(),
    )


def _update(
    snapshot: SnapshotStep,
    *,
    start_word: int,
    end_word: int,
    end_char: int,
    span_text: str,
    probs: tuple[float, float],
    update_kind: UpdateKind,
    previous_top_probability: float | None = None,
) -> SpanScoreUpdate:
    top_label_index = max(range(len(probs)), key=probs.__getitem__)
    top_probability = probs[top_label_index]
    second_probability = min(probs)
    return SpanScoreUpdate(
        run_id=RUN_ID,
        example_id=EXAMPLE_ID,
        step=snapshot.step,
        chunk=snapshot.chunk,
        visible_char_count=snapshot.visible_char_count,
        visible_word_count=snapshot.visible_word_count,
        start_word=start_word,
        end_word=end_word,
        start_char=0,
        end_char=end_char,
        span_text=span_text,
        logits=probs,
        probs=probs,
        top_label_index=top_label_index,
        top_label=LABELS[top_label_index],
        top_probability=top_probability,
        second_probability=second_probability,
        label_margin=top_probability - second_probability,
        previous_top_probability=previous_top_probability,
        top_probability_delta=(
            None
            if previous_top_probability is None
            else top_probability - previous_top_probability
        ),
        update_kind=update_kind,
        tail_distance_words=(snapshot.visible_word_count - 1) - end_word,
    )


def _hand_trace() -> tuple[tuple[SnapshotStep, ...], tuple[SpanScoreUpdate, ...]]:
    snapshots = (
        _snapshot(1, "Sarah", "Sarah", 1),
        _snapshot(2, "Sarah Johnson", " Johnson", 2),
        _snapshot(3, "Sarah Johnson arrived", " arrived", 3),
        _snapshot(4, "Sarah Johnson arrived.", ".", 3),
        _snapshot(5, "Sarah Johnson arrived. Today", " Today", 4),
    )
    updates = (
        _update(
            snapshots[0],
            start_word=0,
            end_word=0,
            end_char=5,
            span_text="Sarah",
            probs=(0.65, 0.10),
            update_kind="new",
        ),
        _update(
            snapshots[1],
            start_word=0,
            end_word=0,
            end_char=5,
            span_text="Sarah",
            probs=(0.72, 0.11),
            update_kind="rescore",
            previous_top_probability=0.65,
        ),
        _update(
            snapshots[1],
            start_word=0,
            end_word=1,
            end_char=13,
            span_text="Sarah Johnson",
            probs=(0.60, 0.12),
            update_kind="new",
        ),
        _update(
            snapshots[2],
            start_word=0,
            end_word=0,
            end_char=5,
            span_text="Sarah",
            probs=(0.48, 0.10),
            update_kind="rescore",
            previous_top_probability=0.72,
        ),
        _update(
            snapshots[2],
            start_word=0,
            end_word=1,
            end_char=13,
            span_text="Sarah Johnson",
            probs=(0.88, 0.13),
            update_kind="rescore",
            previous_top_probability=0.60,
        ),
        _update(
            snapshots[3],
            start_word=0,
            end_word=1,
            end_char=13,
            span_text="Sarah Johnson",
            probs=(0.90, 0.14),
            update_kind="rescore",
            previous_top_probability=0.88,
        ),
    )
    return snapshots, updates


def test_exact_trajectories_explode_labels_and_count_only_actual_updates() -> None:
    snapshots, updates = _hand_trace()
    trajectories = build_candidate_trajectories(updates, LABELS)

    sarah_key = CandidateKey(EXAMPLE_ID, 0, 0, "person")
    sarah = next(item for item in trajectories if item.key == sarah_key)
    assert sarah.rescore_count == 3
    assert tuple(item.step for item in sarah.observations) == (1, 2, 3)
    assert sarah.current_probability == pytest.approx(0.48)
    assert sarah.previous_probability == pytest.approx(0.72)
    assert sarah.probability_delta == pytest.approx(-0.24)
    assert sarah.rolling_deltas() == pytest.approx((0.07, -0.24))
    assert all(step not in {4, 5} for step in (item.step for item in sarah.observations))

    # Every vector is exploded in its declared label order, even when a label
    # is not the argmax.
    organization_key = CandidateKey(EXAMPLE_ID, 0, 0, "organization")
    organization = next(item for item in trajectories if item.key == organization_key)
    assert organization.rescore_count == 3
    assert organization.current_probability == pytest.approx(0.10)

    families = build_hypothesis_families(trajectories)
    person_family = next(
        item
        for item in families
        if item.key == HypothesisFamilyKey(EXAMPLE_ID, 0, "person")
    )
    assert person_family.members == (
        CandidateKey(EXAMPLE_ID, 0, 0, "person"),
        CandidateKey(EXAMPLE_ID, 0, 1, "person"),
    )
    assert snapshots[-1].step == 5  # The trace really does contain frozen later steps.


def test_online_states_keep_snapshot_age_distinct_from_rescore_count() -> None:
    snapshots, updates = _hand_trace()
    observations = build_streaming_observations(
        snapshots,
        updates,
        LABELS,
        rolling_window=2,
    )
    sarah_key = CandidateKey(EXAMPLE_ID, 0, 0, "person")
    johnson_key = CandidateKey(EXAMPLE_ID, 0, 1, "person")

    sarah_step_4 = observations[3].candidate(sarah_key)
    sarah_step_5 = observations[4].candidate(sarah_key)
    johnson_step_5 = observations[4].candidate(johnson_key)
    assert sarah_step_4 is not None
    assert sarah_step_5 is not None
    assert johnson_step_5 is not None

    assert sarah_step_4.rescore_count == sarah_step_5.rescore_count == 3
    assert sarah_step_4.snapshot_age == 1
    assert sarah_step_5.snapshot_age == 2
    assert not sarah_step_4.was_rescored
    assert not sarah_step_5.was_rescored
    assert sarah_step_5.rolling_deltas == pytest.approx((0.07, -0.24))
    assert sarah_step_5.probability_delta == pytest.approx(-0.24)
    assert sarah_step_5.tail_distance_words == 3

    assert johnson_step_5.rescore_count == 3
    assert johnson_step_5.snapshot_age == 1
    assert johnson_step_5.rolling_deltas == pytest.approx((0.28, 0.02))
    assert johnson_step_5.tail_distance_words == 2


def test_primary_family_extends_only_when_boundary_is_visible_in_prefix() -> None:
    snapshots, updates = _hand_trace()
    observations = build_streaming_observations(snapshots, updates, LABELS)
    family_key = HypothesisFamilyKey(EXAMPLE_ID, 0, "person")

    first_family = observations[0].family(family_key)
    second_family = observations[1].family(family_key)
    assert first_family is not None
    assert second_family is not None
    assert first_family.member_keys == (CandidateKey(EXAMPLE_ID, 0, 0, "person"),)
    assert first_family.observed_end_words == (0,)
    assert second_family.member_keys == (
        CandidateKey(EXAMPLE_ID, 0, 0, "person"),
        CandidateKey(EXAMPLE_ID, 0, 1, "person"),
    )
    assert second_family.observed_end_words == (0, 1)
    assert second_family.longest_member_key == CandidateKey(EXAMPLE_ID, 0, 1, "person")


def test_online_views_are_prefix_invariant_and_structurally_exclude_oracles() -> None:
    snapshots, updates = _hand_trace()
    complete = build_streaming_observations(snapshots, updates, LABELS)
    prefix = build_streaming_observations(
        snapshots[:2],
        tuple(update for update in updates if update.step <= 2),
        LABELS,
    )
    assert complete[:2] == prefix

    forbidden = {"gold_entities", "cold_full", "future_observations", "future_updates"}
    assert forbidden.isdisjoint(field.name for field in fields(CandidateState))
    assert forbidden.isdisjoint(field.name for field in fields(StreamingObservation))

    tracker = CandidateTracker(LABELS)
    with pytest.raises(TypeError, match="unexpected keyword"):
        tracker.observe(snapshots[0], (updates[0],), gold_entities=())  # type: ignore[call-arg]
