from __future__ import annotations

import inspect
from dataclasses import fields, replace

import pytest

from streamner_commit.policies import (
    EMA,
    FixedLag,
    FixedThreshold,
    OracleStable,
    RescorePatience,
    SnapshotPatience,
    StabilityGate,
    StabilityGateConfig,
    extract_stability_features,
    simulate_commitments,
    stability_gate_ablations,
)
from streamner_commit.policies.base import ready_candidate
from streamner_commit.policies.resolver import CommitResolver
from streamner_commit.policies.simulator import SimulationError
from streamner_commit.streaming.tracker import CandidateState, StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey, HypothesisFamilyKey

LABELS = ("person", "organization")
EXAMPLE_ID = "sarah-example"


def _state(
    *,
    step: int,
    end_word: int,
    label: str,
    probability: float,
    previous: float | None,
    rolling: tuple[float, ...],
    rescore_count: int,
    last_seen_step: int,
    snapshot_age: int,
    was_rescored: bool,
    visible_word_count: int,
) -> CandidateState:
    text = "Sarah" if end_word == 0 else "Sarah Johnson"
    key = CandidateKey(EXAMPLE_ID, 0, end_word, label)
    return CandidateState(
        key=key,
        family_key=key.family_key,
        step=step,
        first_seen_step=1 if end_word == 0 else 2,
        last_seen_step=last_seen_step,
        snapshot_age=snapshot_age,
        rescore_count=rescore_count,
        current_probability=probability,
        previous_probability=previous,
        probability_delta=rolling[-1] if rolling else None,
        rolling_deltas=rolling,
        current_logit=probability,
        is_top_label=label == "person",
        was_rescored=was_rescored,
        last_update_kind="new" if rescore_count == 1 else "rescore",
        start_char=0,
        end_char=len(text),
        span_text=text,
        visible_char_count=(5, 13, 21, 22, 28)[step - 1],
        visible_word_count=visible_word_count,
        tail_distance_words=(visible_word_count - 1) - end_word,
    )


def _observation(
    step: int,
    visible_word_count: int,
    candidates: list[CandidateState],
    rescored: list[CandidateKey],
) -> StreamingObservation:
    return StreamingObservation(
        run_id="policy-run",
        example_id=EXAMPLE_ID,
        step=step,
        chunk=("Sarah", " Johnson", " arrived", ".", " Today")[step - 1],
        visible_char_count=(5, 13, 21, 22, 28)[step - 1],
        visible_word_count=visible_word_count,
        candidates=tuple(sorted(candidates, key=lambda state: state.key)),
        families=(),
        rescored_keys=tuple(sorted(rescored)),
    )


def _trace() -> tuple[StreamingObservation, ...]:
    sarah_person = CandidateKey(EXAMPLE_ID, 0, 0, "person")
    sarah_org = CandidateKey(EXAMPLE_ID, 0, 0, "organization")
    johnson_person = CandidateKey(EXAMPLE_ID, 0, 1, "person")
    johnson_org = CandidateKey(EXAMPLE_ID, 0, 1, "organization")

    step_1 = [
        _state(
            step=1,
            end_word=0,
            label="person",
            probability=0.65,
            previous=None,
            rolling=(),
            rescore_count=1,
            last_seen_step=1,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=1,
        ),
        _state(
            step=1,
            end_word=0,
            label="organization",
            probability=0.10,
            previous=None,
            rolling=(),
            rescore_count=1,
            last_seen_step=1,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=1,
        ),
    ]
    step_2 = [
        _state(
            step=2,
            end_word=0,
            label="person",
            probability=0.72,
            previous=0.65,
            rolling=(0.07,),
            rescore_count=2,
            last_seen_step=2,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=2,
        ),
        _state(
            step=2,
            end_word=0,
            label="organization",
            probability=0.11,
            previous=0.10,
            rolling=(0.01,),
            rescore_count=2,
            last_seen_step=2,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=2,
        ),
        _state(
            step=2,
            end_word=1,
            label="person",
            probability=0.60,
            previous=None,
            rolling=(),
            rescore_count=1,
            last_seen_step=2,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=2,
        ),
        _state(
            step=2,
            end_word=1,
            label="organization",
            probability=0.12,
            previous=None,
            rolling=(),
            rescore_count=1,
            last_seen_step=2,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=2,
        ),
    ]
    step_3 = [
        _state(
            step=3,
            end_word=0,
            label="person",
            probability=0.48,
            previous=0.72,
            rolling=(0.07, -0.24),
            rescore_count=3,
            last_seen_step=3,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
        _state(
            step=3,
            end_word=0,
            label="organization",
            probability=0.10,
            previous=0.11,
            rolling=(0.01, -0.01),
            rescore_count=3,
            last_seen_step=3,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
        _state(
            step=3,
            end_word=1,
            label="person",
            probability=0.88,
            previous=0.60,
            rolling=(0.28,),
            rescore_count=2,
            last_seen_step=3,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
        _state(
            step=3,
            end_word=1,
            label="organization",
            probability=0.13,
            previous=0.12,
            rolling=(0.01,),
            rescore_count=2,
            last_seen_step=3,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
    ]
    step_4 = [
        replace(
            state,
            step=4,
            snapshot_age=1,
            was_rescored=False,
            visible_char_count=22,
        )
        for state in step_3[:2]
    ] + [
        _state(
            step=4,
            end_word=1,
            label="person",
            probability=0.90,
            previous=0.88,
            rolling=(0.28, 0.02),
            rescore_count=3,
            last_seen_step=4,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
        _state(
            step=4,
            end_word=1,
            label="organization",
            probability=0.14,
            previous=0.13,
            rolling=(0.01, 0.01),
            rescore_count=3,
            last_seen_step=4,
            snapshot_age=0,
            was_rescored=True,
            visible_word_count=3,
        ),
    ]
    step_5 = [
        replace(
            state,
            step=5,
            snapshot_age=state.snapshot_age + 1,
            was_rescored=False,
            visible_char_count=28,
            visible_word_count=4,
            tail_distance_words=3 - state.key.end_word,
        )
        for state in step_4
    ]
    return (
        _observation(1, 1, step_1, [sarah_person, sarah_org]),
        _observation(
            2,
            2,
            step_2,
            [sarah_person, sarah_org, johnson_person, johnson_org],
        ),
        _observation(
            3,
            3,
            step_3,
            [sarah_person, sarah_org, johnson_person, johnson_org],
        ),
        _observation(4, 3, step_4, [johnson_person, johnson_org]),
        _observation(5, 4, step_5, []),
    )


def test_baseline_commit_steps_and_cached_vs_rescore_patience() -> None:
    observations = _trace()
    johnson = CandidateKey(EXAMPLE_ID, 0, 1, "person")

    assert simulate_commitments(
        FixedThreshold(0.8), observations, LABELS
    ).commit_step(johnson) == 3
    assert simulate_commitments(FixedLag(0.8, 2), observations, LABELS).commit_step(
        johnson
    ) == 5
    assert simulate_commitments(
        SnapshotPatience(0.8, 3), observations, LABELS
    ).commit_step(johnson) == 5
    assert simulate_commitments(
        RescorePatience(0.8, 3), observations, LABELS
    ).commit_step(johnson) == 4
    assert simulate_commitments(
        RescorePatience(0.8, 4), observations, LABELS
    ).commit_step(johnson) is None


def test_ema_updates_only_on_actual_rescores() -> None:
    observations = _trace()
    sarah = CandidateKey(EXAMPLE_ID, 0, 0, "person")
    policy = EMA(threshold=0.99, alpha=0.5)
    policy.reset(list(LABELS))

    expected = (0.65, 0.685, 0.5825, 0.5825, 0.5825)
    for observation, expected_value in zip(observations, expected, strict=True):
        state = observation.candidate(sarah)
        assert state is not None
        assert policy.observe(observation, state) == []
        assert policy.value(sarah) == pytest.approx(expected_value)


def test_resolver_is_order_deterministic_immutable_and_counts_revisions() -> None:
    observations = _trace()
    step_2 = observations[1]
    sarah_state = step_2.candidate(CandidateKey(EXAMPLE_ID, 0, 0, "person"))
    johnson_state = step_2.candidate(CandidateKey(EXAMPLE_ID, 0, 1, "person"))
    assert sarah_state is not None and johnson_state is not None
    decisions = [
        ready_candidate("manual", sarah_state),
        ready_candidate("manual", johnson_state),
    ]

    forward = CommitResolver()
    reverse = CommitResolver()
    forward_step = forward.resolve(step_2, decisions)
    assert tuple(item.key for item in forward_step.newly_committed) == (
        sarah_state.key,
    )
    assert forward_step.blocked_revision_count == 1
    assert forward_step.blocked[0].reason == "ready_overlap"
    assert tuple(
        item.key for item in reverse.resolve(step_2, reversed(decisions)).newly_committed
    ) == (sarah_state.key,)

    step_3 = observations[2]
    stronger_extension = step_3.candidate(johnson_state.key)
    relabel = step_3.candidate(CandidateKey(EXAMPLE_ID, 0, 0, "organization"))
    assert stronger_extension is not None and relabel is not None
    blocked = forward.resolve(
        step_3,
        [
            ready_candidate("manual", stronger_extension),
            ready_candidate("manual", relabel, readiness_score=0.95),
        ],
    )
    assert not blocked.newly_committed
    assert {item.reason for item in blocked.blocked} == {"committed_overlap"}
    assert all(item.is_revision for item in blocked.blocked)
    assert blocked.blocked_revision_count == 2
    assert blocked.cumulative_blocked_revision_count == 3
    original = forward.committed[0]
    assert original.commit_step == 2
    assert original.decision.probability == pytest.approx(0.72)


def test_stability_features_gate_ablations_and_highest_label_resolution() -> None:
    observations = _trace()
    step_3 = observations[2]
    sarah = step_3.candidate(CandidateKey(EXAMPLE_ID, 0, 0, "person"))
    assert sarah is not None
    features = extract_stability_features(step_3, sarah, horizon=2)
    assert features.confidence == pytest.approx(0.48)
    assert features.rescore_count == 3
    assert features.recent_instability == pytest.approx(0.24)
    assert features.label_margin == pytest.approx(0.38)
    assert features.extension_advantage == pytest.approx(0.40)
    assert features.tail_distance_words == 2

    config = StabilityGateConfig(
        tau=0.4,
        min_rescores=3,
        instability_horizon=2,
        epsilon=0.3,
        min_label_margin=0.3,
        max_extension_advantage=0.1,
    )
    gate = StabilityGate(config)
    gate.reset(list(LABELS))
    assert gate.observe(step_3, sarah) == []
    ablations = stability_gate_ablations(config, tail_distance_words=2)
    minus_extension = StabilityGate(ablations["minus_extension"])
    minus_extension.reset(list(LABELS))
    assert len(minus_extension.observe(step_3, sarah)) == 1
    assert set(ablations) == {
        "full",
        "minus_instability",
        "minus_label_margin",
        "minus_extension",
        "plus_tail_distance",
    }
    assert StabilityGateConfig.from_mapping(config.to_dict()) == config

    # Several labels and overlapping boundaries are ready; the independent
    # resolver must retain only the highest-scoring label/span.
    permissive = StabilityGate(
        StabilityGateConfig(
            tau=0.1,
            min_rescores=1,
            instability_horizon=2,
            epsilon=1.0,
            min_label_margin=0.0,
            max_extension_advantage=1.0,
        )
    )
    permissive.reset(list(LABELS))
    step_4 = observations[3]
    ready = [
        decision
        for state in step_4.candidates
        for decision in permissive.observe(step_4, state)
    ]
    resolution = CommitResolver().resolve(step_4, ready)
    assert tuple(item.key for item in resolution.newly_committed) == (
        CandidateKey(EXAMPLE_ID, 0, 1, "person"),
    )


def test_deployable_surfaces_are_prefix_only_and_oracle_requires_opt_in() -> None:
    forbidden = {"gold_entities", "cold_full", "future_observations", "future_updates"}
    deployable = (
        FixedThreshold,
        FixedLag,
        SnapshotPatience,
        RescorePatience,
        EMA,
        StabilityGate,
    )
    for policy_type in deployable:
        assert tuple(inspect.signature(policy_type.observe).parameters) == (
            "self",
            "step",
            "state",
        )
        assert forbidden.isdisjoint(field.name for field in fields(policy_type))

    observations = _trace()
    oracle = OracleStable(observations, threshold=0.8)
    with pytest.raises(SimulationError, match="analysis-only"):
        simulate_commitments(oracle, observations, LABELS)
    analysis_run = simulate_commitments(
        oracle,
        observations,
        LABELS,
        allow_analysis_only=True,
    )
    assert analysis_run.analysis_only
    assert analysis_run.commit_step(CandidateKey(EXAMPLE_ID, 0, 1, "person")) == 3
    assert HypothesisFamilyKey(EXAMPLE_ID, 0, "person") not in oracle.stable_steps
