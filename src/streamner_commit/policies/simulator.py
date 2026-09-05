"""Offline replay of online-only policies over cached streaming observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from streamner_commit.policies.base import (
    CommitmentPolicy,
    ReadyCandidate,
    normalize_policy_labels,
)
from streamner_commit.policies.resolver import (
    BlockedCandidate,
    CommitResolver,
    CommittedCandidate,
    ResolutionStep,
)
from streamner_commit.streaming.tracker import StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey


class SimulationError(ValueError):
    """Raised when a policy or observation sequence violates replay rules."""


@dataclass(frozen=True, slots=True)
class CommitmentRun:
    """Complete immutable output from one cached-trace policy replay."""

    policy_name: str
    analysis_only: bool
    resolutions: tuple[ResolutionStep, ...]

    @property
    def committed(self) -> tuple[CommittedCandidate, ...]:
        if not self.resolutions:
            return ()
        return self.resolutions[-1].all_committed

    @property
    def blocked(self) -> tuple[BlockedCandidate, ...]:
        return tuple(item for resolution in self.resolutions for item in resolution.blocked)

    @property
    def blocked_revision_count(self) -> int:
        if not self.resolutions:
            return 0
        return self.resolutions[-1].cumulative_blocked_revision_count

    def commit_step(self, key: CandidateKey) -> int | None:
        commit = next((item for item in self.committed if item.key == key), None)
        return None if commit is None else commit.commit_step


def simulate_commitments(
    policy: CommitmentPolicy,
    observations: Sequence[StreamingObservation],
    labels: Sequence[str],
    *,
    flat_ner: bool = True,
    allow_analysis_only: bool = False,
) -> CommitmentRun:
    """Replay one policy without invoking a model or exposing benchmark oracles."""
    if not isinstance(policy, CommitmentPolicy):
        raise TypeError("policy must implement CommitmentPolicy")
    if not isinstance(flat_ner, bool) or not isinstance(allow_analysis_only, bool):
        raise TypeError("flat_ner and allow_analysis_only must be boolean")
    if policy.analysis_only and not allow_analysis_only:
        raise SimulationError("analysis-only policy requires allow_analysis_only=True")
    ordered_labels = normalize_policy_labels(labels)
    prepared = _observation_tuple(observations)
    _validate_trace(prepared, ordered_labels)

    policy.reset(list(ordered_labels))
    resolver = CommitResolver(flat_ner=flat_ner)
    resolutions: list[ResolutionStep] = []
    for observation in prepared:
        ready: list[ReadyCandidate] = []
        for state in observation.candidates:
            if resolver.is_committed(state.key):
                continue
            decisions = policy.observe(observation, state)
            if not isinstance(decisions, list) or not all(
                isinstance(item, ReadyCandidate) for item in decisions
            ):
                raise SimulationError("policy.observe must return a list of ReadyCandidate values")
            if any(item.key != state.key or item.step != observation.step for item in decisions):
                raise SimulationError("policy returned readiness for a different candidate or step")
            ready.extend(decisions)
        resolutions.append(resolver.resolve(observation, ready))
    return CommitmentRun(
        policy_name=policy.name,
        analysis_only=policy.analysis_only,
        resolutions=tuple(resolutions),
    )


def _observation_tuple(
    observations: Sequence[StreamingObservation],
) -> tuple[StreamingObservation, ...]:
    if isinstance(observations, str | bytes) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence of StreamingObservation records")
    prepared = tuple(observations)
    if not all(isinstance(item, StreamingObservation) for item in prepared):
        raise TypeError("observations must contain only StreamingObservation records")
    return prepared


def _validate_trace(
    observations: tuple[StreamingObservation, ...],
    labels: tuple[str, ...],
) -> None:
    if not observations:
        return
    run_id = observations[0].run_id
    example_id = observations[0].example_id
    previous_step = -1
    for observation in observations:
        if observation.run_id != run_id or observation.example_id != example_id:
            raise SimulationError("one simulation cannot mix traces")
        if observation.step <= previous_step:
            raise SimulationError("observation steps must be strictly increasing")
        if any(candidate.key.label not in labels for candidate in observation.candidates):
            raise SimulationError("observation contains a label absent from labels")
        previous_step = observation.step


__all__ = ["CommitmentRun", "SimulationError", "simulate_commitments"]
