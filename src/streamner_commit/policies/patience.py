"""Snapshot-persistence and actual-rescore patience baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamner_commit.policies.base import (
    ReadyCandidate,
    normalize_policy_labels,
    positive_int,
    probability_threshold,
    ready_candidate,
    validate_policy_input,
)
from streamner_commit.streaming.tracker import CandidateState, StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey


@dataclass(slots=True)
class SnapshotPatience:
    """Require eligibility in consecutive snapshots, including cached ones."""

    threshold: float
    patience: int
    name: str = field(init=False, default="snapshot-patience")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())
    _counts: dict[CandidateKey, int] = field(init=False, repr=False, default_factory=dict)
    _last_steps: dict[CandidateKey, int] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)
        self.patience = positive_int(self.patience, name="patience")

    def reset(self, labels: list[str]) -> None:
        self._labels = normalize_policy_labels(labels)
        self._counts.clear()
        self._last_steps.clear()

    def observe(
        self,
        step: StreamingObservation,
        state: CandidateState,
    ) -> list[ReadyCandidate]:
        validate_policy_input(step, state)
        _validate_label_and_step(state, self._labels, self._last_steps)
        self._last_steps[state.key] = step.step
        if state.current_probability >= self.threshold:
            self._counts[state.key] = self._counts.get(state.key, 0) + 1
        else:
            self._counts[state.key] = 0
        if self._counts[state.key] < self.patience:
            return []
        return [ready_candidate(self.name, state)]

    def eligible_snapshot_count(self, key: CandidateKey) -> int:
        return self._counts.get(key, 0)


@dataclass(slots=True)
class RescorePatience:
    """Require a count of actual model observations, never cached snapshots."""

    threshold: float
    patience: int
    name: str = field(init=False, default="rescore-patience")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)
        self.patience = positive_int(self.patience, name="patience")

    def reset(self, labels: list[str]) -> None:
        self._labels = normalize_policy_labels(labels)

    def observe(
        self,
        step: StreamingObservation,
        state: CandidateState,
    ) -> list[ReadyCandidate]:
        validate_policy_input(step, state)
        if state.key.label not in self._labels:
            raise ValueError("candidate label was not declared at reset")
        if (
            state.current_probability < self.threshold
            or state.rescore_count < self.patience
        ):
            return []
        return [ready_candidate(self.name, state)]


def _validate_label_and_step(
    state: CandidateState,
    labels: tuple[str, ...],
    last_steps: dict[CandidateKey, int],
) -> None:
    if state.key.label not in labels:
        raise ValueError("candidate label was not declared at reset")
    previous_step = last_steps.get(state.key)
    if previous_step is not None and state.step <= previous_step:
        raise ValueError("a stateful policy cannot observe one candidate twice at a step")


SnapshotPatiencePolicy = SnapshotPatience
RescorePatiencePolicy = RescorePatience

__all__ = [
    "RescorePatience",
    "RescorePatiencePolicy",
    "SnapshotPatience",
    "SnapshotPatiencePolicy",
]
