"""Fixed confidence and fixed right-context-lag baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

from streamner_commit.policies.base import (
    ReadyCandidate,
    nonnegative_int,
    normalize_policy_labels,
    probability_threshold,
    ready_candidate,
    validate_policy_input,
)
from streamner_commit.streaming.tracker import CandidateState, StreamingObservation


@dataclass(slots=True)
class FixedThreshold:
    """Declare every current candidate at or above a fixed probability ready."""

    threshold: float
    name: str = field(init=False, default="fixed-threshold")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)

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
        if state.current_probability < self.threshold:
            return []
        return [ready_candidate(self.name, state)]


@dataclass(slots=True)
class FixedLag:
    """Fixed threshold gated by visible right-context words."""

    threshold: float
    lag_words: int
    name: str = field(init=False, default="fixed-lag")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)
        self.lag_words = nonnegative_int(self.lag_words, name="lag_words")

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
            or state.tail_distance_words < self.lag_words
        ):
            return []
        return [ready_candidate(self.name, state)]


FixedThresholdPolicy = FixedThreshold
FixedLagPolicy = FixedLag

__all__ = [
    "FixedLag",
    "FixedLagPolicy",
    "FixedThreshold",
    "FixedThresholdPolicy",
]
