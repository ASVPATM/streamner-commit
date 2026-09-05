"""Backend-independent commitment-policy interface and shared records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from streamner_commit.streaming.tracker import CandidateState, StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey


class PolicyInputError(ValueError):
    """Raised when a policy is given an inconsistent online view."""


@dataclass(frozen=True, slots=True)
class ReadyCandidate:
    """A policy's immutable readiness decision for one current candidate."""

    key: CandidateKey
    step: int
    probability: float
    readiness_score: float
    policy_name: str
    first_seen_step: int
    rescore_count: int
    tail_distance_words: int
    start_char: int
    end_char: int
    span_text: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        for name in ("probability", "readiness_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be nonblank")
        for name in (
            "first_seen_step",
            "rescore_count",
            "tail_distance_words",
            "start_char",
            "end_char",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.rescore_count < 1:
            raise ValueError("rescore_count must be positive")
        if self.end_char <= self.start_char or self.end_char - self.start_char != len(
            self.span_text
        ):
            raise ValueError("candidate character span must match span_text")


@runtime_checkable
class CommitmentPolicy(Protocol):
    """Interface for deployable prefix-only readiness policies."""

    name: str
    analysis_only: bool

    def reset(self, labels: list[str]) -> None:
        """Clear policy state for a new trace."""
        ...

    def observe(
        self,
        step: StreamingObservation,
        state: CandidateState,
    ) -> list[ReadyCandidate]:
        """Return readiness decisions using only this immutable prefix view."""
        ...


def ready_candidate(
    policy_name: str,
    state: CandidateState,
    *,
    readiness_score: float | None = None,
) -> ReadyCandidate:
    """Copy a current candidate into an immutable readiness decision."""
    score = state.current_probability if readiness_score is None else readiness_score
    return ReadyCandidate(
        key=state.key,
        step=state.step,
        probability=state.current_probability,
        readiness_score=score,
        policy_name=policy_name,
        first_seen_step=state.first_seen_step,
        rescore_count=state.rescore_count,
        tail_distance_words=state.tail_distance_words,
        start_char=state.start_char,
        end_char=state.end_char,
        span_text=state.span_text,
    )


def validate_policy_input(step: StreamingObservation, state: CandidateState) -> None:
    """Prove that ``state`` is one of the current prefix's copied states."""
    if not isinstance(step, StreamingObservation):
        raise TypeError("step must be a StreamingObservation")
    if not isinstance(state, CandidateState):
        raise TypeError("state must be a CandidateState")
    if state.step != step.step or state.key.example_id != step.example_id:
        raise PolicyInputError("candidate state does not belong to this step")
    current = step.candidate(state.key)
    if current != state:
        raise PolicyInputError("candidate state is not present in this exact prefix view")


def normalize_policy_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Validate an ordered class-label list passed to ``reset``."""
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be an ordered sequence")
    result = tuple(labels)
    if not result or any(not isinstance(label, str) or not label.strip() for label in result):
        raise ValueError("labels must contain nonblank strings")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return result


def probability_threshold(value: float, *, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


def positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


__all__ = [
    "CommitmentPolicy",
    "PolicyInputError",
    "ReadyCandidate",
    "nonnegative_int",
    "normalize_policy_labels",
    "positive_int",
    "probability_threshold",
    "ready_candidate",
    "validate_policy_input",
]
