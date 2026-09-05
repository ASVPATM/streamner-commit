"""Exponential-moving-average baseline over actual score observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from streamner_commit.policies.base import (
    ReadyCandidate,
    normalize_policy_labels,
    probability_threshold,
    ready_candidate,
    validate_policy_input,
)
from streamner_commit.streaming.tracker import CandidateState, StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey


@dataclass(slots=True)
class EMA:
    """Smooth probabilities only when the model emits a fresh score event."""

    threshold: float
    alpha: float
    name: str = field(init=False, default="ema")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())
    _values: dict[CandidateKey, float] = field(init=False, repr=False, default_factory=dict)
    _last_steps: dict[CandidateKey, int] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int | float):
            raise TypeError("alpha must be a real number")
        self.alpha = float(self.alpha)
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be finite and in (0, 1]")

    def reset(self, labels: list[str]) -> None:
        self._labels = normalize_policy_labels(labels)
        self._values.clear()
        self._last_steps.clear()

    def observe(
        self,
        step: StreamingObservation,
        state: CandidateState,
    ) -> list[ReadyCandidate]:
        validate_policy_input(step, state)
        if state.key.label not in self._labels:
            raise ValueError("candidate label was not declared at reset")
        previous_step = self._last_steps.get(state.key)
        if previous_step is not None and step.step <= previous_step:
            raise ValueError("EMA cannot observe one candidate twice at a step")
        self._last_steps[state.key] = step.step

        previous_ema = self._values.get(state.key)
        if previous_ema is None:
            # Initialization is not repeated smoothing and is safe even if a
            # caller begins replay from a mid-trace cached snapshot.
            self._values[state.key] = state.current_probability
        elif state.was_rescored:
            self._values[state.key] = (
                self.alpha * state.current_probability + (1.0 - self.alpha) * previous_ema
            )
        ema = self._values[state.key]
        if ema < self.threshold:
            return []
        return [ready_candidate(self.name, state, readiness_score=ema)]

    def value(self, key: CandidateKey) -> float | None:
        """Return the current EMA for a candidate (primarily for diagnostics)."""
        return self._values.get(key)


EMAPolicy = EMA

__all__ = ["EMA", "EMAPolicy"]
