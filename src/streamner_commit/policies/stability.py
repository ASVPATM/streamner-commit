"""Online StabilityGate and a strictly separated future-aware oracle."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType

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


@dataclass(frozen=True, slots=True)
class StabilityGateConfig:
    """Externalizable, interpretable StabilityGate parameters."""

    tau: float
    min_rescores: int
    instability_horizon: int
    epsilon: float
    min_label_margin: float
    max_extension_advantage: float
    use_instability: bool = True
    use_label_margin: bool = True
    use_extension: bool = True
    min_tail_distance_words: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau", probability_threshold(self.tau, name="tau"))
        object.__setattr__(
            self,
            "min_rescores",
            positive_int(self.min_rescores, name="min_rescores"),
        )
        object.__setattr__(
            self,
            "instability_horizon",
            positive_int(self.instability_horizon, name="instability_horizon"),
        )
        for name in ("epsilon", "min_label_margin"):
            value = _finite_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        extension = _finite_float(
            self.max_extension_advantage,
            name="max_extension_advantage",
        )
        object.__setattr__(self, "max_extension_advantage", extension)
        for name in ("use_instability", "use_label_margin", "use_extension"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.min_tail_distance_words is not None:
            if (
                isinstance(self.min_tail_distance_words, bool)
                or not isinstance(self.min_tail_distance_words, int)
                or self.min_tail_distance_words < 0
            ):
                raise ValueError("min_tail_distance_words must be null or nonnegative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> StabilityGateConfig:
        if not isinstance(values, Mapping):
            raise TypeError("StabilityGate config must be a mapping")
        allowed = {
            "tau",
            "min_rescores",
            "instability_horizon",
            "epsilon",
            "min_label_margin",
            "max_extension_advantage",
            "use_instability",
            "use_label_margin",
            "use_extension",
            "min_tail_distance_words",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown StabilityGate config keys: {sorted(unknown)}")
        try:
            return cls(**dict(values))  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("StabilityGate config is missing or mistypes a field") from error


@dataclass(frozen=True, slots=True)
class StabilityFeatures:
    """Online-only feature values used by one gate decision."""

    confidence: float
    rescore_count: int
    recent_instability: float
    label_margin: float
    extension_advantage: float
    tail_distance_words: int


@dataclass(slots=True)
class StabilityGate:
    """Interpretable gate over confidence and observed revision behavior."""

    config: StabilityGateConfig
    name: str = field(init=False, default="stability-gate")
    analysis_only: bool = field(init=False, default=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        if not isinstance(self.config, StabilityGateConfig):
            raise TypeError("config must be a StabilityGateConfig")

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
        features = extract_stability_features(
            step,
            state,
            self.config.instability_horizon,
        )
        if features.confidence < self.config.tau:
            return []
        if features.rescore_count < self.config.min_rescores:
            return []
        if self.config.use_instability and features.recent_instability > self.config.epsilon:
            return []
        if self.config.use_label_margin and features.label_margin < self.config.min_label_margin:
            return []
        if (
            self.config.use_extension
            and features.extension_advantage > self.config.max_extension_advantage
        ):
            return []
        if (
            self.config.min_tail_distance_words is not None
            and features.tail_distance_words < self.config.min_tail_distance_words
        ):
            return []
        return [ready_candidate(self.name, state)]


def extract_stability_features(
    step: StreamingObservation,
    state: CandidateState,
    horizon: int,
) -> StabilityFeatures:
    """Derive gate features exclusively from one copied prefix view."""
    validate_policy_input(step, state)
    horizon = positive_int(horizon, name="horizon")
    required_deltas = min(horizon, state.rescore_count - 1)
    if len(state.rolling_deltas) < required_deltas:
        raise ValueError(
            "candidate view retained fewer actual deltas than the requested horizon; "
            "build trajectories with rolling_window >= horizon"
        )
    recent_deltas = state.rolling_deltas[-horizon:]
    instability = max((abs(delta) for delta in recent_deltas), default=0.0)

    same_boundary = tuple(
        candidate.current_probability
        for candidate in step.candidates_for_boundary(state.key.start_word, state.key.end_word)
    )
    if not same_boundary:
        raise ValueError("current boundary has no label scores")
    ordered_scores = sorted(same_boundary, reverse=True)
    label_margin = ordered_scores[0] - (ordered_scores[1] if len(ordered_scores) > 1 else 0.0)

    family = step.family(state.family_key)
    if family is None:
        extension_probabilities = tuple(
            candidate.current_probability
            for candidate in step.candidates
            if candidate.key.example_id == state.key.example_id
            and candidate.key.start_word == state.key.start_word
            and candidate.key.label == state.key.label
            and candidate.key.end_word > state.key.end_word
        )
    else:
        extension_probabilities = tuple(
            candidate.current_probability
            for key in family.member_keys
            if key.end_word > state.key.end_word
            if (candidate := step.candidate(key)) is not None
        )
    # Zero is the documented neutral value when no longer visible alternative exists.
    extension_advantage = (
        max(extension_probabilities) - state.current_probability if extension_probabilities else 0.0
    )
    return StabilityFeatures(
        confidence=state.current_probability,
        rescore_count=state.rescore_count,
        recent_instability=instability,
        label_margin=label_margin,
        extension_advantage=extension_advantage,
        tail_distance_words=state.tail_distance_words,
    )


def stability_gate_ablations(
    base: StabilityGateConfig,
    tail_distance_words: int | None = None,
) -> Mapping[str, StabilityGateConfig]:
    """Return the required named ablations from one frozen base config."""
    if not isinstance(base, StabilityGateConfig):
        raise TypeError("base must be a StabilityGateConfig")
    configurations = {
        "full": base,
        "minus_instability": replace(base, use_instability=False),
        "minus_label_margin": replace(base, use_label_margin=False),
        "minus_extension": replace(base, use_extension=False),
    }
    if tail_distance_words is not None:
        configurations["plus_tail_distance"] = replace(
            base,
            min_tail_distance_words=tail_distance_words,
        )
    return MappingProxyType(configurations)


@dataclass(slots=True)
class OracleStable:
    """ANALYSIS ONLY: use completed future observations to find stable suffixes.

    A final-threshold candidate is ready at the first prefix after which it is
    present and stays above the threshold in every remaining snapshot.  The
    generic simulator rejects this policy unless analysis-only execution is
    explicitly enabled.
    """

    future_observations: Sequence[StreamingObservation]
    threshold: float
    name: str = field(init=False, default="oracle-stable-analysis-only")
    analysis_only: bool = field(init=False, default=True)
    _observations: tuple[StreamingObservation, ...] = field(init=False, repr=False)
    _stable_steps: Mapping[CandidateKey, int] = field(init=False, repr=False)
    _labels: tuple[str, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        self.threshold = probability_threshold(self.threshold)
        if isinstance(self.future_observations, str | bytes) or not isinstance(
            self.future_observations,
            Sequence,
        ):
            raise TypeError("future_observations must be a sequence")
        observations = tuple(self.future_observations)
        if not observations or not all(
            isinstance(item, StreamingObservation) for item in observations
        ):
            raise ValueError("future_observations must contain streaming observations")
        _validate_completed_trace(observations)
        self._observations = observations
        self.future_observations = observations
        self._stable_steps = MappingProxyType(_oracle_stable_steps(observations, self.threshold))

    @property
    def stable_steps(self) -> Mapping[CandidateKey, int]:
        return self._stable_steps

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
        stable_step = self._stable_steps.get(state.key)
        if stable_step is None or step.step < stable_step:
            return []
        return [ready_candidate(self.name, state)]


def _oracle_stable_steps(
    observations: tuple[StreamingObservation, ...],
    threshold: float,
) -> dict[CandidateKey, int]:
    final_keys = {
        state.key for state in observations[-1].candidates if state.current_probability >= threshold
    }
    result: dict[CandidateKey, int] = {}
    for key in sorted(final_keys):
        for index, observation in enumerate(observations):
            candidate = observation.candidate(key)
            if candidate is None:
                continue
            suffix = tuple(item.candidate(key) for item in observations[index:])
            if all(item is not None and item.current_probability >= threshold for item in suffix):
                result[key] = observation.step
                break
    return result


def _validate_completed_trace(observations: tuple[StreamingObservation, ...]) -> None:
    run_id = observations[0].run_id
    example_id = observations[0].example_id
    previous_step = -1
    for observation in observations:
        if observation.run_id != run_id or observation.example_id != example_id:
            raise ValueError("future observations must describe one trace")
        if observation.step <= previous_step:
            raise ValueError("future observation steps must be strictly increasing")
        previous_step = observation.step


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


OracleStablePolicy = OracleStable

__all__ = [
    "OracleStable",
    "OracleStablePolicy",
    "StabilityFeatures",
    "StabilityGate",
    "StabilityGateConfig",
    "extract_stability_features",
    "stability_gate_ablations",
]
