"""Exact candidate trajectories and hypothesis-family indexing.

The records in this module describe *actual* model observations.  In
particular, a cached span that is absent from a later ``SpanScoreUpdate`` does
not acquire another observation.  Deployable policies should consume the
prefix-only views produced by :mod:`streamner_commit.streaming.tracker`, not a
completed ``CandidateTrajectory``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby

from streamner_commit.types import SpanScoreUpdate, UpdateKind


class TrajectoryError(ValueError):
    """Raised when score updates cannot form deterministic trajectories."""


@dataclass(frozen=True, slots=True, order=True)
class CandidateKey:
    """Identity of one label-specific exact span candidate."""

    example_id: str
    start_word: int
    end_word: int
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a nonblank string")
        if isinstance(self.start_word, bool) or not isinstance(self.start_word, int):
            raise TypeError("start_word must be an integer")
        if isinstance(self.end_word, bool) or not isinstance(self.end_word, int):
            raise TypeError("end_word must be an integer")
        if self.start_word < 0 or self.end_word < self.start_word:
            raise ValueError("candidate word boundary is invalid")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a nonblank string")

    @property
    def family_key(self) -> HypothesisFamilyKey:
        """Return the online-safe primary family anchor."""
        return HypothesisFamilyKey(self.example_id, self.start_word, self.label)


@dataclass(frozen=True, slots=True, order=True)
class HypothesisFamilyKey:
    """Primary boundary-extension family: example, start word, and label."""

    example_id: str
    start_word: int
    label: str

    def __post_init__(self) -> None:
        # Reuse the exact-key validation without maintaining a second set of rules.
        CandidateKey(self.example_id, self.start_word, self.start_word, self.label)


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """One label score from one actual span forward-pass update."""

    key: CandidateKey
    run_id: str
    step: int
    chunk: str
    visible_char_count: int
    visible_word_count: int
    start_char: int
    end_char: int
    span_text: str
    label_index: int
    logit: float
    probability: float
    is_top_label: bool
    update_kind: UpdateKind
    tail_distance_words: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a nonblank string")
        for name in (
            "step",
            "visible_char_count",
            "visible_word_count",
            "start_char",
            "end_char",
            "label_index",
            "tail_distance_words",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.end_char <= self.start_char or self.end_char - self.start_char != len(
            self.span_text
        ):
            raise ValueError("observation character span must match span_text")
        for name in ("logit", "probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        if self.update_kind not in {"new", "rescore", "full"}:
            raise ValueError("update_kind must be new, rescore, or full")


@dataclass(frozen=True, slots=True)
class CandidateTrajectory:
    """All actual observations for one exact candidate.

    This completed record is useful for analysis.  It is deliberately not a
    field of the online policy views because it can contain future events.
    """

    key: CandidateKey
    observations: tuple[CandidateObservation, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("candidate trajectory must contain an observation")
        previous_step = -1
        for observation in observations:
            if not isinstance(observation, CandidateObservation):
                raise TypeError("observations must contain CandidateObservation values")
            if observation.key != self.key:
                raise ValueError("all trajectory observations must have the trajectory key")
            if observation.step <= previous_step:
                raise TrajectoryError("candidate observations must have strictly increasing steps")
            previous_step = observation.step
        if observations[0].update_kind == "rescore":
            raise TrajectoryError("a candidate trajectory cannot begin with a rescore")
        if any(item.update_kind == "new" for item in observations[1:]):
            raise TrajectoryError("a known candidate cannot receive another new update")
        object.__setattr__(self, "observations", observations)

    @property
    def first_seen_step(self) -> int:
        return self.observations[0].step

    @property
    def last_seen_step(self) -> int:
        return self.observations[-1].step

    @property
    def rescore_count(self) -> int:
        """Count actual score observations, including the initial observation."""
        return len(self.observations)

    @property
    def current_probability(self) -> float:
        return self.observations[-1].probability

    @property
    def previous_probability(self) -> float | None:
        if len(self.observations) < 2:
            return None
        return self.observations[-2].probability

    @property
    def probability_delta(self) -> float | None:
        previous = self.previous_probability
        if previous is None:
            return None
        return self.current_probability - previous

    @property
    def probability_deltas(self) -> tuple[float, ...]:
        probabilities = tuple(item.probability for item in self.observations)
        return tuple(
            current - previous
            for previous, current in zip(probabilities[:-1], probabilities[1:], strict=True)
        )

    def rolling_deltas(self, window: int = 3) -> tuple[float, ...]:
        """Return at most the most recent ``window`` actual-observation deltas."""
        _positive_int(window, name="window")
        return self.probability_deltas[-window:]


@dataclass(frozen=True, slots=True, order=True)
class HypothesisFamily:
    """Primary same-start boundary-extension family seen in a trace."""

    key: HypothesisFamilyKey
    members: tuple[CandidateKey, ...]

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if not members:
            raise ValueError("hypothesis family must contain a member")
        if tuple(sorted(set(members))) != members:
            raise ValueError("hypothesis family members must be unique and sorted")
        if any(member.family_key != self.key for member in members):
            raise ValueError("hypothesis family member has a different primary anchor")
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class AnalysisOverlapFamily:
    """Future-aware same-label overlap component for analysis only.

    Building these components examines completed trajectories.  The online
    tracker never imports or exposes this type.
    """

    example_id: str
    label: str
    members: tuple[CandidateKey, ...]


def normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze the model's ordered class labels."""
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be an ordered sequence of strings")
    normalized = tuple(labels)
    if not normalized:
        raise ValueError("labels must not be empty")
    if any(not isinstance(label, str) or not label.strip() for label in normalized):
        raise ValueError("labels must contain nonblank strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must be unique")
    return normalized


def explode_span_update(
    update: SpanScoreUpdate,
    labels: Sequence[str],
) -> tuple[CandidateObservation, ...]:
    """Explode one ordered class vector into label-specific observations."""
    if not isinstance(update, SpanScoreUpdate):
        raise TypeError("update must be a SpanScoreUpdate")
    ordered_labels = normalize_labels(labels)
    if len(update.probs) != len(ordered_labels):
        raise TrajectoryError("label count must equal the score-vector width")
    if ordered_labels[update.top_label_index] != update.top_label:
        raise TrajectoryError("ordered labels do not agree with update.top_label_index")

    return tuple(
        CandidateObservation(
            key=CandidateKey(
                update.example_id,
                update.start_word,
                update.end_word,
                label,
            ),
            run_id=update.run_id,
            step=update.step,
            chunk=update.chunk,
            visible_char_count=update.visible_char_count,
            visible_word_count=update.visible_word_count,
            start_char=update.start_char,
            end_char=update.end_char,
            span_text=update.span_text,
            label_index=label_index,
            logit=update.logits[label_index],
            probability=update.probs[label_index],
            is_top_label=label_index == update.top_label_index,
            update_kind=update.update_kind,
            tail_distance_words=update.tail_distance_words,
        )
        for label_index, label in enumerate(ordered_labels)
    )


def build_candidate_trajectories(
    updates: Iterable[SpanScoreUpdate],
    labels: Sequence[str],
) -> tuple[CandidateTrajectory, ...]:
    """Build deterministic exact trajectories from actual update events."""
    ordered_labels = normalize_labels(labels)
    if isinstance(updates, str | bytes) or not isinstance(updates, Iterable):
        raise TypeError("updates must be an iterable of SpanScoreUpdate records")
    prepared = tuple(updates)
    if not all(isinstance(update, SpanScoreUpdate) for update in prepared):
        raise TypeError("updates must contain only SpanScoreUpdate records")
    if not prepared:
        return ()

    run_ids = {update.run_id for update in prepared}
    if len(run_ids) != 1:
        raise TrajectoryError("one trajectory build cannot mix run IDs")
    previous_step_by_example: dict[str, int] = {}
    observations: list[CandidateObservation] = []
    seen_at_step: set[tuple[CandidateKey, int]] = set()
    for update in prepared:
        previous_step = previous_step_by_example.get(update.example_id, -1)
        if update.step < previous_step:
            raise TrajectoryError("updates must be step-ordered within each example")
        previous_step_by_example[update.example_id] = update.step
        for observation in explode_span_update(update, ordered_labels):
            marker = observation.key, observation.step
            if marker in seen_at_step:
                raise TrajectoryError("an exact candidate cannot be updated twice in one step")
            seen_at_step.add(marker)
            observations.append(observation)

    observations.sort(key=lambda item: (item.key, item.step))
    return tuple(
        CandidateTrajectory(key, tuple(group))
        for key, group in groupby(observations, key=lambda item: item.key)
    )


def build_hypothesis_families(
    trajectories: Iterable[CandidateTrajectory],
) -> tuple[HypothesisFamily, ...]:
    """Group exact candidates by the online-safe same-start family anchor."""
    prepared = _trajectory_tuple(trajectories)
    keys = sorted(
        (trajectory.key for trajectory in prepared),
        key=lambda key: (key.family_key, key),
    )
    return tuple(
        HypothesisFamily(family_key, tuple(group))
        for family_key, group in groupby(keys, key=lambda key: key.family_key)
    )


def build_analysis_overlap_families(
    trajectories: Iterable[CandidateTrajectory],
) -> tuple[AnalysisOverlapFamily, ...]:
    """Build future-aware same-label temporal-overlap components.

    This optional analysis helper intentionally consumes completed trajectories.
    It must not be used to produce deployable policy features.
    """
    prepared = _trajectory_tuple(trajectories)
    grouped: dict[tuple[str, str], list[CandidateKey]] = {}
    for trajectory in prepared:
        grouped.setdefault((trajectory.key.example_id, trajectory.key.label), []).append(
            trajectory.key
        )

    result: list[AnalysisOverlapFamily] = []
    for (example_id, label), raw_keys in sorted(grouped.items()):
        # Interval-union components implement transitive overlap deterministically.
        keys = sorted(set(raw_keys), key=lambda key: (key.start_word, key.end_word))
        component: list[CandidateKey] = []
        component_end = -1
        for key in keys:
            if component and key.start_word > component_end:
                result.append(
                    AnalysisOverlapFamily(example_id, label, tuple(sorted(component)))
                )
                component = []
            component.append(key)
            component_end = max(component_end, key.end_word)
        if component:
            result.append(AnalysisOverlapFamily(example_id, label, tuple(sorted(component))))
    return tuple(result)


def _trajectory_tuple(
    trajectories: Iterable[CandidateTrajectory],
) -> tuple[CandidateTrajectory, ...]:
    if isinstance(trajectories, str | bytes) or not isinstance(trajectories, Iterable):
        raise TypeError("trajectories must be an iterable of CandidateTrajectory records")
    prepared = tuple(trajectories)
    if not all(isinstance(item, CandidateTrajectory) for item in prepared):
        raise TypeError("trajectories must contain only CandidateTrajectory records")
    keys = tuple(item.key for item in prepared)
    if len(set(keys)) != len(keys):
        raise TrajectoryError("trajectories must have unique candidate keys")
    return prepared


def _positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "AnalysisOverlapFamily",
    "CandidateKey",
    "CandidateObservation",
    "CandidateTrajectory",
    "HypothesisFamily",
    "HypothesisFamilyKey",
    "TrajectoryError",
    "build_analysis_overlap_families",
    "build_candidate_trajectories",
    "build_hypothesis_families",
    "explode_span_update",
    "normalize_labels",
]
