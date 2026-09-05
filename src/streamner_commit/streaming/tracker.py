"""Prefix-only online candidate feature tracking.

``CandidateTracker.observe`` is the only mutation boundary.  Its inputs are a
current public snapshot and the score events generated at that same step; it
has no parameter or storage slot for gold entities, a cold-full result, or
future updates.  Returned records copy the current prefix state and therefore
remain unchanged as later steps are observed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from streamner_commit.streaming.trajectory import (
    CandidateKey,
    CandidateObservation,
    HypothesisFamilyKey,
    TrajectoryError,
    explode_span_update,
    normalize_labels,
)
from streamner_commit.types import SnapshotStep, SpanScoreUpdate, UpdateKind


class CandidateTrackingError(ValueError):
    """Raised when a snapshot/update prefix is inconsistent."""


@dataclass(frozen=True, slots=True)
class CandidateState:
    """Online-safe features for one exact candidate at one snapshot."""

    key: CandidateKey
    family_key: HypothesisFamilyKey
    step: int
    first_seen_step: int
    last_seen_step: int
    snapshot_age: int
    rescore_count: int
    current_probability: float
    previous_probability: float | None
    probability_delta: float | None
    rolling_deltas: tuple[float, ...]
    current_logit: float
    is_top_label: bool
    was_rescored: bool
    last_update_kind: UpdateKind
    start_char: int
    end_char: int
    span_text: str
    visible_char_count: int
    visible_word_count: int
    tail_distance_words: int


@dataclass(frozen=True, slots=True)
class HypothesisFamilyState:
    """Members of one primary family that have appeared in this prefix."""

    key: HypothesisFamilyKey
    member_keys: tuple[CandidateKey, ...]
    observed_end_words: tuple[int, ...]
    longest_member_key: CandidateKey

    def __post_init__(self) -> None:
        if not self.member_keys:
            raise ValueError("family state must contain a member")
        if tuple(sorted(set(self.member_keys))) != self.member_keys:
            raise ValueError("family member keys must be unique and sorted")
        if any(member.family_key != self.key for member in self.member_keys):
            raise ValueError("family state member has a different primary anchor")
        expected_ends = tuple(sorted({member.end_word for member in self.member_keys}))
        if self.observed_end_words != expected_ends:
            raise ValueError("observed_end_words must match the family members")
        if self.longest_member_key not in self.member_keys:
            raise ValueError("longest_member_key must be a family member")
        if self.longest_member_key.end_word != max(self.observed_end_words):
            raise ValueError("longest_member_key must have the greatest observed end word")


@dataclass(frozen=True, slots=True)
class StreamingObservation:
    """One immutable deployable-policy view derived from a trace prefix only."""

    run_id: str
    example_id: str
    step: int
    chunk: str
    visible_char_count: int
    visible_word_count: int
    candidates: tuple[CandidateState, ...]
    families: tuple[HypothesisFamilyState, ...]
    rescored_keys: tuple[CandidateKey, ...]
    _candidate_by_key: Mapping[CandidateKey, CandidateState] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _family_by_key: Mapping[HypothesisFamilyKey, HypothesisFamilyState] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _boundary_candidates: Mapping[tuple[int, int], tuple[CandidateState, ...]] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        candidates = {state.key: state for state in self.candidates}
        if len(candidates) != len(self.candidates):
            raise ValueError("streaming observation candidates must have unique keys")
        object.__setattr__(self, "_candidate_by_key", MappingProxyType(candidates))
        families = {state.key: state for state in self.families}
        if len(families) != len(self.families):
            raise ValueError("streaming observation families must have unique keys")
        object.__setattr__(self, "_family_by_key", MappingProxyType(families))
        boundaries: dict[tuple[int, int], list[CandidateState]] = {}
        for state in self.candidates:
            boundaries.setdefault((state.key.start_word, state.key.end_word), []).append(state)
        object.__setattr__(
            self,
            "_boundary_candidates",
            MappingProxyType({key: tuple(values) for key, values in boundaries.items()}),
        )

    def candidate(self, key: CandidateKey) -> CandidateState | None:
        """Return an exact candidate state from this snapshot, if present."""
        return self._candidate_by_key.get(key)

    def family(self, key: HypothesisFamilyKey) -> HypothesisFamilyState | None:
        """Return a primary family state from this snapshot, if present."""
        return self._family_by_key.get(key)

    def candidates_for_boundary(self, start_word: int, end_word: int) -> tuple[CandidateState, ...]:
        """Return all current label candidates for one exact word boundary."""
        return self._boundary_candidates.get((start_word, end_word), ())


@dataclass(slots=True)
class _MutableCandidate:
    key: CandidateKey
    first_seen_step: int
    last_seen_step: int
    last_seen_snapshot_index: int
    probabilities: list[float]
    deltas: list[float]
    current_logit: float
    is_top_label: bool
    last_update_kind: UpdateKind
    start_char: int
    end_char: int
    span_text: str

    @classmethod
    def from_observation(
        cls,
        observation: CandidateObservation,
        *,
        snapshot_index: int,
    ) -> _MutableCandidate:
        return cls(
            key=observation.key,
            first_seen_step=observation.step,
            last_seen_step=observation.step,
            last_seen_snapshot_index=snapshot_index,
            probabilities=[observation.probability],
            deltas=[],
            current_logit=observation.logit,
            is_top_label=observation.is_top_label,
            last_update_kind=observation.update_kind,
            start_char=observation.start_char,
            end_char=observation.end_char,
            span_text=observation.span_text,
        )

    def append(self, observation: CandidateObservation, *, snapshot_index: int) -> None:
        if (
            observation.start_char,
            observation.end_char,
            observation.span_text,
        ) != (self.start_char, self.end_char, self.span_text):
            raise CandidateTrackingError("an exact candidate changed character coordinates or text")
        self.deltas.append(observation.probability - self.probabilities[-1])
        self.probabilities.append(observation.probability)
        self.last_seen_step = observation.step
        self.last_seen_snapshot_index = snapshot_index
        self.current_logit = observation.logit
        self.is_top_label = observation.is_top_label
        self.last_update_kind = observation.update_kind

    def freeze(
        self,
        *,
        snapshot: SnapshotStep,
        snapshot_index: int,
        rolling_window: int,
        was_rescored: bool,
    ) -> CandidateState:
        previous = self.probabilities[-2] if len(self.probabilities) >= 2 else None
        delta = self.deltas[-1] if self.deltas else None
        return CandidateState(
            key=self.key,
            family_key=self.key.family_key,
            step=snapshot.step,
            first_seen_step=self.first_seen_step,
            last_seen_step=self.last_seen_step,
            snapshot_age=snapshot_index - self.last_seen_snapshot_index,
            rescore_count=len(self.probabilities),
            current_probability=self.probabilities[-1],
            previous_probability=previous,
            probability_delta=delta,
            rolling_deltas=tuple(self.deltas[-rolling_window:]),
            current_logit=self.current_logit,
            is_top_label=self.is_top_label,
            was_rescored=was_rescored,
            last_update_kind=self.last_update_kind,
            start_char=self.start_char,
            end_char=self.end_char,
            span_text=self.span_text,
            visible_char_count=snapshot.visible_char_count,
            visible_word_count=snapshot.visible_word_count,
            tail_distance_words=(snapshot.visible_word_count - 1) - self.key.end_word,
        )


@dataclass(slots=True)
class CandidateTracker:
    """Incrementally turn span-update events into deployable online features."""

    labels: Sequence[str]
    rolling_window: int = 3
    _ordered_labels: tuple[str, ...] = field(init=False, repr=False)
    _states: dict[CandidateKey, _MutableCandidate] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _run_id: str | None = field(init=False, repr=False, default=None)
    _example_id: str | None = field(init=False, repr=False, default=None)
    _last_step: int | None = field(init=False, repr=False, default=None)
    _snapshot_index: int = field(init=False, repr=False, default=-1)

    def __post_init__(self) -> None:
        self._ordered_labels = normalize_labels(self.labels)
        self.labels = self._ordered_labels
        _positive_int(self.rolling_window, name="rolling_window")

    def observe(
        self,
        snapshot: SnapshotStep,
        updates: Iterable[SpanScoreUpdate],
    ) -> StreamingObservation:
        """Consume one current step and return a copied prefix-only view."""
        if not isinstance(snapshot, SnapshotStep):
            raise TypeError("snapshot must be a SnapshotStep")
        prepared_updates = _update_tuple(updates)
        self._validate_snapshot(snapshot)
        observations = self._validate_and_explode(snapshot, prepared_updates)

        next_snapshot_index = self._snapshot_index + 1
        rescored_keys: set[CandidateKey] = set()
        for observation in observations:
            mutable = self._states.get(observation.key)
            if mutable is None:
                self._states[observation.key] = _MutableCandidate.from_observation(
                    observation,
                    snapshot_index=next_snapshot_index,
                )
            else:
                mutable.append(observation, snapshot_index=next_snapshot_index)
            rescored_keys.add(observation.key)

        self._run_id = snapshot.run_id
        self._example_id = snapshot.example_id
        self._last_step = snapshot.step
        self._snapshot_index = next_snapshot_index

        candidate_states = tuple(
            self._states[key].freeze(
                snapshot=snapshot,
                snapshot_index=self._snapshot_index,
                rolling_window=self.rolling_window,
                was_rescored=key in rescored_keys,
            )
            for key in sorted(self._states)
        )
        family_states = _family_states(candidate_states)
        return StreamingObservation(
            run_id=snapshot.run_id,
            example_id=snapshot.example_id,
            step=snapshot.step,
            chunk=snapshot.chunk,
            visible_char_count=snapshot.visible_char_count,
            visible_word_count=snapshot.visible_word_count,
            candidates=candidate_states,
            families=family_states,
            rescored_keys=tuple(sorted(rescored_keys)),
        )

    def _validate_snapshot(self, snapshot: SnapshotStep) -> None:
        if self._run_id is not None and snapshot.run_id != self._run_id:
            raise CandidateTrackingError("a tracker cannot mix run IDs")
        if self._example_id is not None and snapshot.example_id != self._example_id:
            raise CandidateTrackingError("a tracker cannot mix examples")
        if self._last_step is not None and snapshot.step <= self._last_step:
            raise CandidateTrackingError("snapshot steps must be strictly increasing")

    def _validate_and_explode(
        self,
        snapshot: SnapshotStep,
        updates: tuple[SpanScoreUpdate, ...],
    ) -> tuple[CandidateObservation, ...]:
        observations: list[CandidateObservation] = []
        seen: set[CandidateKey] = set()
        for update in updates:
            if update.run_id != snapshot.run_id or update.example_id != snapshot.example_id:
                raise CandidateTrackingError("updates must belong to the current snapshot trace")
            if update.step != snapshot.step:
                raise CandidateTrackingError("updates must belong to the current snapshot step")
            if (
                update.chunk != snapshot.chunk
                or update.visible_char_count != snapshot.visible_char_count
                or update.visible_word_count != snapshot.visible_word_count
            ):
                raise CandidateTrackingError("update metadata must match the current snapshot")
            if snapshot.accumulated_text[update.start_char : update.end_char] != update.span_text:
                raise CandidateTrackingError("update span text does not match the visible prefix")
            try:
                exploded = explode_span_update(update, self._ordered_labels)
            except TrajectoryError as error:
                raise CandidateTrackingError(str(error)) from error
            for observation in exploded:
                if observation.key in seen:
                    raise CandidateTrackingError("a candidate cannot be updated twice in one step")
                seen.add(observation.key)
                known = observation.key in self._states
                if observation.update_kind == "new" and known:
                    raise CandidateTrackingError("new update repeats an existing candidate")
                if observation.update_kind == "rescore" and not known:
                    raise CandidateTrackingError("rescore update references an unseen candidate")
                if known:
                    prior = self._states[observation.key]
                    if (
                        observation.start_char,
                        observation.end_char,
                        observation.span_text,
                    ) != (prior.start_char, prior.end_char, prior.span_text):
                        raise CandidateTrackingError(
                            "an exact candidate changed character coordinates or text"
                        )
                observations.append(observation)
        return tuple(observations)


def build_streaming_observations(
    snapshots: Iterable[SnapshotStep],
    updates: Iterable[SpanScoreUpdate],
    labels: Sequence[str],
    *,
    rolling_window: int = 3,
) -> tuple[StreamingObservation, ...]:
    """Build immutable online views for every snapshot in one trace.

    Rebuilding a prefix produces byte-for-byte equivalent records for its
    shared steps: no record contains a reference to later observations.
    """
    prepared_snapshots = _snapshot_tuple(snapshots)
    prepared_updates = _update_tuple(updates)
    updates_by_step: dict[int, list[SpanScoreUpdate]] = {}
    for update in prepared_updates:
        updates_by_step.setdefault(update.step, []).append(update)

    tracker = CandidateTracker(labels, rolling_window=rolling_window)
    result = tuple(
        tracker.observe(snapshot, updates_by_step.pop(snapshot.step, ()))
        for snapshot in prepared_snapshots
    )
    if updates_by_step:
        raise CandidateTrackingError(
            f"updates reference missing snapshot steps: {sorted(updates_by_step)}"
        )
    return result


def _family_states(
    candidates: tuple[CandidateState, ...],
) -> tuple[HypothesisFamilyState, ...]:
    grouped: dict[HypothesisFamilyKey, list[CandidateKey]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.family_key, []).append(candidate.key)
    families: list[HypothesisFamilyState] = []
    for family_key in sorted(grouped):
        members = tuple(sorted(grouped[family_key]))
        longest = max(members, key=lambda key: (key.end_word, key.start_word, key.label))
        families.append(
            HypothesisFamilyState(
                key=family_key,
                member_keys=members,
                observed_end_words=tuple(sorted({member.end_word for member in members})),
                longest_member_key=longest,
            )
        )
    return tuple(families)


def _snapshot_tuple(snapshots: Iterable[SnapshotStep]) -> tuple[SnapshotStep, ...]:
    if isinstance(snapshots, str | bytes) or not isinstance(snapshots, Iterable):
        raise TypeError("snapshots must be an iterable of SnapshotStep records")
    prepared = tuple(snapshots)
    if not all(isinstance(snapshot, SnapshotStep) for snapshot in prepared):
        raise TypeError("snapshots must contain only SnapshotStep records")
    return prepared


def _update_tuple(updates: Iterable[SpanScoreUpdate]) -> tuple[SpanScoreUpdate, ...]:
    if isinstance(updates, str | bytes) or not isinstance(updates, Iterable):
        raise TypeError("updates must be an iterable of SpanScoreUpdate records")
    prepared = tuple(updates)
    if not all(isinstance(update, SpanScoreUpdate) for update in prepared):
        raise TypeError("updates must contain only SpanScoreUpdate records")
    return prepared


def _positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "CandidateState",
    "CandidateTracker",
    "CandidateTrackingError",
    "HypothesisFamilyState",
    "StreamingObservation",
    "build_streaming_observations",
]
