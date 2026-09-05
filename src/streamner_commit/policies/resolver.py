"""Deterministic flat-NER conflict resolution and immutable hard commits."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from streamner_commit.policies.base import ReadyCandidate
from streamner_commit.streaming.tracker import StreamingObservation
from streamner_commit.streaming.trajectory import CandidateKey

BlockReason = Literal["already_committed", "committed_overlap", "ready_overlap"]


class ResolutionError(ValueError):
    """Raised when readiness decisions do not belong to the current step."""


@dataclass(frozen=True, slots=True)
class CommittedCandidate:
    """A hard commitment; its copied decision is never revised in place."""

    decision: ReadyCandidate
    commit_step: int

    @property
    def key(self) -> CandidateKey:
        return self.decision.key


@dataclass(frozen=True, slots=True)
class BlockedCandidate:
    """One ready candidate rejected by an existing or same-step commitment."""

    decision: ReadyCandidate
    blocker_key: CandidateKey
    reason: BlockReason
    is_revision: bool


@dataclass(frozen=True, slots=True)
class ResolutionStep:
    """Resolver output for one streaming snapshot."""

    step: int
    newly_committed: tuple[CommittedCandidate, ...]
    all_committed: tuple[CommittedCandidate, ...]
    blocked: tuple[BlockedCandidate, ...]
    cumulative_blocked_revision_count: int

    @property
    def blocked_revision_count(self) -> int:
        return sum(item.is_revision for item in self.blocked)

    @property
    def blocked_conflict_count(self) -> int:
        return len(self.blocked)


@dataclass(slots=True)
class CommitResolver:
    """Rank readiness decisions, enforce overlap, and retain immutable commits."""

    flat_ner: bool = True
    _committed: dict[CandidateKey, CommittedCandidate] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _blocked_revision_count: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.flat_ner, bool):
            raise TypeError("flat_ner must be boolean")

    def reset(self) -> None:
        self._committed.clear()
        self._blocked_revision_count = 0

    def is_committed(self, key: CandidateKey) -> bool:
        return key in self._committed

    @property
    def committed(self) -> tuple[CommittedCandidate, ...]:
        return tuple(self._committed[key] for key in sorted(self._committed))

    @property
    def blocked_revision_count(self) -> int:
        return self._blocked_revision_count

    def resolve(
        self,
        step: StreamingObservation,
        ready: Iterable[ReadyCandidate],
    ) -> ResolutionStep:
        """Resolve one batch using score-descending, stable lexical tie breaks."""
        if not isinstance(step, StreamingObservation):
            raise TypeError("step must be a StreamingObservation")
        decisions = _ready_tuple(ready)
        seen: set[CandidateKey] = set()
        for decision in decisions:
            if decision.key in seen:
                raise ResolutionError("a candidate may appear at most once in a ready batch")
            seen.add(decision.key)
            _validate_current_decision(step, decision)

        preexisting_keys = frozenset(self._committed)
        new_commits: list[CommittedCandidate] = []
        blocked: list[BlockedCandidate] = []
        for decision in sorted(decisions, key=_rank_key):
            existing_exact = self._committed.get(decision.key)
            if existing_exact is not None:
                blocked.append(
                    BlockedCandidate(
                        decision,
                        existing_exact.key,
                        "already_committed",
                        False,
                    )
                )
                continue

            blocker = self._overlapping_commit(decision.key, preexisting_keys)
            if blocker is not None:
                blocked.append(
                    BlockedCandidate(
                        decision,
                        blocker.key,
                        "committed_overlap",
                        _is_revision(decision.key, blocker.key),
                    )
                )
                continue

            same_step_blocker = next(
                (
                    commit
                    for commit in new_commits
                    if self.flat_ner and _overlaps(decision.key, commit.key)
                ),
                None,
            )
            if same_step_blocker is not None:
                blocked.append(
                    BlockedCandidate(
                        decision,
                        same_step_blocker.key,
                        "ready_overlap",
                        _is_revision(decision.key, same_step_blocker.key),
                    )
                )
                continue

            commit = CommittedCandidate(decision, step.step)
            self._committed[decision.key] = commit
            new_commits.append(commit)

        current_revisions = sum(item.is_revision for item in blocked)
        self._blocked_revision_count += current_revisions
        return ResolutionStep(
            step=step.step,
            newly_committed=tuple(new_commits),
            all_committed=self.committed,
            blocked=tuple(blocked),
            cumulative_blocked_revision_count=self._blocked_revision_count,
        )

    def _overlapping_commit(
        self,
        key: CandidateKey,
        eligible_keys: frozenset[CandidateKey],
    ) -> CommittedCandidate | None:
        if not self.flat_ner:
            return None
        blockers = [
            commit
            for commit in self._committed.values()
            if commit.key in eligible_keys
            and _overlaps(key, commit.key)
        ]
        if not blockers:
            return None
        return min(blockers, key=lambda item: (item.commit_step, item.key))


def _rank_key(decision: ReadyCandidate) -> tuple[float, float, int, int, str, str]:
    return (
        -decision.readiness_score,
        -decision.probability,
        decision.key.start_word,
        decision.key.end_word,
        decision.key.label,
        decision.policy_name,
    )


def _overlaps(left: CandidateKey, right: CandidateKey) -> bool:
    return (
        left.example_id == right.example_id
        and left.start_word <= right.end_word
        and right.start_word <= left.end_word
    )


def _is_revision(candidate: CandidateKey, blocker: CandidateKey) -> bool:
    return (
        candidate != blocker
        and candidate.example_id == blocker.example_id
        and _overlaps(candidate, blocker)
    )


def _validate_current_decision(
    step: StreamingObservation,
    decision: ReadyCandidate,
) -> None:
    if decision.step != step.step or decision.key.example_id != step.example_id:
        raise ResolutionError("ready candidate does not belong to the current step")
    state = step.candidate(decision.key)
    if state is None:
        raise ResolutionError("ready candidate is absent from the current prefix")
    if not math.isclose(decision.probability, state.current_probability, abs_tol=1e-12):
        raise ResolutionError("ready candidate probability is stale")
    if (
        decision.start_char,
        decision.end_char,
        decision.span_text,
    ) != (state.start_char, state.end_char, state.span_text):
        raise ResolutionError("ready candidate coordinates are stale")


def _ready_tuple(ready: Iterable[ReadyCandidate]) -> tuple[ReadyCandidate, ...]:
    if isinstance(ready, str | bytes) or not isinstance(ready, Iterable):
        raise TypeError("ready must be an iterable of ReadyCandidate records")
    decisions = tuple(ready)
    if not all(isinstance(item, ReadyCandidate) for item in decisions):
        raise TypeError("ready must contain only ReadyCandidate records")
    return decisions


__all__ = [
    "BlockedCandidate",
    "BlockReason",
    "CommitResolver",
    "CommittedCandidate",
    "ResolutionError",
    "ResolutionStep",
]
