"""Validated ownership and lifecycle for one native MLX streaming session.

The Qwen adapter owns validation of MLX-LM's native KV objects.  This module owns
the higher-level state that KV alone cannot represent: exact word coordinates,
unprocessed word states, cached pre-projection label representations, and the
complete historical span-logit map.

Historical logits cross the device boundary once and are stored as private,
read-only, owned NumPy arrays.  No PyTorch or GLiNER package is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from streamner_commit.mlx.preprocessing import normalize_labels

Boundary = tuple[int, int]
FloatArray = NDArray[np.float32]


class StreamingStateError(ValueError):
    """A streaming session violates an ownership or shape invariant."""


class ContextLimitError(StreamingStateError):
    """An append would exceed the configured Qwen context."""


class SessionLabelsError(StreamingStateError):
    """An append attempted to change the labels of a live session."""


class ReleasedStateError(RuntimeError):
    """A released session state was accessed again."""


@runtime_checkable
class QwenCacheOwner(Protocol):
    """Narrow QwenAdapter surface needed by the streaming state."""

    configuration: Any

    def validate_cache(
        self,
        cache: Sequence[Any],
        *,
        expected_offset: int | None = None,
    ) -> int: ...


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamingStateError(f"{name} must be an integer")
    if value < 0:
        raise StreamingStateError(f"{name} must be nonnegative")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    result = _require_nonnegative_int(value, name=name)
    if result == 0:
        raise StreamingStateError(f"{name} must be positive")
    return result


def _shape(value: object, *, name: str, rank: int) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple | list):
        raise StreamingStateError(f"{name} must be an array with a shape")
    result: list[int] = []
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise StreamingStateError(f"{name} contains an invalid shape dimension")
        result.append(dimension)
    if len(result) != rank:
        raise StreamingStateError(f"{name} must have rank {rank}, got shape {tuple(result)}")
    return tuple(result)


def _require_floating_array(value: object, *, name: str, rank: int) -> tuple[int, ...]:
    shape = _shape(value, name=name, rank=rank)
    dtype = getattr(value, "dtype", None)
    dtype_name = str(dtype).lower()
    if "float" not in dtype_name and "bfloat" not in dtype_name:
        raise StreamingStateError(f"{name} must have a floating-point dtype, got {dtype}")
    return shape


def _mask_values(value: object, *, name: str, rank: int) -> NDArray[np.bool_]:
    expected_shape = _shape(value, name=name, rank=rank)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise StreamingStateError(f"{name} cannot be copied to CPU: {exc}") from exc
    if tuple(raw.shape) != expected_shape:
        raise StreamingStateError(f"{name} changed shape while materializing")
    if raw.dtype.kind not in {"b", "i", "u"}:
        raise StreamingStateError(f"{name} must be boolean or zero/one integer data")
    if raw.size and not np.isin(raw, (0, 1)).all():
        raise StreamingStateError(f"{name} must contain only zero/one values")
    return np.ascontiguousarray(raw, dtype=np.bool_)


def _normalized_labels(labels: Sequence[str]) -> tuple[str, ...]:
    try:
        values = tuple(labels)
        normalized = normalize_labels(values)
    except (TypeError, ValueError) as exc:
        raise StreamingStateError(str(exc)) from exc
    if normalized != values:
        raise StreamingStateError("session labels must be normalized, unique, and ordered")
    return values


def _validated_boundary(boundary: object, *, word_count: int) -> Boundary:
    if (
        not isinstance(boundary, tuple)
        or len(boundary) != 2
        or isinstance(boundary[0], bool)
        or not isinstance(boundary[0], int)
        or isinstance(boundary[1], bool)
        or not isinstance(boundary[1], int)
    ):
        raise StreamingStateError("historical boundaries must be (start, end) integer tuples")
    start, end = boundary
    if not 0 <= start <= end < word_count:
        raise StreamingStateError(
            f"historical boundary {(start, end)} is outside {word_count} visible words"
        )
    return start, end


def _owned_score_vector(value: object, *, class_count: int, name: str) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise StreamingStateError(f"{name} cannot be copied to CPU: {exc}") from exc
    if raw.ndim != 1 or raw.shape[0] != class_count:
        raise StreamingStateError(
            f"{name} must be one complete {class_count}-class logit vector, got {raw.shape}"
        )
    if raw.dtype.kind not in {"f", "i", "u"}:
        raise StreamingStateError(f"{name} must contain real numeric logits")
    if not np.isfinite(raw).all():
        raise StreamingStateError(f"{name} contains non-finite logits")
    result = np.ascontiguousarray(raw, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _owned_history(
    history: Mapping[Boundary, object] | None,
    *,
    class_count: int,
    word_count: int,
) -> dict[Boundary, FloatArray]:
    if history is None:
        return {}
    if not isinstance(history, Mapping):
        raise StreamingStateError("historical logits must be a boundary mapping")
    copied: dict[Boundary, FloatArray] = {}
    for raw_boundary, scores in history.items():
        boundary = _validated_boundary(raw_boundary, word_count=word_count)
        copied[boundary] = _owned_score_vector(
            scores,
            class_count=class_count,
            name=f"historical logits for {boundary}",
        )
    return copied


class MLXStreamingState:
    """All reusable, single-example state for one append-only MLX session.

    The object is mutable because MLX-LM updates native KV objects in place.  Callers
    must check context capacity before a decoder call and commit the new token count
    immediately afterward.  If a model call fails after touching KV state, ``clear``
    must be used; the previous metadata cannot safely be retried.
    """

    __slots__ = (
        "_history",
        "_qwen",
        "_released",
        "context_limit",
        "labels",
        "past_word_embeddings",
        "past_word_mask",
        "prompt_mask",
        "prompt_representations",
        "qwen_cache",
        "session_id",
        "text",
        "token_count",
        "word_char_ends",
        "word_char_starts",
        "word_count",
        "word_tokens",
    )

    def __init__(
        self,
        *,
        session_id: str,
        qwen: QwenCacheOwner,
        qwen_cache: Sequence[Any],
        token_count: int,
        past_word_embeddings: Any,
        past_word_mask: Any,
        word_count: int,
        prompt_representations: Any,
        prompt_mask: Any,
        labels: Sequence[str],
        text: str,
        word_tokens: Sequence[str],
        word_char_starts: Sequence[int],
        word_char_ends: Sequence[int],
        historical_logits: Mapping[Boundary, object] | None = None,
        context_limit: int | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise StreamingStateError("session_id must be a nonblank string")
        if not isinstance(qwen, QwenCacheOwner):
            raise StreamingStateError("qwen must expose configuration and validate_cache")
        if isinstance(qwen_cache, str | bytes) or not isinstance(qwen_cache, Sequence):
            raise StreamingStateError("qwen_cache must be a sequence of native layer caches")

        configuration = qwen.configuration
        hidden_size = _require_positive_int(
            getattr(configuration, "hidden_size", None),
            name="Qwen hidden size",
        )
        backbone_limit = _require_positive_int(
            getattr(configuration, "max_position_embeddings", None),
            name="Qwen context limit",
        )
        resolved_limit = (
            backbone_limit
            if context_limit is None
            else _require_positive_int(context_limit, name="context_limit")
        )
        if resolved_limit > backbone_limit:
            raise StreamingStateError(
                f"context_limit {resolved_limit} exceeds Qwen limit {backbone_limit}"
            )

        self.session_id = session_id
        self._qwen: QwenCacheOwner | None = qwen
        self.qwen_cache = list(qwen_cache)
        self.token_count = _require_positive_int(token_count, name="token_count")
        self.context_limit = resolved_limit
        self.past_word_embeddings: Any | None = past_word_embeddings
        self.past_word_mask: Any | None = past_word_mask
        self.word_count = _require_positive_int(word_count, name="word_count")
        self.prompt_representations: Any | None = prompt_representations
        self.prompt_mask: Any | None = prompt_mask
        self.labels = _normalized_labels(labels)
        self.text = text
        self.word_tokens = tuple(word_tokens)
        self.word_char_starts = tuple(word_char_starts)
        self.word_char_ends = tuple(word_char_ends)
        self._released = False

        self._validate_context()
        self._validate_cache()
        self._validate_word_state(hidden_size)
        self._validate_prompt_state(hidden_size)
        self._validate_text_coordinates()
        self._history = _owned_history(
            historical_logits,
            class_count=len(self.labels),
            word_count=self.word_count,
        )
        self._validate_cross_field_invariants()

    @property
    def is_released(self) -> bool:
        """Whether model and prediction storage has been irreversibly released."""

        return self._released

    @property
    def token_offset(self) -> int:
        """Return the native Qwen cache offset, validated across every layer."""

        self._require_active()
        assert self._qwen is not None
        try:
            return self._qwen.validate_cache(self.qwen_cache)
        except Exception as exc:
            raise StreamingStateError(f"invalid native Qwen cache: {exc}") from exc

    @property
    def next_token_position(self) -> int:
        """Absolute RoPE position to use for the next decoder token."""

        self._require_active()
        return self.token_count

    @property
    def historical_logits(self) -> Mapping[Boundary, FloatArray]:
        """Return a detached, read-only snapshot of the CPU score history."""

        self._require_active()
        snapshot: dict[Boundary, FloatArray] = {}
        for boundary, values in self._history.items():
            copied = values.copy()
            copied.setflags(write=False)
            snapshot[boundary] = copied
        return MappingProxyType(snapshot)

    def _require_active(self) -> None:
        if self._released:
            raise ReleasedStateError(f"streaming session {self.session_id!r} has been released")

    def _validate_context(self) -> None:
        if self.token_count > self.context_limit:
            raise ContextLimitError(
                f"session {self.session_id!r} contains {self.token_count} decoder tokens, "
                f"exceeding its context limit of {self.context_limit}"
            )

    def _validate_cache(self) -> None:
        assert self._qwen is not None
        try:
            offset = self._qwen.validate_cache(
                self.qwen_cache,
                expected_offset=self.token_count,
            )
        except Exception as exc:
            raise StreamingStateError(f"invalid native Qwen cache: {exc}") from exc
        if offset != self.token_count:
            raise StreamingStateError(
                f"native Qwen offset {offset} differs from token_count {self.token_count}"
            )

    def _validate_word_state(self, hidden_size: int) -> None:
        embedding_shape = _require_floating_array(
            self.past_word_embeddings,
            name="past_word_embeddings",
            rank=3,
        )
        if embedding_shape[0] != 1 or embedding_shape[2] != hidden_size:
            raise StreamingStateError(
                "past_word_embeddings must have shape "
                f"(1, capacity, {hidden_size}), got {embedding_shape}"
            )
        mask = _mask_values(self.past_word_mask, name="past_word_mask", rank=2)
        if tuple(mask.shape) != embedding_shape[:2]:
            raise StreamingStateError("past_word_mask must match the word-state batch/capacity")
        if self.word_count > embedding_shape[1]:
            raise StreamingStateError("word_count exceeds word-state capacity")
        expected = np.arange(embedding_shape[1]) < self.word_count
        if not np.array_equal(mask[0], expected):
            raise StreamingStateError(
                "past_word_mask must be a contiguous valid prefix matching word_count"
            )

    def _validate_prompt_state(self, hidden_size: int) -> None:
        representation_shape = _require_floating_array(
            self.prompt_representations,
            name="prompt_representations",
            rank=3,
        )
        expected = (1, len(self.labels), hidden_size)
        if representation_shape != expected:
            raise StreamingStateError(
                f"prompt_representations must have shape {expected}, got {representation_shape}"
            )
        mask = _mask_values(self.prompt_mask, name="prompt_mask", rank=2)
        if tuple(mask.shape) != expected[:2] or not mask.all():
            raise StreamingStateError(
                "prompt_mask must mark exactly one cached representation per label"
            )

    def _validate_text_coordinates(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise StreamingStateError("active session text must be nonblank")
        if not (
            len(self.word_tokens)
            == len(self.word_char_starts)
            == len(self.word_char_ends)
            == self.word_count
        ):
            raise StreamingStateError(
                "word_count, word tokens, and character offsets must be aligned"
            )
        previous_end = 0
        for index, (word, start, end) in enumerate(
            zip(
                self.word_tokens,
                self.word_char_starts,
                self.word_char_ends,
                strict=True,
            )
        ):
            if not isinstance(word, str) or not word:
                raise StreamingStateError(f"word_tokens[{index}] must be nonempty")
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < previous_end
                or not start < end <= len(self.text)
                or self.text[start:end] != word
            ):
                raise StreamingStateError(
                    f"word coordinate {index} does not round-trip through accumulated text"
                )
            previous_end = end

    def _validate_cross_field_invariants(self) -> None:
        if self.token_count < self.word_count:
            raise StreamingStateError("token_count cannot be smaller than word_count")
        for boundary, scores in self._history.items():
            _validated_boundary(boundary, word_count=self.word_count)
            if scores.shape != (len(self.labels),) or scores.dtype != np.float32:
                raise StreamingStateError("historical logit storage is internally inconsistent")
            if scores.flags.writeable or not scores.flags.owndata:
                raise StreamingStateError("historical logits must be owned read-only CPU arrays")

    def validate(self) -> None:
        """Recheck every live-state invariant, including native KV offsets."""

        self._require_active()
        assert self._qwen is not None
        hidden_size = _require_positive_int(
            getattr(self._qwen.configuration, "hidden_size", None),
            name="Qwen hidden size",
        )
        self._validate_context()
        self._validate_cache()
        self._validate_word_state(hidden_size)
        self._validate_prompt_state(hidden_size)
        self._validate_text_coordinates()
        self._validate_cross_field_invariants()

    def ensure_labels(self, labels: Sequence[str]) -> tuple[str, ...]:
        """Normalize request labels and reject changes during a live session."""

        self._require_active()
        try:
            normalized = normalize_labels(labels)
        except (TypeError, ValueError) as exc:
            raise SessionLabelsError(str(exc)) from exc
        if normalized != self.labels:
            raise SessionLabelsError(
                f"labels for session {self.session_id!r} changed; clear the session first"
            )
        return normalized

    def ensure_context_capacity(self, additional_tokens: int) -> int:
        """Validate an append before native KV mutation and return its future offset."""

        self._require_active()
        additional = _require_nonnegative_int(
            additional_tokens,
            name="additional_tokens",
        )
        self._validate_cache()
        future = self.token_count + additional
        if future > self.context_limit:
            raise ContextLimitError(
                f"session {self.session_id!r} would contain {future} decoder tokens, "
                f"exceeding its context limit of {self.context_limit}; clear the session "
                "before continuing"
            )
        return future

    def commit_token_append(self, appended_tokens: int) -> int:
        """Commit metadata after Qwen has appended exactly ``appended_tokens`` IDs."""

        self._require_active()
        appended = _require_nonnegative_int(appended_tokens, name="appended_tokens")
        expected = self.token_count + appended
        if expected > self.context_limit:
            raise ContextLimitError(
                f"session {self.session_id!r} reached {expected} decoder tokens, "
                f"exceeding its context limit of {self.context_limit}"
            )
        assert self._qwen is not None
        try:
            actual = self._qwen.validate_cache(
                self.qwen_cache,
                expected_offset=expected,
            )
        except Exception as exc:
            raise StreamingStateError(
                "native Qwen cache did not advance by the declared token count; "
                "the session must be cleared"
            ) from exc
        if actual != expected:
            raise StreamingStateError(
                f"native Qwen offset {actual} differs from expected offset {expected}"
            )
        self.token_count = expected
        return expected

    def replace_word_history(
        self,
        *,
        past_word_embeddings: Any,
        past_word_mask: Any,
        word_count: int,
        text: str,
        word_tokens: Sequence[str],
        word_char_starts: Sequence[int],
        word_char_ends: Sequence[int],
    ) -> None:
        """Atomically install an append-only visible-word state after model success."""

        self._require_active()
        new_word_count = _require_positive_int(word_count, name="word_count")
        new_tokens = tuple(word_tokens)
        new_starts = tuple(word_char_starts)
        new_ends = tuple(word_char_ends)
        if new_word_count < self.word_count:
            raise StreamingStateError("word history cannot shrink during an append")
        if not isinstance(text, str) or not text.startswith(self.text):
            raise StreamingStateError("accumulated text must preserve the prior text prefix")
        if (
            new_tokens[: self.word_count] != self.word_tokens
            or new_starts[: self.word_count] != self.word_char_starts
            or new_ends[: self.word_count] != self.word_char_ends
        ):
            raise StreamingStateError("word history must preserve all prior words and offsets")
        if new_word_count == self.word_count and (
            text != self.text
            or new_tokens != self.word_tokens
            or new_starts != self.word_char_starts
            or new_ends != self.word_char_ends
        ):
            raise StreamingStateError(
                "an append with no new model words must not mutate accumulated text"
            )
        if new_word_count > self.word_count:
            if text == self.text:
                raise StreamingStateError("new model words require appended text")
            if any(start < len(self.text) for start in new_starts[self.word_count :]):
                raise StreamingStateError("new word offsets must begin in the appended text")

        assert self._qwen is not None
        hidden_size = _require_positive_int(
            getattr(self._qwen.configuration, "hidden_size", None),
            name="Qwen hidden size",
        )
        old_values = (
            self.past_word_embeddings,
            self.past_word_mask,
            self.word_count,
            self.text,
            self.word_tokens,
            self.word_char_starts,
            self.word_char_ends,
        )
        self.past_word_embeddings = past_word_embeddings
        self.past_word_mask = past_word_mask
        self.word_count = new_word_count
        self.text = text
        self.word_tokens = new_tokens
        self.word_char_starts = new_starts
        self.word_char_ends = new_ends
        try:
            self._validate_word_state(hidden_size)
            self._validate_text_coordinates()
            self._validate_cross_field_invariants()
        except Exception:
            (
                self.past_word_embeddings,
                self.past_word_mask,
                self.word_count,
                self.text,
                self.word_tokens,
                self.word_char_starts,
                self.word_char_ends,
            ) = old_values
            raise

    def merge_candidate_logits(
        self,
        boundaries: object,
        logits: object,
        mask: object | None = None,
        *,
        replace_all: bool = False,
    ) -> int:
        """Copy valid candidate vectors to CPU and replace their historical entries.

        Single-example flattened ``(N, C)``, batched ``(1, N, C)``, and marker-grid
        ``(1, words, widths, C)`` logits are accepted.  Boundaries and masks are
        flattened in the same deterministic order.  Invalid masked candidates are
        ignored, including their intentionally out-of-range ends.
        """

        self._require_active()
        if not isinstance(replace_all, bool):
            raise StreamingStateError("replace_all must be a boolean")
        try:
            boundary_values = np.asarray(boundaries)
            logit_values = np.asarray(logits)
        except (TypeError, ValueError) as exc:
            raise StreamingStateError(f"candidate arrays cannot be copied to CPU: {exc}") from exc
        if boundary_values.dtype.kind not in {"i", "u"}:
            raise StreamingStateError("candidate boundaries must contain integers")
        if boundary_values.ndim >= 3:
            if boundary_values.shape[0] != 1:
                raise StreamingStateError("streaming history accepts one boundary batch only")
            boundary_values = boundary_values.reshape(-1, 2)
        if boundary_values.ndim != 2 or boundary_values.shape[1] != 2:
            raise StreamingStateError("candidate boundaries must have shape (N, 2)")

        if logit_values.ndim >= 3:
            if logit_values.shape[0] != 1:
                raise StreamingStateError("streaming history accepts one logit batch only")
            logit_values = logit_values.reshape(-1, logit_values.shape[-1])
        if logit_values.ndim != 2 or logit_values.shape[1] != len(self.labels):
            raise StreamingStateError(f"candidate logits must have shape (N, {len(self.labels)})")
        if boundary_values.shape[0] != logit_values.shape[0]:
            raise StreamingStateError("candidate boundaries and logits must have equal row counts")

        if mask is None:
            valid = np.ones(boundary_values.shape[0], dtype=np.bool_)
        else:
            try:
                raw_mask = np.asarray(mask).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise StreamingStateError(f"candidate mask cannot be copied to CPU: {exc}") from exc
            if raw_mask.shape != (boundary_values.shape[0],):
                raise StreamingStateError("candidate mask must contain one value per row")
            if raw_mask.dtype.kind not in {"b", "i", "u"} or (
                raw_mask.size and not np.isin(raw_mask, (0, 1)).all()
            ):
                raise StreamingStateError("candidate mask must contain only zero/one values")
            valid = raw_mask.astype(np.bool_, copy=False)

        pending: list[tuple[Boundary, FloatArray]] = []
        seen: set[Boundary] = set()
        for row, scores in zip(boundary_values[valid], logit_values[valid], strict=True):
            raw_boundary = (int(row[0]), int(row[1]))
            boundary = _validated_boundary(raw_boundary, word_count=self.word_count)
            if boundary in seen:
                raise StreamingStateError(
                    f"candidate update contains duplicate boundary {boundary}"
                )
            seen.add(boundary)
            pending.append(
                (
                    boundary,
                    _owned_score_vector(
                        scores,
                        class_count=len(self.labels),
                        name=f"candidate logits for {boundary}",
                    ),
                )
            )

        merged = {} if replace_all else self._history.copy()
        merged.update(pending)
        self._history = merged
        self._validate_cross_field_invariants()
        return len(pending)

    def ordered_history_arrays(self) -> tuple[NDArray[np.int64], FloatArray]:
        """Return copied, sorted boundary and complete-logit matrices for decoding."""

        self._require_active()
        rows = sorted(self._history.items())
        if not rows:
            boundaries = np.empty((0, 2), dtype=np.int64)
            logits = np.empty((0, len(self.labels)), dtype=np.float32)
        else:
            boundaries = np.asarray([boundary for boundary, _ in rows], dtype=np.int64)
            logits = np.stack([scores for _, scores in rows]).astype(np.float32, copy=True)
        boundaries.setflags(write=False)
        logits.setflags(write=False)
        return boundaries, logits

    def clear(self) -> None:
        """Release native KV, device tensors, prompt state, and CPU history.

        Clearing is idempotent.  The session identifier is retained only for useful
        diagnostics; every model or prediction reference is removed.
        """

        if self._released:
            return
        for layer in self.qwen_cache:
            # MLX-LM 0.31.3 KVCache exposes these mutable fields.  Resetting them
            # explicitly releases unified-memory storage even if a caller accidentally
            # retained the individual cache object.
            if hasattr(layer, "keys"):
                layer.keys = None
            if hasattr(layer, "values"):
                layer.values = None
            if hasattr(layer, "offset"):
                layer.offset = 0
        self.qwen_cache.clear()
        self.past_word_embeddings = None
        self.past_word_mask = None
        self.prompt_representations = None
        self.prompt_mask = None
        self._history.clear()
        self.labels = ()
        self.text = ""
        self.word_tokens = ()
        self.word_char_starts = ()
        self.word_char_ends = ()
        self.token_count = 0
        self.word_count = 0
        self._qwen = None
        self._released = True

    def release(self) -> None:
        """Alias for ``clear`` emphasizing ownership release."""

        self.clear()


__all__ = [
    "Boundary",
    "ContextLimitError",
    "FloatArray",
    "MLXStreamingState",
    "QwenCacheOwner",
    "ReleasedStateError",
    "SessionLabelsError",
    "StreamingStateError",
]
