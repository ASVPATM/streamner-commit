"""Transactional single-stream backend for the native MLX StreamingSpan port."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from streamner_commit.mlx.cache import Boundary, MLXStreamingState
from streamner_commit.mlx.decoder import sigmoid
from streamner_commit.mlx.preprocessing import (
    SessionPreprocessingState,
    normalize_labels,
)
from streamner_commit.types import ColdFullResult, PublicEntity, SpanBoundary

if TYPE_CHECKING:
    from streamner_commit.mlx.assets import AssetBundle


UpdateKind = Literal["new", "rescore"]


class MLXStreamingBackendError(RuntimeError):
    """The live MLX backend cannot safely complete an operation."""


class ClearedSessionError(MLXStreamingBackendError):
    """A caller attempted to reuse a cleared session."""


class InvalidatedSessionError(ClearedSessionError):
    """A model error occurred after native KV state may have changed."""


@dataclass(frozen=True, slots=True)
class StreamingSpanUpdate:
    """One complete class-vector replacement produced by the current append."""

    start_word: int
    end_word: int
    logits: tuple[float, ...]
    probs: tuple[float, ...]
    update_kind: UpdateKind

    def __post_init__(self) -> None:
        if not 0 <= self.start_word <= self.end_word:
            raise ValueError("streaming update boundary must be inclusive and ordered")
        if not self.logits or len(self.logits) != len(self.probs):
            raise ValueError("streaming update must contain equal nonempty logit/prob vectors")
        if not all(math.isfinite(value) for value in self.logits):
            raise ValueError("streaming update logits must be finite")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.probs):
            raise ValueError("streaming update probabilities must be finite probabilities")
        if self.update_kind not in {"new", "rescore"}:
            raise ValueError("streaming update kind must be 'new' or 'rescore'")

    @property
    def boundary(self) -> Boundary:
        """Return the inclusive absolute word boundary."""

        return self.start_word, self.end_word

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe update row."""

        return {
            "start_word": self.start_word,
            "end_word": self.end_word,
            "logits": list(self.logits),
            "probs": list(self.probs),
            "update_kind": self.update_kind,
        }


@dataclass(frozen=True, slots=True)
class StreamingStateMetadata:
    """Small host-only description of the session after an append."""

    session_id: str
    accumulated_text: str
    word_tokens: tuple[str, ...]
    word_char_starts: tuple[int, ...]
    word_char_ends: tuple[int, ...]
    labels: tuple[str, ...]
    token_count: int
    cache_offset: int
    word_count: int
    historical_span_count: int
    context_limit: int
    is_initialized: bool
    is_released: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe state metadata."""

        return {
            "session_id": self.session_id,
            "accumulated_text": self.accumulated_text,
            "word_tokens": list(self.word_tokens),
            "word_char_starts": list(self.word_char_starts),
            "word_char_ends": list(self.word_char_ends),
            "labels": list(self.labels),
            "token_count": self.token_count,
            "cache_offset": self.cache_offset,
            "word_count": self.word_count,
            "historical_span_count": self.historical_span_count,
            "context_limit": self.context_limit,
            "is_initialized": self.is_initialized,
            "is_released": self.is_released,
        }


@dataclass(frozen=True, slots=True)
class StreamingAppendResult:
    """Public snapshot, raw update vectors, timing, and resulting state metadata."""

    public_entities: tuple[PublicEntity, ...]
    span_updates: tuple[StreamingSpanUpdate, ...]
    elapsed_ms: float
    state: StreamingStateMetadata
    is_noop: bool = False

    @property
    def public_snapshot(self) -> tuple[PublicEntity, ...]:
        """Compatibility alias emphasizing that entities are a complete snapshot."""

        return self.public_entities

    @property
    def raw_updated_span_scores(self) -> tuple[StreamingSpanUpdate, ...]:
        """Compatibility alias for the current append's actual score replacements."""

        return self.span_updates

    @property
    def state_metadata(self) -> StreamingStateMetadata:
        """Compatibility alias for consumers building parity traces."""

        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe append result."""

        return {
            "public_entities": [entity.to_dict() for entity in self.public_entities],
            "span_updates": [update.to_dict() for update in self.span_updates],
            "elapsed_ms": self.elapsed_ms,
            "state": self.state.to_dict(),
            "is_noop": self.is_noop,
        }


def _probability_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("threshold must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_model(model: object) -> Any:
    required = (
        "preprocessor",
        "qwen",
        "prompt_projection",
        "marker",
        "decoder",
        "forward_preprocessed",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"model is missing streaming components: {missing}")
    return model


def _runtime() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    # Keep importing this backend cheap and headless-safe.  MLX is resolved only when
    # the first nonblank model call actually runs.
    import mlx.core as mx

    from streamner_commit.mlx.span_rep import score_spans
    from streamner_commit.mlx.word_pooling import pool_first_subtokens

    return mx, pool_first_subtokens, score_spans


def _release_native_cache(cache: Sequence[Any]) -> None:
    for layer in cache:
        if hasattr(layer, "keys"):
            layer.keys = None
        if hasattr(layer, "values"):
            layer.values = None
        if hasattr(layer, "offset"):
            layer.offset = 0
    if isinstance(cache, list):
        cache.clear()


def _candidate_updates(
    boundaries: object,
    logits: object,
    mask: object,
    *,
    prior_boundaries: set[Boundary],
) -> tuple[StreamingSpanUpdate, ...]:
    boundary_values = np.asarray(boundaries).reshape(-1, 2)
    logit_values = np.asarray(logits)
    if logit_values.ndim < 2:
        raise MLXStreamingBackendError("candidate logits must include rows and classes")
    logit_values = logit_values.reshape(-1, logit_values.shape[-1])
    mask_values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if not (boundary_values.shape[0] == logit_values.shape[0] == mask_values.shape[0]):
        raise MLXStreamingBackendError("candidate arrays have inconsistent row counts")
    valid_boundaries = boundary_values[mask_values]
    valid_logits = logit_values[mask_values]
    probabilities = sigmoid(valid_logits)
    updates: list[StreamingSpanUpdate] = []
    seen: set[Boundary] = set()
    for row, score_row, probability_row in zip(
        valid_boundaries,
        valid_logits,
        probabilities,
        strict=True,
    ):
        boundary = int(row[0]), int(row[1])
        if boundary in seen:
            raise MLXStreamingBackendError(f"duplicate scored boundary {boundary}")
        seen.add(boundary)
        updates.append(
            StreamingSpanUpdate(
                start_word=boundary[0],
                end_word=boundary[1],
                logits=tuple(float(value) for value in score_row),
                probs=tuple(float(value) for value in probability_row),
                update_kind="rescore" if boundary in prior_boundaries else "new",
            )
        )
    return tuple(updates)


class MLXStreamingSession:
    """One append-only stream with prompt, word, KV, and score ownership."""

    def __init__(
        self,
        model: object,
        *,
        labels: Sequence[str],
        session_id: str,
        context_limit: int,
        right_context_width: int,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
        clock: Callable[[], float],
        on_clear: Callable[[str], None] | None = None,
    ) -> None:
        self._model = _require_model(model)
        self.labels = normalize_labels(labels)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonblank string")
        self.session_id = session_id
        self.context_limit = _positive_int(context_limit, name="context_limit")
        self.right_context_width = _nonnegative_int(
            right_context_width,
            name="right_context_width",
        )
        self.threshold = _probability_threshold(threshold)
        if not isinstance(flat_ner, bool) or not isinstance(multi_label, bool):
            raise TypeError("flat_ner and multi_label must be booleans")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.flat_ner = flat_ner
        self.multi_label = multi_label
        self._clock = clock
        self._on_clear = on_clear
        self._state: MLXStreamingState | None = None
        self._cleared = False
        self._invalidated = False

    @property
    def is_initialized(self) -> bool:
        return self._state is not None and not self._state.is_released

    @property
    def is_cleared(self) -> bool:
        return self._cleared

    @property
    def historical_logits(self) -> Mapping[Boundary, np.ndarray[Any, Any]]:
        """Return a detached read-only snapshot of the complete score history."""

        self._require_open()
        if self._state is None:
            return MappingProxyType({})
        return self._state.historical_logits

    @property
    def state_metadata(self) -> StreamingStateMetadata:
        """Describe current state without exposing device tensors or native caches."""

        if self._state is None:
            return StreamingStateMetadata(
                session_id=self.session_id,
                accumulated_text="",
                word_tokens=(),
                word_char_starts=(),
                word_char_ends=(),
                labels=self.labels if not self._cleared else (),
                token_count=0,
                cache_offset=0,
                word_count=0,
                historical_span_count=0,
                context_limit=self.context_limit,
                is_initialized=False,
                is_released=self._cleared,
            )
        state = self._state
        if state.is_released:
            return StreamingStateMetadata(
                session_id=self.session_id,
                accumulated_text="",
                word_tokens=(),
                word_char_starts=(),
                word_char_ends=(),
                labels=(),
                token_count=0,
                cache_offset=0,
                word_count=0,
                historical_span_count=0,
                context_limit=self.context_limit,
                is_initialized=False,
                is_released=True,
            )
        history_count = len(state.historical_logits)
        return StreamingStateMetadata(
            session_id=self.session_id,
            accumulated_text=state.text,
            word_tokens=state.word_tokens,
            word_char_starts=state.word_char_starts,
            word_char_ends=state.word_char_ends,
            labels=state.labels,
            token_count=state.token_count,
            cache_offset=state.token_offset,
            word_count=state.word_count,
            historical_span_count=history_count,
            context_limit=state.context_limit,
            is_initialized=True,
            is_released=False,
        )

    def _require_open(self) -> None:
        if self._invalidated:
            raise InvalidatedSessionError(
                f"streaming session {self.session_id!r} was invalidated after model mutation"
            )
        if self._cleared:
            raise ClearedSessionError(f"streaming session {self.session_id!r} is cleared")

    def _notify_clear(self) -> None:
        callback, self._on_clear = self._on_clear, None
        if callback is not None:
            callback(self.session_id)

    def _invalidate(self, cache: Sequence[Any] | None = None) -> None:
        if self._state is not None:
            self._state.clear()
        elif cache is not None:
            _release_native_cache(cache)
        self._cleared = True
        self._invalidated = True
        self._notify_clear()

    def _decode_history(self) -> tuple[PublicEntity, ...]:
        state = self._state
        if state is None:
            return ()
        boundaries, logits = state.ordered_history_arrays()
        if not len(boundaries):
            return ()
        valid = np.ones((1, len(boundaries)), dtype=np.bool_)
        decoded = self._model.decoder.decode(
            logits[None, :, :],
            boundaries[None, :, :],
            valid,
            state.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            multi_label=self.multi_label,
        )[0]
        return self._model.decoder.to_public_entities(
            decoded,
            state.text,
            state.word_char_starts,
            state.word_char_ends,
        )

    def _result(
        self,
        *,
        started: float,
        updates: tuple[StreamingSpanUpdate, ...],
        entities: tuple[PublicEntity, ...],
        is_noop: bool,
    ) -> StreamingAppendResult:
        elapsed_ms = (self._clock() - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise MLXStreamingBackendError("clock produced an invalid append duration")
        return StreamingAppendResult(
            public_entities=entities,
            span_updates=updates,
            elapsed_ms=elapsed_ms,
            state=self.state_metadata,
            is_noop=is_noop,
        )

    def _cold_append(self, chunk: str, *, started: float) -> StreamingAppendResult:
        initialized = self._model.preprocessor.initialize_session(
            chunk,
            self.labels,
            context_limit=self.context_limit,
        )
        cache: list[Any] | None = None
        mutation_started = False
        try:
            cache = self._model.qwen.create_cache()
            mutation_started = True
            qwen_hidden = self._model.qwen.cached_hidden_states(
                initialized.cold.input_ids,
                cache,
            )
            output = self._model.forward_preprocessed(
                initialized.cold,
                qwen_hidden_states=qwen_hidden,
            )
            state = MLXStreamingState(
                session_id=self.session_id,
                qwen=self._model.qwen,
                qwen_cache=cache,
                token_count=initialized.state.decoder_token_count,
                context_limit=self.context_limit,
                past_word_embeddings=output.pooled_word_states,
                past_word_mask=output.pooled_word_mask,
                word_count=len(initialized.state.word_tokens),
                prompt_representations=output.labels.label_representations,
                prompt_mask=output.labels.label_mask,
                labels=self.labels,
                text=initialized.state.accumulated_text,
                word_tokens=initialized.state.word_tokens,
                word_char_starts=initialized.state.word_char_starts,
                word_char_ends=initialized.state.word_char_ends,
            )
            updates = _candidate_updates(
                initialized.cold.span_idx,
                output.logits,
                initialized.cold.span_mask,
                prior_boundaries=set(),
            )
            state.merge_candidate_logits(
                initialized.cold.span_idx,
                output.logits,
                initialized.cold.span_mask,
                replace_all=True,
            )
            state.validate()
            self._state = state
            entities = self._decode_history()
            return self._result(
                started=started,
                updates=updates,
                entities=entities,
                is_noop=False,
            )
        except Exception:
            if mutation_started:
                self._invalidate(cache)
            raise

    def _warm_append(self, chunk: str, *, started: float) -> StreamingAppendResult:
        state = self._state
        assert state is not None
        semantic_state = SessionPreprocessingState(
            labels=state.labels,
            accumulated_text=state.text,
            word_tokens=state.word_tokens,
            word_char_starts=state.word_char_starts,
            word_char_ends=state.word_char_ends,
            decoder_token_count=state.token_count,
        )
        prepared = self._model.preprocessor.preprocess_warm(
            semantic_state,
            chunk,
            self.labels,
            context_limit=self.context_limit,
            right_context_width=self.right_context_width,
        )
        if prepared.is_noop:
            return self._result(
                started=started,
                updates=(),
                entities=(),
                is_noop=True,
            )

        # Both preprocessing and state perform the context check before the call that
        # can mutate native KV.  A failure here leaves the session fully reusable.
        state.ensure_context_capacity(prepared.new_token_count)
        mutation_started = False
        try:
            mx, pool_first_subtokens, score_spans = _runtime()
            mutation_started = True
            new_hidden = self._model.qwen.cached_hidden_states(
                prepared.input_ids,
                state.qwen_cache,
            )
            state.commit_token_append(prepared.new_token_count)
            new_words, new_mask = pool_first_subtokens(
                new_hidden,
                mx.array(prepared.words_mask, dtype=mx.int32),
                mx.array(prepared.attention_mask, dtype=mx.int32),
                mx.array(prepared.text_lengths, dtype=mx.int32),
            )
            if state.past_word_embeddings is None or state.past_word_mask is None:
                raise MLXStreamingBackendError("live session lost its cached word states")
            compact_past_words = state.past_word_embeddings[:, : state.word_count]
            compact_past_mask = state.past_word_mask[:, : state.word_count]
            combined_words = mx.concatenate(
                [compact_past_words, new_words],
                axis=1,
            )
            combined_mask = mx.concatenate(
                [compact_past_mask, new_mask],
                axis=1,
            )
            span_idx = mx.array(prepared.span_idx, dtype=mx.int32)
            span_mask = mx.array(prepared.span_mask, dtype=mx.bool_)
            safe_span_idx = span_idx * span_mask[..., None]
            span_states = self._model.marker(combined_words, safe_span_idx, combined_mask)
            if state.prompt_representations is None:
                raise MLXStreamingBackendError("live session lost cached prompt representations")
            projected_labels = self._model.prompt_projection(state.prompt_representations)
            logits = score_spans(span_states, projected_labels)
            mx.eval(
                new_hidden,
                new_words,
                new_mask,
                combined_words,
                combined_mask,
                span_states,
                projected_labels,
                logits,
            )

            prior_boundaries = set(state.historical_logits)
            updates = _candidate_updates(
                prepared.span_idx,
                logits,
                prepared.span_mask,
                prior_boundaries=prior_boundaries,
            )
            next_semantic = prepared.next_state
            state.replace_word_history(
                past_word_embeddings=combined_words,
                past_word_mask=combined_mask,
                word_count=len(next_semantic.word_tokens),
                text=next_semantic.accumulated_text,
                word_tokens=next_semantic.word_tokens,
                word_char_starts=next_semantic.word_char_starts,
                word_char_ends=next_semantic.word_char_ends,
            )
            state.merge_candidate_logits(prepared.span_idx, logits, prepared.span_mask)
            state.validate()
            entities = self._decode_history()
            return self._result(
                started=started,
                updates=updates,
                entities=entities,
                is_noop=False,
            )
        except Exception:
            if mutation_started:
                self._invalidate()
            raise

    def append(self, chunk: str) -> StreamingAppendResult:
        """Append exact text and return the current reference-compatible result."""

        self._require_open()
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        started = self._clock()
        if not chunk.strip():
            # GLiNER 0.2.28 returns an empty public row and does not create or touch a
            # session for a blank call, even when a prior session exists.
            return self._result(
                started=started,
                updates=(),
                entities=(),
                is_noop=True,
            )
        if self._state is None:
            return self._cold_append(chunk, started=started)
        return self._warm_append(chunk, started=started)

    def clear(self) -> None:
        """Release all native cache/model state. Calling twice is harmless."""

        if self._cleared:
            return
        if self._state is not None:
            self._state.clear()
        self._cleared = True
        self._notify_clear()

    def __enter__(self) -> MLXStreamingSession:
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.clear()


class MLXStreamingBackend:
    """Factory and owner for lightweight live sessions over one loaded MLX model."""

    def __init__(
        self,
        model: object,
        *,
        context_limit: int | None = None,
        right_context_width: int = 12,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = _require_model(model)
        backbone_limit = _positive_int(
            getattr(self.model.qwen.configuration, "max_position_embeddings", None),
            name="Qwen context limit",
        )
        self.context_limit = (
            backbone_limit
            if context_limit is None
            else _positive_int(context_limit, name="context_limit")
        )
        if self.context_limit > backbone_limit:
            raise ValueError(
                f"context_limit {self.context_limit} exceeds Qwen limit {backbone_limit}"
            )
        self.right_context_width = _nonnegative_int(
            right_context_width,
            name="right_context_width",
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._next_session_number = 1
        self._sessions: dict[str, MLXStreamingSession] = {}

    @classmethod
    def from_asset_bundle(
        cls,
        bundle: AssetBundle,
        *,
        context_limit: int | None = None,
        right_context_width: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> Self:
        """Load the exact cold model and configure its pinned streaming window."""

        from streamner_commit.mlx.model import MLXColdModel

        configured_width = bundle.config.get("right_context_width")
        if right_context_width is None:
            right_context_width = _nonnegative_int(
                configured_width,
                name="config.right_context_width",
            )
        return cls(
            MLXColdModel.from_asset_bundle(bundle),
            context_limit=context_limit,
            right_context_width=right_context_width,
            clock=clock,
        )

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def start_session(
        self,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        session_id: str | None = None,
    ) -> MLXStreamingSession:
        """Create an empty session; the first nonblank append performs cold init."""

        if session_id is None:
            session_id = f"mlx-session-{self._next_session_number}"
            self._next_session_number += 1
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonblank string")
        if session_id in self._sessions:
            raise ValueError(f"session {session_id!r} already exists")
        session = MLXStreamingSession(
            self.model,
            labels=labels,
            session_id=session_id,
            context_limit=self.context_limit,
            right_context_width=self.right_context_width,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            clock=self._clock,
            on_clear=self._drop_session,
        )
        self._sessions[session_id] = session
        return session

    def clear_sessions(self) -> None:
        """Release every session created by this backend."""

        for session in tuple(self._sessions.values()):
            session.clear()
        self._sessions.clear()

    def infer_full(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
    ) -> list[dict[str, Any]]:
        """Delegate an independent cold call to the shared loaded model."""

        return self.model.infer_full(
            text,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
        )

    def infer_full_trace(
        self,
        text: str,
        labels: Sequence[str],
        *,
        example_id: str,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> ColdFullResult:
        """Run one stateless cold graph and retain every valid raw candidate.

        This is deliberately separate from :meth:`start_session`: it neither creates
        nor advances streaming state, and it never obtains a full result by appending a
        sentinel to a warm session.  The returned raw map is the complete valid cold
        candidate set in inclusive model-word coordinates.
        """

        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("example_id must be a nonblank string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ordered_labels = normalize_labels(labels)
        normalized_threshold = _probability_threshold(threshold)
        if not isinstance(flat_ner, bool) or not isinstance(multi_label, bool):
            raise TypeError("flat_ner and multi_label must be booleans")

        predict = getattr(self.model, "predict", None)
        if not callable(predict):
            raise TypeError("model must expose predict() for raw cold trace capture")
        prediction = predict(
            text,
            ordered_labels,
            threshold=normalized_threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        forward = getattr(prediction, "forward", None)
        prepared = getattr(forward, "preprocessing", None)
        if prepared is None:
            raise MLXStreamingBackendError("cold prediction is missing preprocessing metadata")
        if getattr(prepared, "text", None) != text:
            raise MLXStreamingBackendError("cold prediction text differs from the requested text")
        if tuple(getattr(prepared, "labels", ())) != ordered_labels:
            raise MLXStreamingBackendError("cold prediction labels differ from requested order")

        boundaries = np.asarray(getattr(prepared, "span_idx", None))
        raw_mask = np.asarray(getattr(prepared, "span_mask", None))
        logits = np.asarray(getattr(forward, "logits", None))
        if boundaries.ndim < 3 or boundaries.shape[0] != 1 or boundaries.shape[-1] != 2:
            raise MLXStreamingBackendError("cold span_idx must have shape (1, ..., 2)")
        if boundaries.dtype.kind not in {"i", "u"}:
            raise MLXStreamingBackendError("cold span_idx must contain integers")
        if raw_mask.ndim < 2 or raw_mask.shape[0] != 1:
            raise MLXStreamingBackendError("cold span_mask must have shape (1, ...)")
        if raw_mask.dtype.kind not in {"b", "i", "u"} or (
            raw_mask.size and not np.isin(raw_mask, (0, 1)).all()
        ):
            raise MLXStreamingBackendError("cold span_mask must contain only zero/one values")
        if logits.ndim not in {3, 4} or logits.shape[0] != 1:
            raise MLXStreamingBackendError(
                "cold logits must have shape (1, spans, classes) or "
                "(1, starts, widths, classes)"
            )
        boundary_rows = boundaries.reshape(-1, 2)
        mask_rows = raw_mask.reshape(-1).astype(np.bool_, copy=False)
        logit_rows = logits.reshape(-1, logits.shape[-1])
        if not (len(boundary_rows) == len(mask_rows) == len(logit_rows)):
            raise MLXStreamingBackendError("cold candidate arrays have inconsistent row counts")
        if logit_rows.shape[-1] != len(ordered_labels):
            raise MLXStreamingBackendError(
                "cold class-vector width differs from the ordered label count"
            )
        if not np.isfinite(logit_rows).all():
            raise MLXStreamingBackendError("cold logits contain non-finite values")

        word_tokens = tuple(getattr(prepared, "word_tokens", ()))
        if not word_tokens:
            raise MLXStreamingBackendError("cold preprocessing has no model words")
        raw_state: dict[SpanBoundary, tuple[float, ...]] = {}
        for raw_boundary, raw_logits in zip(
            boundary_rows[mask_rows],
            logit_rows[mask_rows],
            strict=True,
        ):
            boundary = SpanBoundary(int(raw_boundary[0]), int(raw_boundary[1]))
            if boundary.end_word >= len(word_tokens):
                raise MLXStreamingBackendError(
                    f"cold boundary {boundary.to_tuple()} exceeds model-word coordinates"
                )
            if boundary in raw_state:
                raise MLXStreamingBackendError(
                    f"cold graph generated duplicate boundary {boundary.to_tuple()}"
                )
            raw_state[boundary] = tuple(float(value) for value in raw_logits)

        entities = tuple(getattr(prediction, "entities", ()))
        if not all(isinstance(entity, PublicEntity) for entity in entities):
            raise MLXStreamingBackendError("cold prediction entities must be PublicEntity values")
        return ColdFullResult(
            example_id=example_id,
            full_text=text,
            public_entities=entities,
            raw_final_span_state=raw_state,
        )


__all__ = [
    "ClearedSessionError",
    "InvalidatedSessionError",
    "MLXStreamingBackend",
    "MLXStreamingBackendError",
    "MLXStreamingSession",
    "StreamingAppendResult",
    "StreamingSpanUpdate",
    "StreamingStateMetadata",
]
