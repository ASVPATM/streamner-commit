"""Exact, Torch-free preprocessing for the pinned streaming-span checkpoint.

The reference processor presents the tokenizer with a list of already split
"words".  Prompt words are whole label strings followed by ``<<LABEL>>`` and a
final ``<<SEP>>``; text words come from GLiNER's pinned whitespace-splitter
regular expression.  This module reproduces that discrete contract with NumPy
arrays so downstream MLX code never needs to import PyTorch or GLiNER.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from streamner_commit.chunking import word_char_spans
from streamner_commit.mlx.assets import AssetBundle

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class PreprocessingError(ValueError):
    """The tokenizer or exported configuration violated the pinned contract."""


class ContextLimitExceededError(PreprocessingError):
    """A cold or warm decoder input would exceed its explicit context limit."""


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingError(f"{name} must be a nonblank string")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreprocessingError(f"{name} must be an integer")
    if value < 1:
        raise PreprocessingError(f"{name} must be positive")
    return value


def normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """Remove exact duplicates while preserving first occurrence order.

    Streaming GLiNER 0.2.28 normalizes labels with ``dict.fromkeys`` before it
    creates the one-based class mapping.  Whitespace inside a nonblank label is
    intentionally preserved because it affects Qwen tokenization.
    """

    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be an ordered sequence of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise TypeError(f"labels[{index}] must be a string")
        if not label.strip():
            raise ValueError(f"labels[{index}] must be nonblank")
        if label not in seen:
            seen.add(label)
            normalized.append(label)
    if not normalized:
        raise ValueError("at least one label is required")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """The discrete subset of the validated export configuration."""

    max_width: int
    label_token: str
    label_token_id: int
    separator_token: str
    separator_token_id: int
    words_splitter_type: str = "whitespace"
    subtoken_pooling: str = "first"

    def __post_init__(self) -> None:
        _require_positive_int(self.max_width, name="max_width")
        _require_nonempty_string(self.label_token, name="label_token")
        _require_nonempty_string(self.separator_token, name="separator_token")
        for value, name in (
            (self.label_token_id, "label_token_id"),
            (self.separator_token_id, "separator_token_id"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PreprocessingError(f"{name} must be a nonnegative integer")
        if self.label_token_id == self.separator_token_id:
            raise PreprocessingError("label and separator token IDs must differ")
        if self.words_splitter_type != "whitespace":
            raise PreprocessingError(
                "the pinned preprocessing port supports words_splitter_type='whitespace' only"
            )
        if self.subtoken_pooling != "first":
            raise PreprocessingError(
                "the pinned preprocessing port supports subtoken_pooling='first' only"
            )

    @classmethod
    def from_asset_bundle(cls, bundle: AssetBundle) -> PreprocessingConfig:
        """Read and cross-check preprocessing fields from a validated bundle."""

        if not isinstance(bundle, AssetBundle):
            raise TypeError("bundle must be a validated AssetBundle")
        model = bundle.config
        special = bundle.tokenizer_special_tokens

        config = cls(
            max_width=_require_positive_int(model.get("max_width"), name="config.max_width"),
            label_token=_require_nonempty_string(
                model.get("label_token"), name="config.label_token"
            ),
            label_token_id=_require_token_id(
                model.get("class_token_index"), name="config.class_token_index"
            ),
            separator_token=_require_nonempty_string(
                model.get("sep_token"), name="config.sep_token"
            ),
            separator_token_id=_require_token_id(
                model.get("sep_token_index"), name="config.sep_token_index"
            ),
            words_splitter_type=_require_nonempty_string(
                model.get("words_splitter_type"), name="config.words_splitter_type"
            ),
            subtoken_pooling=_require_nonempty_string(
                model.get("subtoken_pooling"), name="config.subtoken_pooling"
            ),
        )

        expected_special = {
            "label_marker_token": config.label_token,
            "label_marker_token_id": config.label_token_id,
            "separator_marker_token": config.separator_token,
            "separator_marker_token_id": config.separator_token_id,
        }
        for field, expected in expected_special.items():
            if special.get(field) != expected:
                raise PreprocessingError(
                    f"tokenizer metadata {field} disagrees with model config: "
                    f"expected {expected!r}, got {special.get(field)!r}"
                )
        if special.get("padding_side") not in {None, "right"}:
            raise PreprocessingError("the pinned tokenizer must use right padding")
        return config


def _require_token_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreprocessingError(f"{name} must be a nonnegative integer")
    return value


def _validate_context_limit(context_limit: int | None) -> int | None:
    if context_limit is None:
        return None
    if isinstance(context_limit, bool) or not isinstance(context_limit, int):
        raise TypeError("context_limit must be an integer or None")
    if context_limit < 1:
        raise ValueError("context_limit must be positive when provided")
    return context_limit


def _require_context_capacity(
    token_count: int,
    context_limit: int | None,
    *,
    operation: str,
) -> None:
    limit = _validate_context_limit(context_limit)
    if limit is not None and token_count > limit:
        raise ContextLimitExceededError(
            f"{operation} would contain {token_count} decoder tokens, exceeding "
            f"the explicit context limit of {limit}; clear the session before continuing"
        )


def _readonly_int_array(value: object, *, name: str, ndim: int) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise PreprocessingError(f"{name} must have {ndim} dimensions, got {raw.ndim}")
    if raw.dtype.kind not in {"i", "u"}:
        raise PreprocessingError(f"{name} must contain integers, got {raw.dtype}")
    array = np.ascontiguousarray(raw, dtype=np.int64).copy()
    array.setflags(write=False)
    return array


def _readonly_bool_array(value: object, *, name: str, ndim: int) -> BoolArray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise PreprocessingError(f"{name} must have {ndim} dimensions, got {raw.ndim}")
    if raw.dtype.kind not in {"b", "i", "u"}:
        raise PreprocessingError(f"{name} must contain booleans, got {raw.dtype}")
    if raw.size and not np.isin(raw, (0, 1)).all():
        raise PreprocessingError(f"{name} must contain only zero/one values")
    array = np.ascontiguousarray(raw, dtype=np.bool_).copy()
    array.setflags(write=False)
    return array


def _span_rows(first_start: int, num_starts: int, max_width: int) -> IntArray:
    starts = np.repeat(
        np.arange(first_start, first_start + num_starts, dtype=np.int64),
        max_width,
    )
    offsets = np.tile(np.arange(max_width, dtype=np.int64), num_starts)
    rows = np.stack((starts, starts + offsets), axis=1)
    rows.setflags(write=False)
    return rows


def prepare_streaming_span_candidates(
    past_words: int,
    new_words: int,
    max_width: int,
    *,
    recompute_all: bool = False,
    right_context_width: int = 0,
) -> tuple[IntArray, BoolArray]:
    """Reproduce GLiNER's absolute inclusive streaming candidate enumeration."""

    for value, name in ((past_words, "past_words"), (new_words, "new_words")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    _require_positive_int(max_width, name="max_width")
    if not isinstance(recompute_all, bool):
        raise TypeError("recompute_all must be a boolean")
    if isinstance(right_context_width, bool) or not isinstance(right_context_width, int):
        raise TypeError("right_context_width must be an integer")
    if right_context_width < 0:
        raise ValueError("right_context_width must be nonnegative")

    total_words = past_words + new_words
    if total_words == 0:
        empty_rows = np.empty((0, 2), dtype=np.int64)
        empty_mask = np.empty((0,), dtype=np.bool_)
        empty_rows.setflags(write=False)
        empty_mask.setflags(write=False)
        return empty_rows, empty_mask

    if recompute_all:
        minimum_end = 0
    else:
        latest_word = total_words - 1
        rolling_minimum_end = max(0, latest_word - right_context_width)
        minimum_end = min(past_words, rolling_minimum_end)

    first_start = max(0, minimum_end - (max_width - 1))
    rows = _span_rows(first_start, total_words - first_start, max_width)
    mask = rows[:, 1] < total_words
    if not recompute_all:
        mask = mask & (rows[:, 1] >= minimum_end)
        if new_words == 0:
            mask = np.zeros_like(mask)
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    mask.setflags(write=False)
    return rows, mask


def prepare_cold_span_candidates(num_words: int, max_width: int) -> tuple[IntArray, BoolArray]:
    """Enumerate one cold candidate matrix, including masked invalid end rows."""

    return prepare_streaming_span_candidates(
        0,
        num_words,
        max_width,
        recompute_all=True,
    )


@dataclass(frozen=True, slots=True)
class ColdPreprocessingResult:
    """One validated, immutable cold preprocessing result."""

    text: str
    labels: tuple[str, ...]
    word_tokens: tuple[str, ...]
    word_char_starts: tuple[int, ...]
    word_char_ends: tuple[int, ...]
    serialized_prompt: str
    serialized_prompt_words: tuple[str, ...]
    serialized_input_words: tuple[str, ...]
    prompt_word_length: int
    tokenizer_tokens: tuple[str, ...]
    max_width: int
    label_token_id: int
    separator_token_id: int
    input_ids: IntArray
    attention_mask: IntArray
    label_attention_mask: IntArray
    words_mask: IntArray
    text_lengths: IntArray
    span_idx: IntArray
    span_mask: BoolArray
    label_token_positions: IntArray
    separator_token_positions: IntArray

    def __post_init__(self) -> None:
        _require_nonempty_string(self.text, name="text")
        if normalize_labels(self.labels) != self.labels:
            raise PreprocessingError("labels must already be normalized and ordered")
        _require_positive_int(self.max_width, name="max_width")
        _require_token_id(self.label_token_id, name="label_token_id")
        _require_token_id(self.separator_token_id, name="separator_token_id")

        for field in (
            "word_tokens",
            "word_char_starts",
            "word_char_ends",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        object.__setattr__(self, "serialized_prompt_words", tuple(self.serialized_prompt_words))
        object.__setattr__(self, "serialized_input_words", tuple(self.serialized_input_words))
        object.__setattr__(self, "tokenizer_tokens", tuple(self.tokenizer_tokens))

        word_count = len(self.word_tokens)
        if not (
            word_count == len(self.word_char_starts) == len(self.word_char_ends) and word_count > 0
        ):
            raise PreprocessingError(
                "word tokens and character offsets must be nonempty and aligned"
            )
        for token, start, end in zip(
            self.word_tokens,
            self.word_char_starts,
            self.word_char_ends,
            strict=True,
        ):
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(self.text)
                or self.text[start:end] != token
            ):
                raise PreprocessingError("word character offsets do not round-trip to text")

        if self.prompt_word_length != len(self.serialized_prompt_words):
            raise PreprocessingError("prompt_word_length differs from serialized prompt words")
        if self.serialized_prompt != "".join(self.serialized_prompt_words):
            raise PreprocessingError("serialized_prompt differs from its prompt-word packing")
        if self.serialized_input_words != self.serialized_prompt_words + self.word_tokens:
            raise PreprocessingError("serialized input must be prompt words followed by text words")

        array_fields = {
            "input_ids": (self.input_ids, 2),
            "attention_mask": (self.attention_mask, 2),
            "label_attention_mask": (self.label_attention_mask, 2),
            "words_mask": (self.words_mask, 2),
            "text_lengths": (self.text_lengths, 2),
            "span_idx": (self.span_idx, 3),
            "label_token_positions": (self.label_token_positions, 2),
            "separator_token_positions": (self.separator_token_positions, 2),
        }
        for field, (value, ndim) in array_fields.items():
            object.__setattr__(
                self,
                field,
                _readonly_int_array(value, name=field, ndim=ndim),
            )
        object.__setattr__(
            self,
            "span_mask",
            _readonly_bool_array(self.span_mask, name="span_mask", ndim=2),
        )

        token_shape = self.input_ids.shape
        if token_shape[0] != 1 or any(
            value.shape != token_shape
            for value in (self.attention_mask, self.label_attention_mask, self.words_mask)
        ):
            raise PreprocessingError("token IDs and token-level masks must share shape (1, tokens)")
        if not np.all(self.attention_mask == 1):
            raise PreprocessingError("single-example cold preprocessing must not contain padding")
        if not np.array_equal(self.label_attention_mask, self.attention_mask):
            raise PreprocessingError("cold label attention must equal full attention")
        if len(self.tokenizer_tokens) != token_shape[1]:
            raise PreprocessingError("tokenizer token strings must align with input IDs")
        if self.text_lengths.shape != (1, 1) or int(self.text_lengths[0, 0]) != word_count:
            raise PreprocessingError("text_lengths must contain the model-word count")

        positive_word_ids = self.words_mask[self.words_mask > 0].tolist()
        if positive_word_ids != list(range(1, word_count + 1)):
            raise PreprocessingError(
                "words_mask must select each text word once in one-based order"
            )

        expected_rows, expected_mask = prepare_cold_span_candidates(word_count, self.max_width)
        if self.span_idx.shape != (1, word_count * self.max_width, 2) or not np.array_equal(
            self.span_idx[0], expected_rows
        ):
            raise PreprocessingError("span_idx differs from pinned cold candidate order")
        if self.span_mask.shape != (1, word_count * self.max_width) or not np.array_equal(
            self.span_mask[0], expected_mask
        ):
            raise PreprocessingError("span_mask differs from pinned cold validity mask")

        expected_label_positions = np.argwhere(self.input_ids == self.label_token_id)
        expected_separator_positions = np.argwhere(self.input_ids == self.separator_token_id)
        if not np.array_equal(self.label_token_positions, expected_label_positions):
            raise PreprocessingError("label token positions disagree with input IDs")
        if not np.array_equal(self.separator_token_positions, expected_separator_positions):
            raise PreprocessingError("separator token positions disagree with input IDs")
        if self.label_token_positions.shape != (len(self.labels), 2):
            raise PreprocessingError("prompt must contain exactly one label marker per label")
        if self.separator_token_positions.shape != (1, 2):
            raise PreprocessingError("cold input must contain exactly one separator marker")

    def as_model_inputs(self) -> Mapping[str, IntArray | BoolArray]:
        """Expose the complete discrete batch under reference model input names."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "label_attention_mask": self.label_attention_mask,
            "words_mask": self.words_mask,
            "text_lengths": self.text_lengths,
            "span_idx": self.span_idx,
            "span_mask": self.span_mask,
        }


@dataclass(frozen=True, slots=True)
class SessionPreprocessingState:
    """Semantic session metadata needed to preprocess a later warm append.

    Model caches and word embeddings intentionally live elsewhere.  This small
    record owns only the exact text/coordinate history and decoder token count
    that preprocessing must extend.
    """

    labels: tuple[str, ...]
    accumulated_text: str
    word_tokens: tuple[str, ...]
    word_char_starts: tuple[int, ...]
    word_char_ends: tuple[int, ...]
    decoder_token_count: int

    def __post_init__(self) -> None:
        if normalize_labels(self.labels) != self.labels:
            raise PreprocessingError("session labels must be normalized and ordered")
        if not isinstance(self.accumulated_text, str) or not self.accumulated_text.strip():
            raise PreprocessingError("an initialized session must contain nonblank text")
        for field in ("word_tokens", "word_char_starts", "word_char_ends"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not (
            len(self.word_tokens) == len(self.word_char_starts) == len(self.word_char_ends) > 0
        ):
            raise PreprocessingError("session word tokens and character offsets must align")
        previous_end = 0
        for token, start, end in zip(
            self.word_tokens,
            self.word_char_starts,
            self.word_char_ends,
            strict=True,
        ):
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not previous_end <= start < end <= len(self.accumulated_text)
                or self.accumulated_text[start:end] != token
            ):
                raise PreprocessingError("session word offsets do not round-trip in order")
            previous_end = end
        if (
            isinstance(self.decoder_token_count, bool)
            or not isinstance(self.decoder_token_count, int)
            or self.decoder_token_count < 1
        ):
            raise PreprocessingError("decoder_token_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class SessionInitializationResult:
    """A cold prompt-bearing batch paired with its committed semantic state."""

    cold: ColdPreprocessingResult
    state: SessionPreprocessingState
    context_limit: int | None

    def __post_init__(self) -> None:
        _validate_context_limit(self.context_limit)
        if self.state.labels != self.cold.labels:
            raise PreprocessingError("cold and session label order differs")
        if self.state.accumulated_text != self.cold.text:
            raise PreprocessingError("cold and session text differs")
        if self.state.word_tokens != self.cold.word_tokens:
            raise PreprocessingError("cold and session word tokens differ")
        if self.state.word_char_starts != self.cold.word_char_starts:
            raise PreprocessingError("cold and session word starts differ")
        if self.state.word_char_ends != self.cold.word_char_ends:
            raise PreprocessingError("cold and session word ends differ")
        token_count = int(self.cold.attention_mask.sum())
        if self.state.decoder_token_count != token_count:
            raise PreprocessingError("cold token count differs from committed session state")
        _require_context_capacity(token_count, self.context_limit, operation="cold initialization")


@dataclass(frozen=True, slots=True)
class WarmChunkPreprocessingResult:
    """One no-prompt warm append, or an explicit blank no-op.

    ``attention_mask`` is the tokenizer mask for only the current decoder
    tokens.  ``full_attention_mask`` is the mask passed to Qwen together with
    the existing KV cache.  ``as_model_inputs`` deliberately maps the latter
    to the model's ``attention_mask`` argument.
    """

    chunk: str
    is_noop: bool
    previous_state: SessionPreprocessingState
    next_state: SessionPreprocessingState
    current_word_tokens: tuple[str, ...]
    current_word_char_starts: tuple[int, ...]
    current_word_char_ends: tuple[int, ...]
    absolute_word_char_starts: tuple[int, ...]
    absolute_word_char_ends: tuple[int, ...]
    tokenizer_tokens: tuple[str, ...]
    context_limit: int | None
    max_width: int
    right_context_width: int
    recompute_all_candidates: bool
    input_ids: IntArray
    attention_mask: IntArray
    label_attention_mask: IntArray
    full_attention_mask: IntArray
    position_ids: IntArray
    words_mask: IntArray
    text_lengths: IntArray
    past_word_lengths: IntArray
    span_idx: IntArray
    span_mask: BoolArray

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, str):
            raise TypeError("chunk must be a string")
        if not isinstance(self.is_noop, bool):
            raise TypeError("is_noop must be a boolean")
        if not isinstance(self.previous_state, SessionPreprocessingState) or not isinstance(
            self.next_state, SessionPreprocessingState
        ):
            raise TypeError("previous_state and next_state must be session preprocessing states")
        _validate_context_limit(self.context_limit)
        _require_positive_int(self.max_width, name="max_width")
        if (
            isinstance(self.right_context_width, bool)
            or not isinstance(self.right_context_width, int)
            or self.right_context_width < 0
        ):
            raise PreprocessingError("right_context_width must be a nonnegative integer")
        if not isinstance(self.recompute_all_candidates, bool):
            raise TypeError("recompute_all_candidates must be a boolean")

        tuple_fields = (
            "current_word_tokens",
            "current_word_char_starts",
            "current_word_char_ends",
            "absolute_word_char_starts",
            "absolute_word_char_ends",
            "tokenizer_tokens",
        )
        for field in tuple_fields:
            object.__setattr__(self, field, tuple(getattr(self, field)))

        integer_arrays = {
            "input_ids": (self.input_ids, 2),
            "attention_mask": (self.attention_mask, 2),
            "label_attention_mask": (self.label_attention_mask, 2),
            "full_attention_mask": (self.full_attention_mask, 2),
            "position_ids": (self.position_ids, 2),
            "words_mask": (self.words_mask, 2),
            "text_lengths": (self.text_lengths, 2),
            "past_word_lengths": (self.past_word_lengths, 1),
            "span_idx": (self.span_idx, 3),
        }
        for field, (value, ndim) in integer_arrays.items():
            object.__setattr__(
                self,
                field,
                _readonly_int_array(value, name=field, ndim=ndim),
            )
        object.__setattr__(
            self,
            "span_mask",
            _readonly_bool_array(self.span_mask, name="span_mask", ndim=2),
        )

        new_word_count = len(self.current_word_tokens)
        if not (
            new_word_count
            == len(self.current_word_char_starts)
            == len(self.current_word_char_ends)
            == len(self.absolute_word_char_starts)
            == len(self.absolute_word_char_ends)
        ):
            raise PreprocessingError("warm word tokens and offsets must align")
        char_offset = len(self.previous_state.accumulated_text)
        for token, local_start, local_end, absolute_start, absolute_end in zip(
            self.current_word_tokens,
            self.current_word_char_starts,
            self.current_word_char_ends,
            self.absolute_word_char_starts,
            self.absolute_word_char_ends,
            strict=True,
        ):
            if (
                self.chunk[local_start:local_end] != token
                or absolute_start != char_offset + local_start
                or absolute_end != char_offset + local_end
                or self.next_state.accumulated_text[absolute_start:absolute_end] != token
            ):
                raise PreprocessingError("warm local and absolute word offsets disagree")

        current_shape = self.input_ids.shape
        if current_shape[0] != 1 or any(
            value.shape != current_shape
            for value in (
                self.attention_mask,
                self.label_attention_mask,
                self.position_ids,
                self.words_mask,
            )
        ):
            raise PreprocessingError("warm token arrays must share shape (1, new_tokens)")
        if len(self.tokenizer_tokens) != current_shape[1]:
            raise PreprocessingError("warm tokenizer tokens must align with input IDs")
        if not np.all(self.attention_mask == 1) or not np.array_equal(
            self.label_attention_mask, self.attention_mask
        ):
            raise PreprocessingError("warm current attention must be unpadded and label-local")

        cached_tokens = self.previous_state.decoder_token_count
        total_tokens = cached_tokens + current_shape[1]
        if self.next_state.decoder_token_count != total_tokens:
            raise PreprocessingError("warm next-state decoder count is inconsistent")
        if self.full_attention_mask.shape != (1, total_tokens) or not np.all(
            self.full_attention_mask == 1
        ):
            raise PreprocessingError("warm full attention must cover cached and current tokens")
        expected_positions = np.arange(cached_tokens, total_tokens, dtype=np.int64)[None, :]
        if not np.array_equal(self.position_ids, expected_positions):
            raise PreprocessingError("warm position IDs must continue from the cache offset")
        _require_context_capacity(total_tokens, self.context_limit, operation="warm append")

        if self.text_lengths.shape != (1, 1) or int(self.text_lengths[0, 0]) != new_word_count:
            raise PreprocessingError("warm text_lengths must contain the new model-word count")
        if self.past_word_lengths.shape != (1,) or int(self.past_word_lengths[0]) != len(
            self.previous_state.word_tokens
        ):
            raise PreprocessingError("past_word_lengths must contain the historical word count")
        positive_word_ids = self.words_mask[self.words_mask > 0].tolist()
        if positive_word_ids != list(range(1, new_word_count + 1)):
            raise PreprocessingError("warm words_mask must be local and one-based")

        if self.is_noop:
            if self.chunk.strip():
                raise PreprocessingError("only blank chunks may be preprocessing no-ops")
            if self.next_state != self.previous_state:
                raise PreprocessingError("blank chunks must leave session state unchanged")
            if new_word_count or current_shape[1] or self.span_idx.shape != (1, 0, 2):
                raise PreprocessingError("blank chunks must not tokenize or enumerate spans")
            if self.span_mask.shape != (1, 0):
                raise PreprocessingError("blank chunks must have an empty span mask")
            return

        if not self.chunk.strip() or new_word_count == 0 or current_shape[1] == 0:
            raise PreprocessingError("a non-noop warm append must contain text tokens and words")
        if self.next_state.labels != self.previous_state.labels:
            raise PreprocessingError("warm appends must retain the session label order")
        if self.next_state.accumulated_text != self.previous_state.accumulated_text + self.chunk:
            raise PreprocessingError("warm append must preserve exact accumulated text")
        if (
            self.next_state.word_tokens
            != self.previous_state.word_tokens + self.current_word_tokens
        ):
            raise PreprocessingError("warm append must extend the exact model-word list")
        if (
            self.next_state.word_char_starts
            != self.previous_state.word_char_starts + self.absolute_word_char_starts
            or self.next_state.word_char_ends
            != self.previous_state.word_char_ends + self.absolute_word_char_ends
        ):
            raise PreprocessingError("warm append must extend absolute character offsets")

        expected_rows, expected_mask = prepare_streaming_span_candidates(
            len(self.previous_state.word_tokens),
            new_word_count,
            self.max_width,
            recompute_all=self.recompute_all_candidates,
            right_context_width=self.right_context_width,
        )
        if not np.array_equal(self.span_idx[0], expected_rows) or not np.array_equal(
            self.span_mask[0], expected_mask
        ):
            raise PreprocessingError("warm span candidates differ from pinned enumeration")

    @property
    def past_word_count(self) -> int:
        return len(self.previous_state.word_tokens)

    @property
    def new_word_count(self) -> int:
        return len(self.current_word_tokens)

    @property
    def cached_token_count(self) -> int:
        return self.previous_state.decoder_token_count

    @property
    def new_token_count(self) -> int:
        return self.input_ids.shape[1]

    @property
    def total_token_count(self) -> int:
        return self.next_state.decoder_token_count

    def as_model_inputs(self) -> Mapping[str, IntArray | BoolArray]:
        """Expose warm arrays under the reference forward argument names."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.full_attention_mask,
            "label_attention_mask": self.label_attention_mask,
            "position_ids": self.position_ids,
            "words_mask": self.words_mask,
            "text_lengths": self.text_lengths,
            "past_word_length": self.past_word_lengths,
            "span_idx": self.span_idx,
            "span_mask": self.span_mask,
        }


class ColdPreprocessor:
    """Tokenizer-bound implementation of the pinned cold preprocessing path."""

    def __init__(self, config: PreprocessingConfig, tokenizer: Any) -> None:
        if not isinstance(config, PreprocessingConfig):
            raise TypeError("config must be a PreprocessingConfig")
        if tokenizer is None or not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        self.config = config
        self.tokenizer = tokenizer
        self._validate_tokenizer()

    @classmethod
    def from_asset_bundle(
        cls,
        bundle: AssetBundle,
        *,
        tokenizer: Any | None = None,
    ) -> ColdPreprocessor:
        """Bind the locally loaded tokenizer from a validated safe asset bundle."""

        config = PreprocessingConfig.from_asset_bundle(bundle)
        return cls(config, bundle.load_tokenizer() if tokenizer is None else tokenizer)

    def _validate_tokenizer(self) -> None:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if not callable(convert):
            raise PreprocessingError("tokenizer must expose convert_tokens_to_ids")
        for token, expected_id, name in (
            (self.config.label_token, self.config.label_token_id, "label"),
            (self.config.separator_token, self.config.separator_token_id, "separator"),
        ):
            actual_id = convert(token)
            if actual_id != expected_id:
                raise PreprocessingError(
                    f"tokenizer {name} marker ID mismatch: expected {expected_id}, got {actual_id}"
                )
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise PreprocessingError("the pinned tokenizer must use right padding")

    def _tokenize_word_row(
        self,
        words: tuple[str, ...],
        *,
        prompt_word_length: int,
    ) -> tuple[IntArray, IntArray, IntArray, tuple[str, ...]]:
        if not words:
            raise PreprocessingError("cannot tokenize an empty word row")
        encoded = self.tokenizer(
            [list(words)],
            is_split_into_words=True,
            return_tensors="np",
            truncation=False,
            padding="longest",
            add_special_tokens=False,
        )
        if not isinstance(encoded, Mapping):
            raise PreprocessingError("tokenizer output must be a mapping")
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise PreprocessingError("tokenizer output is missing input IDs or attention mask")
        input_ids = _readonly_int_array(encoded["input_ids"], name="input_ids", ndim=2)
        attention = _readonly_int_array(encoded["attention_mask"], name="attention_mask", ndim=2)
        if input_ids.shape != attention.shape or input_ids.shape[0] != 1:
            raise PreprocessingError("tokenizer must return one aligned batch row")
        if attention.size == 0 or not np.isin(attention, (0, 1)).all():
            raise PreprocessingError("attention_mask must be a nonempty zero/one array")

        word_ids_method = getattr(encoded, "word_ids", None)
        if not callable(word_ids_method):
            raise PreprocessingError("fast tokenizer output must expose word_ids")
        words_mask = _prepare_first_word_mask(
            word_ids_method(batch_index=0),
            prompt_word_length=prompt_word_length,
            expected_input_words=len(words),
            expected_token_count=input_ids.shape[1],
        )

        convert_ids = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if not callable(convert_ids):
            raise PreprocessingError("tokenizer must expose convert_ids_to_tokens")
        decoded_tokens = convert_ids(input_ids[0].tolist())
        if isinstance(decoded_tokens, str) or not isinstance(decoded_tokens, Sequence):
            raise PreprocessingError("tokenizer did not return one token string per input ID")
        tokenizer_tokens = tuple(str(token) for token in decoded_tokens)
        return input_ids, attention, words_mask, tokenizer_tokens

    def preprocess(self, text: str, labels: Sequence[str]) -> ColdPreprocessingResult:
        """Create all exact discrete inputs for one nonblank cold example."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must be nonblank")
        ordered_labels = normalize_labels(labels)

        offsets = word_char_spans(text)
        if not offsets:
            raise PreprocessingError("text produced no model words")
        words = tuple(text[start:end] for start, end in offsets)
        starts = tuple(start for start, _ in offsets)
        ends = tuple(end for _, end in offsets)

        prompt_words_list: list[str] = []
        for label in ordered_labels:
            prompt_words_list.extend((label, self.config.label_token))
        prompt_words_list.append(self.config.separator_token)
        prompt_words = tuple(prompt_words_list)
        input_words = prompt_words + words
        input_ids, attention, words_mask, tokenizer_tokens = self._tokenize_word_row(
            input_words,
            prompt_word_length=len(prompt_words),
        )

        span_rows, span_validity = prepare_cold_span_candidates(len(words), self.config.max_width)
        return ColdPreprocessingResult(
            text=text,
            labels=ordered_labels,
            word_tokens=words,
            word_char_starts=starts,
            word_char_ends=ends,
            serialized_prompt="".join(prompt_words),
            serialized_prompt_words=prompt_words,
            serialized_input_words=input_words,
            prompt_word_length=len(prompt_words),
            tokenizer_tokens=tokenizer_tokens,
            max_width=self.config.max_width,
            label_token_id=self.config.label_token_id,
            separator_token_id=self.config.separator_token_id,
            input_ids=input_ids,
            attention_mask=attention,
            label_attention_mask=attention,
            words_mask=words_mask[np.newaxis, :],
            text_lengths=np.asarray([[len(words)]], dtype=np.int64),
            span_idx=span_rows[np.newaxis, :, :],
            span_mask=span_validity[np.newaxis, :],
            label_token_positions=np.argwhere(input_ids == self.config.label_token_id),
            separator_token_positions=np.argwhere(input_ids == self.config.separator_token_id),
        )

    def initialize_session(
        self,
        text: str,
        labels: Sequence[str],
        *,
        context_limit: int | None,
    ) -> SessionInitializationResult:
        """Cold-start a session with the label prompt included exactly once.

        A blank initial chunk is rejected, matching the reference's absence of a
        session after a blank-only call.  ``context_limit`` is mandatory and may
        be ``None`` only when the caller explicitly chooses no finite limit.
        """

        _validate_context_limit(context_limit)
        cold = self.preprocess(text, labels)
        decoder_token_count = int(cold.attention_mask.sum())
        _require_context_capacity(
            decoder_token_count,
            context_limit,
            operation="cold initialization",
        )
        state = SessionPreprocessingState(
            labels=cold.labels,
            accumulated_text=cold.text,
            word_tokens=cold.word_tokens,
            word_char_starts=cold.word_char_starts,
            word_char_ends=cold.word_char_ends,
            decoder_token_count=decoder_token_count,
        )
        return SessionInitializationResult(cold, state, context_limit)

    def preprocess_warm(
        self,
        state: SessionPreprocessingState,
        chunk: str,
        labels: Sequence[str],
        *,
        context_limit: int | None,
        right_context_width: int,
        recompute_all_candidates: bool = False,
    ) -> WarmChunkPreprocessingResult:
        """Preprocess one append without serializing the cached label prompt.

        ``recompute_all_candidates`` controls span enumeration only; a user
        request to rebuild the whole decoder cache must instead cold-preprocess
        the complete accumulated text.  Blank chunks are explicit no-ops: they
        do not invoke the tokenizer, consume positions, append whitespace, or
        check a changed label set, which matches GLiNER 0.2.28's public session
        inference path.
        """

        if not isinstance(state, SessionPreprocessingState):
            raise TypeError("state must be a SessionPreprocessingState")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        ordered_labels = normalize_labels(labels)
        _validate_context_limit(context_limit)
        if (
            isinstance(right_context_width, bool)
            or not isinstance(right_context_width, int)
            or right_context_width < 0
        ):
            raise ValueError("right_context_width must be a nonnegative integer")
        if not isinstance(recompute_all_candidates, bool):
            raise TypeError("recompute_all_candidates must be a boolean")
        _require_context_capacity(
            state.decoder_token_count,
            context_limit,
            operation="existing session",
        )

        if not chunk.strip():
            empty_tokens = np.empty((1, 0), dtype=np.int64)
            empty_spans = np.empty((1, 0, 2), dtype=np.int64)
            empty_span_mask = np.empty((1, 0), dtype=np.bool_)
            return WarmChunkPreprocessingResult(
                chunk=chunk,
                is_noop=True,
                previous_state=state,
                next_state=state,
                current_word_tokens=(),
                current_word_char_starts=(),
                current_word_char_ends=(),
                absolute_word_char_starts=(),
                absolute_word_char_ends=(),
                tokenizer_tokens=(),
                context_limit=context_limit,
                max_width=self.config.max_width,
                right_context_width=right_context_width,
                recompute_all_candidates=recompute_all_candidates,
                input_ids=empty_tokens,
                attention_mask=empty_tokens,
                label_attention_mask=empty_tokens,
                full_attention_mask=np.ones((1, state.decoder_token_count), dtype=np.int64),
                position_ids=empty_tokens,
                words_mask=empty_tokens,
                text_lengths=np.zeros((1, 1), dtype=np.int64),
                past_word_lengths=np.asarray([len(state.word_tokens)], dtype=np.int64),
                span_idx=empty_spans,
                span_mask=empty_span_mask,
            )

        if ordered_labels != state.labels:
            raise ValueError(
                "labels changed for an initialized session; rebuild cold or clear the session"
            )

        offsets = word_char_spans(chunk)
        if not offsets:
            raise PreprocessingError("nonblank warm chunk produced no model words")
        current_words = tuple(chunk[start:end] for start, end in offsets)
        current_starts = tuple(start for start, _ in offsets)
        current_ends = tuple(end for _, end in offsets)
        input_ids, current_attention, words_mask, tokenizer_tokens = self._tokenize_word_row(
            current_words,
            prompt_word_length=0,
        )
        new_token_count = int(current_attention.sum())
        total_token_count = state.decoder_token_count + new_token_count
        _require_context_capacity(total_token_count, context_limit, operation="warm append")

        char_offset = len(state.accumulated_text)
        absolute_starts = tuple(char_offset + start for start in current_starts)
        absolute_ends = tuple(char_offset + end for end in current_ends)
        next_state = SessionPreprocessingState(
            labels=state.labels,
            accumulated_text=state.accumulated_text + chunk,
            word_tokens=state.word_tokens + current_words,
            word_char_starts=state.word_char_starts + absolute_starts,
            word_char_ends=state.word_char_ends + absolute_ends,
            decoder_token_count=total_token_count,
        )
        span_rows, span_validity = prepare_streaming_span_candidates(
            len(state.word_tokens),
            len(current_words),
            self.config.max_width,
            recompute_all=recompute_all_candidates,
            right_context_width=right_context_width,
        )
        return WarmChunkPreprocessingResult(
            chunk=chunk,
            is_noop=False,
            previous_state=state,
            next_state=next_state,
            current_word_tokens=current_words,
            current_word_char_starts=current_starts,
            current_word_char_ends=current_ends,
            absolute_word_char_starts=absolute_starts,
            absolute_word_char_ends=absolute_ends,
            tokenizer_tokens=tokenizer_tokens,
            context_limit=context_limit,
            max_width=self.config.max_width,
            right_context_width=right_context_width,
            recompute_all_candidates=recompute_all_candidates,
            input_ids=input_ids,
            attention_mask=current_attention,
            label_attention_mask=current_attention,
            full_attention_mask=np.ones((1, total_token_count), dtype=np.int64),
            position_ids=np.arange(
                state.decoder_token_count,
                total_token_count,
                dtype=np.int64,
            )[None, :],
            words_mask=words_mask[None, :],
            text_lengths=np.asarray([[len(current_words)]], dtype=np.int64),
            past_word_lengths=np.asarray([len(state.word_tokens)], dtype=np.int64),
            span_idx=span_rows[None, :, :],
            span_mask=span_validity[None, :],
        )


def _prepare_first_word_mask(
    word_ids: object,
    *,
    prompt_word_length: int,
    expected_input_words: int,
    expected_token_count: int,
) -> IntArray:
    if isinstance(word_ids, str | bytes) or not isinstance(word_ids, Sequence):
        raise PreprocessingError("tokenizer word_ids must be a sequence")
    if len(word_ids) != expected_token_count:
        raise PreprocessingError("tokenizer word_ids do not align with input IDs")

    mask: list[int] = []
    previous_word_id: int | None = None
    seen_words = 0
    for raw_word_id in word_ids:
        if raw_word_id is None:
            mask.append(0)
            previous_word_id = None
            continue
        if isinstance(raw_word_id, bool) or not isinstance(raw_word_id, int) or raw_word_id < 0:
            raise PreprocessingError("tokenizer word IDs must be nonnegative integers or None")
        first_subtoken = raw_word_id != previous_word_id
        if first_subtoken:
            if raw_word_id != seen_words:
                raise PreprocessingError("tokenizer word IDs must be contiguous and ordered")
            seen_words += 1
        if seen_words <= prompt_word_length or not first_subtoken:
            mask.append(0)
        else:
            mask.append(seen_words - prompt_word_length)
        previous_word_id = raw_word_id

    if seen_words != expected_input_words:
        raise PreprocessingError(
            f"tokenizer exposed {seen_words} words, expected {expected_input_words}"
        )
    return _readonly_int_array(mask, name="words_mask", ndim=1)


__all__ = [
    "BoolArray",
    "ColdPreprocessingResult",
    "ColdPreprocessor",
    "ContextLimitExceededError",
    "IntArray",
    "PreprocessingConfig",
    "PreprocessingError",
    "SessionInitializationResult",
    "SessionPreprocessingState",
    "WarmChunkPreprocessingResult",
    "normalize_labels",
    "prepare_cold_span_candidates",
    "prepare_streaming_span_candidates",
]
