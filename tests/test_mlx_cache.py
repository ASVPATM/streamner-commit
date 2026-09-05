from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from streamner_commit.mlx.cache import (
    ContextLimitError,
    MLXStreamingState,
    ReleasedStateError,
    SessionLabelsError,
    StreamingStateError,
)


@dataclass
class FakeConfiguration:
    hidden_size: int = 4
    max_position_embeddings: int = 16


@dataclass
class FakeKVCache:
    offset: int
    keys: object | None = None
    values: object | None = None


class FakeQwen:
    def __init__(self, *, hidden_size: int = 4, context_limit: int = 16) -> None:
        self.configuration = FakeConfiguration(hidden_size, context_limit)
        self.validation_calls: list[int | None] = []

    def validate_cache(
        self,
        cache: list[FakeKVCache],
        *,
        expected_offset: int | None = None,
    ) -> int:
        self.validation_calls.append(expected_offset)
        if len(cache) != 2 or len({id(layer) for layer in cache}) != 2:
            raise ValueError("invalid native cache layers")
        offsets = {layer.offset for layer in cache}
        if len(offsets) != 1:
            raise ValueError("layer offsets disagree")
        offset = offsets.pop()
        if expected_offset is not None and offset != expected_offset:
            raise ValueError(f"expected {expected_offset}, got {offset}")
        return offset


def _state(
    *,
    qwen: FakeQwen | None = None,
    caches: list[FakeKVCache] | None = None,
    token_count: int = 6,
    context_limit: int = 12,
    past_word_embeddings: object | None = None,
    past_word_mask: object | None = None,
    word_count: int = 2,
    prompt_representations: object | None = None,
    prompt_mask: object | None = None,
    labels: tuple[str, ...] = ("person", "email"),
    text: str = "Ada.",
    word_tokens: tuple[str, ...] = ("Ada", "."),
    word_char_starts: tuple[int, ...] = (0, 3),
    word_char_ends: tuple[int, ...] = (3, 4),
    historical_logits: dict[tuple[int, int], object] | None = None,
) -> MLXStreamingState:
    qwen = qwen or FakeQwen()
    caches = caches or [FakeKVCache(token_count), FakeKVCache(token_count)]
    return MLXStreamingState(
        session_id="session-1",
        qwen=qwen,
        qwen_cache=caches,
        token_count=token_count,
        context_limit=context_limit,
        past_word_embeddings=(
            np.zeros((1, 3, 4), dtype=np.float32)
            if past_word_embeddings is None
            else past_word_embeddings
        ),
        past_word_mask=(
            np.asarray([[1, 1, 0]], dtype=np.bool_) if past_word_mask is None else past_word_mask
        ),
        word_count=word_count,
        prompt_representations=(
            np.zeros((1, len(labels), 4), dtype=np.float32)
            if prompt_representations is None
            else prompt_representations
        ),
        prompt_mask=(
            np.ones((1, len(labels)), dtype=np.bool_) if prompt_mask is None else prompt_mask
        ),
        labels=labels,
        text=text,
        word_tokens=word_tokens,
        word_char_starts=word_char_starts,
        word_char_ends=word_char_ends,
        historical_logits=(
            {(0, 0): np.asarray([0.25, -0.5], dtype=np.float32)}
            if historical_logits is None
            else historical_logits
        ),
    )


def test_state_records_independent_token_and_word_coordinates() -> None:
    qwen = FakeQwen()
    state = _state(qwen=qwen)

    assert state.session_id == "session-1"
    assert state.token_count == 6
    assert state.token_offset == 6
    assert state.next_token_position == 6
    assert state.word_count == 2
    assert state.labels == ("person", "email")
    assert state.word_tokens == ("Ada", ".")
    assert qwen.validation_calls[0] == 6
    state.validate()


def test_importing_state_does_not_import_torch_or_gliner() -> None:
    assert "torch" not in sys.modules
    assert "gliner" not in sys.modules


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"token_count": 13}, "context limit"),
        (
            {
                "caches": [FakeKVCache(6), FakeKVCache(5)],
            },
            "native Qwen cache",
        ),
        ({"past_word_mask": np.asarray([[1, 0, 1]])}, "contiguous valid prefix"),
        (
            {"prompt_representations": np.zeros((1, 1, 4), dtype=np.float32)},
            "prompt_representations",
        ),
        ({"labels": ("person", "person")}, "normalized, unique"),
        ({"word_char_ends": (2, 4)}, "round-trip"),
        (
            {"historical_logits": {(0, 0): np.asarray([0.1], dtype=np.float32)}},
            "complete 2-class",
        ),
        (
            {"historical_logits": {(0, 2): np.asarray([0.1, 0.2])}},
            "outside 2 visible words",
        ),
    ],
)
def test_construction_fails_closed_on_cross_field_mismatch(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(StreamingStateError, match=message):
        _state(**changes)


def test_context_limit_cannot_exceed_qwen_backbone_limit() -> None:
    with pytest.raises(StreamingStateError, match="exceeds Qwen limit"):
        _state(qwen=FakeQwen(context_limit=10), context_limit=11)


def test_labels_are_fixed_for_the_session_after_request_normalization() -> None:
    state = _state()

    assert state.ensure_labels(["person", "email", "person"]) == ("person", "email")
    with pytest.raises(SessionLabelsError, match="changed"):
        state.ensure_labels(["email", "person"])
    with pytest.raises(SessionLabelsError, match="nonblank"):
        state.ensure_labels([""])


def test_context_check_precedes_qwen_mutation_and_commit_checks_exact_offset() -> None:
    caches = [FakeKVCache(6), FakeKVCache(6)]
    state = _state(caches=caches, context_limit=8)

    assert state.ensure_context_capacity(2) == 8
    with pytest.raises(ContextLimitError, match="clear the session"):
        state.ensure_context_capacity(3)

    for cache in caches:
        cache.offset = 8
    assert state.commit_token_append(2) == 8
    assert state.token_count == 8
    assert state.next_token_position == 8

    for cache in caches:
        cache.offset = 9
    with pytest.raises(StreamingStateError, match="must be cleared"):
        state.commit_token_append(0)
    assert state.token_count == 8


def test_word_history_replacement_is_append_only_and_rolls_back_on_failure() -> None:
    state = _state()
    new_embeddings = np.ones((1, 4, 4), dtype=np.float32)
    new_mask = np.asarray([[1, 1, 1, 0]], dtype=np.bool_)

    state.replace_word_history(
        past_word_embeddings=new_embeddings,
        past_word_mask=new_mask,
        word_count=3,
        text="Ada. Jo",
        word_tokens=("Ada", ".", "Jo"),
        word_char_starts=(0, 3, 5),
        word_char_ends=(3, 4, 7),
    )

    assert state.word_count == 3
    assert state.text == "Ada. Jo"
    assert state.past_word_embeddings is new_embeddings
    with pytest.raises(StreamingStateError, match="preserve the prior"):
        state.replace_word_history(
            past_word_embeddings=np.zeros((1, 4, 4), dtype=np.float32),
            past_word_mask=new_mask,
            word_count=3,
            text="Ada. XX",
            word_tokens=("Ada", ".", "XX"),
            word_char_starts=(0, 3, 5),
            word_char_ends=(3, 4, 7),
        )
    assert state.text == "Ada. Jo"
    assert state.word_tokens == ("Ada", ".", "Jo")


def test_no_new_model_words_cannot_mutate_text() -> None:
    state = _state()

    with pytest.raises(StreamingStateError, match="no new model words"):
        state.replace_word_history(
            past_word_embeddings=state.past_word_embeddings,
            past_word_mask=state.past_word_mask,
            word_count=2,
            text="Ada.   ",
            word_tokens=state.word_tokens,
            word_char_starts=state.word_char_starts,
            word_char_ends=state.word_char_ends,
        )


def test_historical_logits_are_copied_to_private_read_only_cpu_storage() -> None:
    source = np.asarray([1.25, -3.5], dtype=np.float64)
    state = _state(historical_logits={(0, 0): source})
    source[:] = 99.0

    first_snapshot = state.historical_logits
    np.testing.assert_array_equal(first_snapshot[(0, 0)], [1.25, -3.5])
    assert first_snapshot[(0, 0)].dtype == np.float32
    assert first_snapshot[(0, 0)].flags.owndata
    assert not first_snapshot[(0, 0)].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first_snapshot[(0, 0)][0] = 7.0
    with pytest.raises(TypeError):
        first_snapshot[(1, 1)] = np.zeros(2, dtype=np.float32)  # type: ignore[index]

    detached = first_snapshot[(0, 0)]
    detached.setflags(write=True)
    detached[:] = 123.0
    np.testing.assert_array_equal(state.historical_logits[(0, 0)], [1.25, -3.5])


def test_candidate_merge_flattens_marker_grid_masks_invalid_rows_and_replaces() -> None:
    state = _state()
    boundaries = np.asarray(
        [[[[0, 0], [0, 1]], [[1, 1], [1, 2]]]],
        dtype=np.int64,
    )
    logits = np.asarray(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        dtype=np.float32,
    )
    mask = np.asarray([[[1, 1], [1, 0]]], dtype=np.bool_)

    updated = state.merge_candidate_logits(boundaries, logits, mask, replace_all=True)

    assert updated == 3
    history = state.historical_logits
    assert tuple(history) == ((0, 0), (0, 1), (1, 1))
    np.testing.assert_array_equal(history[(0, 1)], [3.0, 4.0])
    ordered_boundaries, ordered_logits = state.ordered_history_arrays()
    np.testing.assert_array_equal(ordered_boundaries, [[0, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(ordered_logits, [[1, 2], [3, 4], [5, 6]])
    assert not ordered_boundaries.flags.writeable
    assert not ordered_logits.flags.writeable


def test_candidate_merge_overwrites_complete_vector_and_rejects_duplicates() -> None:
    state = _state()
    source = np.asarray([[9.0, 10.0]], dtype=np.float32)

    assert state.merge_candidate_logits([[0, 0]], source) == 1
    source[:] = -1.0
    np.testing.assert_array_equal(state.historical_logits[(0, 0)], [9.0, 10.0])

    with pytest.raises(StreamingStateError, match="duplicate boundary"):
        state.merge_candidate_logits(
            [[0, 1], [0, 1]],
            [[1.0, 2.0], [3.0, 4.0]],
        )


def test_candidate_validation_is_atomic() -> None:
    state = _state()
    before = state.historical_logits

    with pytest.raises(StreamingStateError, match="non-finite"):
        state.merge_candidate_logits(
            [[0, 1], [1, 1]],
            [[1.0, 2.0], [np.nan, 4.0]],
        )

    np.testing.assert_array_equal(state.historical_logits[(0, 0)], before[(0, 0)])
    assert tuple(state.historical_logits) == ((0, 0),)


def test_empty_history_has_stable_decoder_shapes() -> None:
    state = _state(historical_logits={})

    boundaries, logits = state.ordered_history_arrays()

    assert boundaries.shape == (0, 2)
    assert boundaries.dtype == np.int64
    assert logits.shape == (0, 2)
    assert logits.dtype == np.float32


def test_clear_releases_every_owned_model_and_score_reference_idempotently() -> None:
    caches = [
        FakeKVCache(6, keys=object(), values=object()),
        FakeKVCache(6, keys=object(), values=object()),
    ]
    state = _state(caches=caches)

    state.clear()
    state.release()

    assert state.is_released
    assert state.qwen_cache == []
    assert all(
        cache.offset == 0 and cache.keys is None and cache.values is None for cache in caches
    )
    assert state.past_word_embeddings is None
    assert state.past_word_mask is None
    assert state.prompt_representations is None
    assert state.prompt_mask is None
    assert state.labels == ()
    assert state.text == ""
    assert state.word_tokens == ()
    assert state.token_count == 0
    assert state.word_count == 0
    with pytest.raises(ReleasedStateError, match="released"):
        state.validate()
    with pytest.raises(ReleasedStateError, match="released"):
        _ = state.historical_logits
    with pytest.raises(ReleasedStateError, match="released"):
        state.ensure_context_capacity(1)
