from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.preprocessing import (
    ColdPreprocessor,
    ContextLimitExceededError,
    PreprocessingConfig,
    SessionPreprocessingState,
    prepare_streaming_span_candidates,
)


class FakeEncoding(dict[str, np.ndarray[Any, Any]]):
    def __init__(self, token_ids: list[int], word_ids: list[int]) -> None:
        super().__init__(
            input_ids=np.asarray([token_ids], dtype=np.int64),
            attention_mask=np.ones((1, len(token_ids)), dtype=np.int64),
        )
        self._word_ids = tuple(word_ids)

    def word_ids(self, batch_index: int = 0) -> list[int]:
        if batch_index != 0:
            raise IndexError(batch_index)
        return list(self._word_ids)


class RecordingTokenizer:
    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._ids = {
            "person": 1,
            "email": 2,
            "address": 3,
            "Ada": 4,
            "em": 5,
            "ailed": 6,
            "Jo": 7,
            ".": 8,
            "Lovelace": 9,
            "phone": 10,
            "marker-content": 90,
            "<<LABEL>>": 90,
            "<<SEP>>": 91,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, 99)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        reverse = {value: key for key, value in self._ids.items()}
        return [reverse.get(token_id, "[UNK]") for token_id in token_ids]

    def __call__(
        self,
        rows: list[list[str]],
        *,
        is_split_into_words: bool,
        return_tensors: str,
        truncation: bool,
        padding: str,
        add_special_tokens: bool,
    ) -> FakeEncoding:
        assert len(rows) == 1
        assert is_split_into_words is True
        assert return_tensors == "np"
        assert truncation is False
        assert padding == "longest"
        assert add_special_tokens is False
        self.calls.append(tuple(rows[0]))

        token_ids: list[int] = []
        word_ids: list[int] = []
        for word_id, word in enumerate(rows[0]):
            if word == "email address":
                pieces = ("email", "address")
            elif word == "emailed":
                pieces = ("em", "ailed")
            else:
                pieces = (word,)
            token_ids.extend(self.convert_tokens_to_ids(piece) for piece in pieces)
            word_ids.extend([word_id] * len(pieces))
        return FakeEncoding(token_ids, word_ids)


@pytest.fixture
def streaming_preprocessor() -> tuple[ColdPreprocessor, RecordingTokenizer]:
    tokenizer = RecordingTokenizer()
    preprocessor = ColdPreprocessor(
        PreprocessingConfig(
            max_width=3,
            label_token="<<LABEL>>",
            label_token_id=90,
            separator_token="<<SEP>>",
            separator_token_id=91,
        ),
        tokenizer,
    )
    return preprocessor, tokenizer


def test_session_initialization_is_cold_and_includes_prompt_once(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, tokenizer = streaming_preprocessor

    initialized = preprocessor.initialize_session(
        "Ada ",
        ["person", "email address", "person"],
        context_limit=32,
    )

    assert initialized.cold.labels == ("person", "email address")
    assert tokenizer.calls == [
        (
            "person",
            "<<LABEL>>",
            "email address",
            "<<LABEL>>",
            "<<SEP>>",
            "Ada",
        )
    ]
    assert initialized.state == SessionPreprocessingState(
        labels=("person", "email address"),
        accumulated_text="Ada ",
        word_tokens=("Ada",),
        word_char_starts=(0,),
        word_char_ends=(3,),
        decoder_token_count=7,
    )
    assert initialized.context_limit == 32


def test_warm_append_tokenizes_only_new_words_and_extends_absolute_metadata(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session(
        "Ada ",
        ["person", "email address"],
        context_limit=32,
    )

    warm = preprocessor.preprocess_warm(
        initialized.state,
        "emailed Jo.",
        ["person", "email address", "person"],
        context_limit=32,
        right_context_width=2,
    )

    assert tokenizer.calls[-1] == ("emailed", "Jo", ".")
    assert "<<LABEL>>" not in tokenizer.calls[-1]
    assert "<<SEP>>" not in tokenizer.calls[-1]
    assert warm.is_noop is False
    assert warm.current_word_tokens == ("emailed", "Jo", ".")
    assert warm.current_word_char_starts == (0, 8, 10)
    assert warm.current_word_char_ends == (7, 10, 11)
    assert warm.absolute_word_char_starts == (4, 12, 14)
    assert warm.absolute_word_char_ends == (11, 14, 15)
    assert warm.next_state.accumulated_text == "Ada emailed Jo."
    assert warm.next_state.word_tokens == ("Ada", "emailed", "Jo", ".")
    assert warm.next_state.word_char_starts == (0, 4, 12, 14)
    assert warm.next_state.word_char_ends == (3, 11, 14, 15)

    np.testing.assert_array_equal(warm.input_ids, [[5, 6, 7, 8]])
    np.testing.assert_array_equal(warm.attention_mask, [[1, 1, 1, 1]])
    np.testing.assert_array_equal(warm.label_attention_mask, warm.attention_mask)
    np.testing.assert_array_equal(warm.words_mask, [[1, 0, 2, 3]])
    np.testing.assert_array_equal(warm.text_lengths, [[3]])
    np.testing.assert_array_equal(warm.past_word_lengths, [1])
    np.testing.assert_array_equal(warm.position_ids, [[7, 8, 9, 10]])
    np.testing.assert_array_equal(warm.full_attention_mask, np.ones((1, 11), dtype=np.int64))
    assert warm.cached_token_count == 7
    assert warm.new_token_count == 4
    assert warm.total_token_count == 11
    assert warm.past_word_count == 1
    assert warm.new_word_count == 3
    assert warm.as_model_inputs()["attention_mask"] is warm.full_attention_mask
    assert warm.as_model_inputs()["label_attention_mask"] is warm.label_attention_mask

    expected_rows, expected_mask = _direct_reference_candidates(1, 3, 3, 2, False)
    np.testing.assert_array_equal(warm.span_idx[0], expected_rows)
    np.testing.assert_array_equal(warm.span_mask[0], expected_mask)


def test_chunk_boundaries_do_not_retokenize_or_merge_prior_words(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, _tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session("Ada", ["person"], context_limit=None)

    warm = preprocessor.preprocess_warm(
        initialized.state,
        "Lovelace",
        ["person"],
        context_limit=None,
        right_context_width=3,
    )

    assert warm.next_state.accumulated_text == "AdaLovelace"
    assert warm.next_state.word_tokens == ("Ada", "Lovelace")
    assert warm.next_state.word_char_starts == (0, 3)
    assert warm.next_state.word_char_ends == (3, 11)
    assert warm.current_word_char_starts == (0,)
    assert warm.absolute_word_char_starts == (3,)


def test_warm_content_may_tokenize_to_a_prompt_marker_id(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, _tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session("Ada ", ["person"], context_limit=16)

    warm = preprocessor.preprocess_warm(
        initialized.state,
        "marker-content",
        ["person"],
        context_limit=16,
        right_context_width=3,
    )

    np.testing.assert_array_equal(warm.input_ids, [[90]])
    assert warm.current_word_tokens == ("marker-content",)
    assert warm.next_state.accumulated_text == "Ada marker-content"


@pytest.mark.parametrize("blank", ["", " ", "\t\r\n  "])
def test_blank_warm_chunks_are_explicit_noops_before_label_comparison(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
    blank: str,
) -> None:
    preprocessor, tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session("Ada", ["person"], context_limit=16)
    call_count = len(tokenizer.calls)

    result = preprocessor.preprocess_warm(
        initialized.state,
        blank,
        ["phone"],
        context_limit=16,
        right_context_width=3,
    )

    assert result.is_noop is True
    assert result.previous_state is initialized.state
    assert result.next_state is initialized.state
    assert result.next_state.accumulated_text == "Ada"
    assert len(tokenizer.calls) == call_count
    assert result.input_ids.shape == (1, 0)
    assert result.position_ids.shape == (1, 0)
    assert result.span_idx.shape == (1, 0, 2)
    assert result.span_mask.shape == (1, 0)
    assert result.full_attention_mask.shape == (1, initialized.state.decoder_token_count)


def test_nonblank_warm_append_requires_fixed_normalized_labels(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session("Ada ", ["person"], context_limit=16)
    call_count = len(tokenizer.calls)

    with pytest.raises(ValueError, match="labels changed"):
        preprocessor.preprocess_warm(
            initialized.state,
            "Lovelace",
            ["phone"],
            context_limit=16,
            right_context_width=3,
        )

    assert len(tokenizer.calls) == call_count
    assert initialized.state.accumulated_text == "Ada "


def test_context_limit_is_explicit_and_never_evicts_state(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, _tokenizer = streaming_preprocessor

    with pytest.raises(ContextLimitExceededError, match="cold initialization.*7.*6"):
        preprocessor.initialize_session(
            "Ada ",
            ["person", "email address"],
            context_limit=6,
        )

    initialized = preprocessor.initialize_session(
        "Ada ",
        ["person", "email address"],
        context_limit=7,
    )
    with pytest.raises(ContextLimitExceededError, match="warm append.*11.*10"):
        preprocessor.preprocess_warm(
            initialized.state,
            "emailed Jo.",
            ["person", "email address"],
            context_limit=10,
            right_context_width=2,
        )

    assert initialized.state.decoder_token_count == 7
    assert initialized.state.accumulated_text == "Ada "


@pytest.mark.parametrize("context_limit", [0, -1, True, 1.5, "10"])
def test_invalid_context_limits_fail_closed(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
    context_limit: Any,
) -> None:
    preprocessor, _tokenizer = streaming_preprocessor

    with pytest.raises((TypeError, ValueError), match="context_limit"):
        preprocessor.initialize_session(
            "Ada",
            ["person"],
            context_limit=context_limit,
        )


def test_session_records_and_warm_arrays_are_immutable(
    streaming_preprocessor: tuple[ColdPreprocessor, RecordingTokenizer],
) -> None:
    preprocessor, _tokenizer = streaming_preprocessor
    initialized = preprocessor.initialize_session("Ada ", ["person"], context_limit=16)
    warm = preprocessor.preprocess_warm(
        initialized.state,
        "Lovelace",
        ["person"],
        context_limit=16,
        right_context_width=3,
    )

    with pytest.raises(FrozenInstanceError):
        initialized.state.accumulated_text = "changed"  # type: ignore[misc]
    for array in warm.as_model_inputs().values():
        assert array.flags.owndata
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0


def _direct_reference_candidates(
    past_words: int,
    new_words: int,
    max_width: int,
    right_context_width: int,
    recompute_all: bool,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Literal small-sequence oracle for GLiNER 0.2.28's helper semantics."""

    total_words = past_words + new_words
    if total_words == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.bool_)
    if recompute_all:
        minimum_end = 0
    else:
        minimum_end = min(
            past_words,
            max(0, total_words - 1 - right_context_width),
        )
    first_start = max(0, minimum_end - max_width + 1)
    rows = np.asarray(
        [
            (start, start + width_offset)
            for start in range(first_start, total_words)
            for width_offset in range(max_width)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    valid = np.asarray(
        [
            end < total_words and (recompute_all or (new_words > 0 and end >= minimum_end))
            for _start, end in rows.tolist()
        ],
        dtype=np.bool_,
    )
    return rows, valid


def test_candidate_generation_exhaustively_matches_direct_reference_semantics() -> None:
    combinations = 0
    for past_words in range(6):
        for new_words in range(6):
            for max_width in range(1, 6):
                for right_context_width in range(6):
                    for recompute_all in (False, True):
                        expected_rows, expected_mask = _direct_reference_candidates(
                            past_words,
                            new_words,
                            max_width,
                            right_context_width,
                            recompute_all,
                        )
                        actual_rows, actual_mask = prepare_streaming_span_candidates(
                            past_words,
                            new_words,
                            max_width,
                            recompute_all=recompute_all,
                            right_context_width=right_context_width,
                        )
                        np.testing.assert_array_equal(actual_rows, expected_rows)
                        np.testing.assert_array_equal(actual_mask, expected_mask)
                        combinations += 1
    assert combinations == 2_160


def _exported_parity_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "reference" / REFERENCE_REVISION


def test_pinned_artifact_cold_then_warm_reconstructs_reference_cold_fixture() -> None:
    root = _exported_parity_root()
    parity_root = root / "parity"
    if (
        not (root / "export_manifest.json").is_file()
        or not (parity_root / "parity_arrays.npz").is_file()
    ):
        pytest.skip("ignored local reference export is not present")

    had_torch = "torch" in sys.modules
    had_gliner = "gliner" in sys.modules
    bundle = load_asset_bundle(
        root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    preprocessor = ColdPreprocessor.from_asset_bundle(bundle)
    metadata = json.loads((parity_root / "parity_metadata.json").read_text(encoding="utf-8"))
    initialized = preprocessor.initialize_session(
        "Ada ",
        metadata["labels"],
        context_limit=40_960,
    )
    warm = preprocessor.preprocess_warm(
        initialized.state,
        "emailed Jo.",
        metadata["labels"],
        context_limit=40_960,
        right_context_width=12,
    )

    with np.load(parity_root / "parity_arrays.npz", allow_pickle=False) as arrays:
        combined_input_ids = np.concatenate((initialized.cold.input_ids, warm.input_ids), axis=1)
        combined_words_mask = np.concatenate(
            (
                initialized.cold.words_mask,
                np.where(warm.words_mask > 0, warm.words_mask + 1, 0),
            ),
            axis=1,
        )
        np.testing.assert_array_equal(combined_input_ids, arrays["input_ids"])
        np.testing.assert_array_equal(warm.full_attention_mask, arrays["attention_mask"])
        np.testing.assert_array_equal(combined_words_mask, arrays["words_mask"])
        np.testing.assert_array_equal(warm.span_idx, arrays["span_idx"])
        np.testing.assert_array_equal(warm.span_mask, arrays["span_mask"])

    assert warm.next_state.accumulated_text == metadata["text"]
    assert list(warm.next_state.word_tokens) == metadata["word_tokens"]
    assert list(warm.next_state.word_char_starts) == metadata["word_char_starts"]
    assert list(warm.next_state.word_char_ends) == metadata["word_char_ends"]
    if not had_torch:
        assert "torch" not in sys.modules
    if not had_gliner:
        assert "gliner" not in sys.modules
