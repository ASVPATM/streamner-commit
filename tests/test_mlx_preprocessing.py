from __future__ import annotations

import json
import sys
from dataclasses import replace
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
    PreprocessingConfig,
    PreprocessingError,
    normalize_labels,
    prepare_cold_span_candidates,
    prepare_streaming_span_candidates,
)


class FakeEncoding(dict[str, np.ndarray[Any, Any]]):
    def __init__(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        word_ids: list[int | None],
    ) -> None:
        super().__init__(
            input_ids=np.asarray([input_ids], dtype=np.int64),
            attention_mask=np.asarray([attention_mask], dtype=np.int64),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        if batch_index != 0:
            raise IndexError(batch_index)
        return list(self._word_ids)


class FakeTokenizer:
    padding_side = "right"

    def __init__(self, *, label_id: int = 90, separator_id: int = 91) -> None:
        self.label_id = label_id
        self.separator_id = separator_id
        self._piece_ids = {
            "person": 1,
            "email": 2,
            "address": 3,
            "Ada": 4,
            "em": 5,
            "ailed": 6,
            "Jo": 7,
            ".": 8,
            "José": 9,
            ",": 10,
            "東京": 11,
            "+": 12,
            "1": 13,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<<LABEL>>":
            return self.label_id
        if token == "<<SEP>>":
            return self.separator_id
        return self._piece_ids.get(token, 99)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        reverse = {value: key for key, value in self._piece_ids.items()}
        reverse[self.label_id] = "<<LABEL>>"
        reverse[self.separator_id] = "<<SEP>>"
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

        token_ids: list[int] = []
        word_ids: list[int] = []
        for word_id, word in enumerate(rows[0]):
            if word == "email address":
                pieces = ["email", "address"]
            elif word == "emailed":
                pieces = ["em", "ailed"]
            else:
                pieces = [word]
            token_ids.extend(self.convert_tokens_to_ids(piece) for piece in pieces)
            word_ids.extend([word_id] * len(pieces))
        return FakeEncoding(token_ids, [1] * len(token_ids), word_ids)


@pytest.fixture
def fake_preprocessor() -> ColdPreprocessor:
    return ColdPreprocessor(
        PreprocessingConfig(
            max_width=3,
            label_token="<<LABEL>>",
            label_token_id=90,
            separator_token="<<SEP>>",
            separator_token_id=91,
        ),
        FakeTokenizer(),
    )


def test_cold_preprocessing_matches_reference_packing_and_candidate_order(
    fake_preprocessor: ColdPreprocessor,
) -> None:
    result = fake_preprocessor.preprocess(
        "Ada emailed Jo.",
        ["person", "email address", "person"],
    )

    assert result.labels == ("person", "email address")
    assert result.word_tokens == ("Ada", "emailed", "Jo", ".")
    assert result.word_char_starts == (0, 4, 12, 14)
    assert result.word_char_ends == (3, 11, 14, 15)
    assert result.serialized_prompt_words == (
        "person",
        "<<LABEL>>",
        "email address",
        "<<LABEL>>",
        "<<SEP>>",
    )
    assert result.serialized_prompt == "person<<LABEL>>email address<<LABEL>><<SEP>>"
    assert result.serialized_input_words == result.serialized_prompt_words + result.word_tokens
    assert result.prompt_word_length == 5
    np.testing.assert_array_equal(
        result.input_ids,
        [[1, 90, 2, 3, 90, 91, 4, 5, 6, 7, 8]],
    )
    np.testing.assert_array_equal(result.attention_mask, np.ones((1, 11), dtype=np.int64))
    np.testing.assert_array_equal(result.label_attention_mask, result.attention_mask)
    np.testing.assert_array_equal(
        result.words_mask,
        [[0, 0, 0, 0, 0, 0, 1, 2, 0, 3, 4]],
    )
    np.testing.assert_array_equal(result.text_lengths, [[4]])
    np.testing.assert_array_equal(result.label_token_positions, [[0, 1], [0, 4]])
    np.testing.assert_array_equal(result.separator_token_positions, [[0, 5]])

    expected_rows = [[start, start + width] for start in range(4) for width in range(3)]
    np.testing.assert_array_equal(result.span_idx, [expected_rows])
    np.testing.assert_array_equal(
        result.span_mask,
        [[[end < 4 for _, end in expected_rows]]][0],
    )
    assert tuple(result.as_model_inputs()) == (
        "input_ids",
        "attention_mask",
        "label_attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask",
    )


def test_model_word_offsets_preserve_unicode_and_split_symbols(
    fake_preprocessor: ColdPreprocessor,
) -> None:
    text = "  José,\n東京 +1"

    result = fake_preprocessor.preprocess(text, ["person"])

    assert result.word_tokens == ("José", ",", "東京", "+", "1")
    assert (
        tuple(
            text[start:end]
            for start, end in zip(result.word_char_starts, result.word_char_ends, strict=True)
        )
        == result.word_tokens
    )
    np.testing.assert_array_equal(result.text_lengths, [[5]])


def test_result_arrays_are_owned_and_read_only(fake_preprocessor: ColdPreprocessor) -> None:
    result = fake_preprocessor.preprocess("Ada.", ["person"])

    for array in result.as_model_inputs().values():
        assert array.flags.owndata
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 123
    assert not result.label_token_positions.flags.writeable
    assert not result.separator_token_positions.flags.writeable


@pytest.mark.parametrize(
    ("labels", "exception", "message"),
    [
        ([], ValueError, "at least one"),
        ([""], ValueError, "nonblank"),
        (["   "], ValueError, "nonblank"),
        (["person", 3], TypeError, "must be a string"),
        ("person", TypeError, "ordered sequence"),
    ],
)
def test_label_validation_is_fail_closed(
    labels: Any, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        normalize_labels(labels)


def test_duplicate_label_normalization_preserves_first_occurrence() -> None:
    assert normalize_labels(["email", "person", "email", "phone"]) == (
        "email",
        "person",
        "phone",
    )


@pytest.mark.parametrize("text", ["", " ", "\t\r\n"])
def test_blank_text_is_rejected(fake_preprocessor: ColdPreprocessor, text: str) -> None:
    with pytest.raises(ValueError, match="nonblank"):
        fake_preprocessor.preprocess(text, ["person"])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_width": 0}, "positive"),
        ({"words_splitter_type": "spacy"}, "whitespace"),
        ({"subtoken_pooling": "mean"}, "first"),
        ({"label_token_id": 91}, "must differ"),
    ],
)
def test_only_pinned_discrete_config_is_accepted(changes: dict[str, Any], message: str) -> None:
    base = PreprocessingConfig(3, "<<LABEL>>", 90, "<<SEP>>", 91)

    with pytest.raises(PreprocessingError, match=message):
        replace(base, **changes)


def test_tokenizer_marker_ids_are_cross_checked() -> None:
    config = PreprocessingConfig(3, "<<LABEL>>", 90, "<<SEP>>", 91)

    with pytest.raises(PreprocessingError, match="label marker ID mismatch"):
        ColdPreprocessor(config, FakeTokenizer(label_id=999))


def test_reserved_marker_collision_is_rejected(fake_preprocessor: ColdPreprocessor) -> None:
    with pytest.raises(PreprocessingError, match="exactly one label marker"):
        fake_preprocessor.preprocess("Ada", ["<<LABEL>>"])


def test_cold_candidate_helper_keeps_masked_out_of_range_rows() -> None:
    rows, mask = prepare_cold_span_candidates(2, 3)

    np.testing.assert_array_equal(
        rows,
        [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [1, 3]],
    )
    np.testing.assert_array_equal(mask, [True, True, False, True, False, False])
    assert not rows.flags.writeable
    assert not mask.flags.writeable


def test_streaming_candidate_helper_matches_rolling_reference_rules() -> None:
    rows, mask = prepare_streaming_span_candidates(
        4,
        2,
        3,
        right_context_width=2,
    )

    expected_rows = [[start, start + width] for start in range(1, 6) for width in range(3)]
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_array_equal(mask, [end < 6 and end >= 3 for _, end in expected_rows])

    no_new_rows, no_new_mask = prepare_streaming_span_candidates(
        4,
        0,
        3,
        right_context_width=2,
    )
    assert no_new_rows.shape == (4 * 3, 2)
    assert not no_new_mask.any()

    all_rows, all_mask = prepare_streaming_span_candidates(
        4,
        2,
        3,
        recompute_all=True,
        right_context_width=2,
    )
    assert all_rows.shape == (6 * 3, 2)
    np.testing.assert_array_equal(all_mask, all_rows[:, 1] < 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"past_words": -1, "new_words": 1, "max_width": 3},
        {"past_words": 0, "new_words": -1, "max_width": 3},
        {"past_words": 0, "new_words": 1, "max_width": 0},
        {
            "past_words": 0,
            "new_words": 1,
            "max_width": 3,
            "right_context_width": -1,
        },
    ],
)
def test_candidate_helper_rejects_invalid_dimensions(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        prepare_streaming_span_candidates(**kwargs)


def _exported_parity_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "reference" / REFERENCE_REVISION


def test_exported_reference_discrete_parity_when_local_artifacts_exist() -> None:
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
    with np.load(parity_root / "parity_arrays.npz", allow_pickle=False) as arrays:
        result = preprocessor.preprocess(metadata["text"], metadata["labels"])
        for field in (
            "input_ids",
            "attention_mask",
            "label_attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
            "label_token_positions",
            "separator_token_positions",
        ):
            np.testing.assert_array_equal(getattr(result, field), arrays[field], err_msg=field)

    assert list(result.labels) == metadata["labels"]
    assert list(result.word_tokens) == metadata["word_tokens"]
    assert list(result.word_char_starts) == metadata["word_char_starts"]
    assert list(result.word_char_ends) == metadata["word_char_ends"]
    assert result.serialized_prompt == metadata["serialized_prompt"]
    assert list(result.serialized_prompt_words) == metadata["serialized_prompt_words"]
    assert list(result.serialized_input_words) == metadata["serialized_input_words"]
    assert result.prompt_word_length == metadata["prompt_word_length"]
    assert list(result.tokenizer_tokens) == metadata["tokenizer_tokens"]
    if not had_torch:
        assert "torch" not in sys.modules
    if not had_gliner:
        assert "gliner" not in sys.modules
