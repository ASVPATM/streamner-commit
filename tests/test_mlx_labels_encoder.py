from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from streamner_commit.mlx.labels_encoder import (
    extract_label_representations,
    slice_prompt,
)


def test_prompt_slice_compacts_left_padding_through_first_separator() -> None:
    hidden = mx.array(np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 7, 3))
    input_ids = mx.array(
        [
            [0, 41, 10, 99, 7, 8, 0],
            [41, 20, 21, 41, 99, 30, 31],
        ]
    )
    attention_mask = mx.array(
        [
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )

    prompt = slice_prompt(
        hidden,
        input_ids,
        attention_mask,
        separator_token_id=99,
    )

    np.testing.assert_array_equal(
        prompt.input_ids,
        np.array([[41, 10, 99, 0, 0], [41, 20, 21, 41, 99]]),
    )
    np.testing.assert_array_equal(
        prompt.attention_mask,
        np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
    )
    np.testing.assert_array_equal(prompt.hidden_states[0, :3], hidden[0, 1:4])
    np.testing.assert_array_equal(prompt.hidden_states[0, 3:], np.zeros((2, 3)))
    np.testing.assert_array_equal(prompt.hidden_states[1], hidden[1, :5])


def test_prompt_slice_rejects_rows_without_a_valid_separator() -> None:
    hidden = mx.zeros((2, 3, 4))
    input_ids = mx.array([[1, 99, 2], [1, 99, 2]])
    attention_mask = mx.array([[1, 1, 1], [1, 0, 1]])

    with pytest.raises(ValueError, match=r"batch rows \[1\]"):
        slice_prompt(
            hidden,
            input_ids,
            attention_mask,
            separator_token_id=99,
        )


def test_label_extraction_preserves_marker_order_and_pads_batch_rows() -> None:
    contextualized = mx.array(np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3))
    input_ids = mx.array(
        [
            [41, 10, 41, 11, 99, 0],
            [41, 12, 99, 0, 0, 0],
        ]
    )
    attention_mask = mx.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]])

    labels, label_mask = extract_label_representations(
        contextualized,
        input_ids,
        attention_mask,
        label_token_id=41,
    )

    np.testing.assert_array_equal(label_mask, np.array([[1, 1], [1, 0]]))
    np.testing.assert_array_equal(labels[0, 0], contextualized[0, 0])
    np.testing.assert_array_equal(labels[0, 1], contextualized[0, 2])
    np.testing.assert_array_equal(labels[1, 0], contextualized[1, 0])
    np.testing.assert_array_equal(labels[1, 1], np.zeros(3))


def test_label_extraction_ignores_masked_markers() -> None:
    contextualized = mx.ones((1, 4, 5))
    labels, label_mask = extract_label_representations(
        contextualized,
        mx.array([[41, 8, 41, 99]]),
        mx.array([[1, 1, 0, 1]]),
        label_token_id=41,
    )

    assert labels.shape == (1, 1, 5)
    np.testing.assert_array_equal(label_mask, np.array([[1]]))
