from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from streamner_commit.mlx.word_pooling import WordPoolingError, pool_first_subtokens


def test_first_subtoken_pooling_matches_one_based_word_coordinates() -> None:
    states = mx.arange(2 * 5 * 2, dtype=mx.float32).reshape(2, 5, 2)
    words_mask = mx.array([[0, 1, 0, 2, 0], [1, 0, 2, 3, 0]])
    attention = mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    lengths = mx.array([[2], [3]])

    pooled, mask = pool_first_subtokens(states, words_mask, attention, lengths)
    mx.eval(pooled, mask)

    expected = np.array(
        [
            [[2.0, 3.0], [6.0, 7.0], [0.0, 0.0]],
            [[10.0, 11.0], [14.0, 15.0], [16.0, 17.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(pooled), expected)
    np.testing.assert_array_equal(
        np.asarray(mask),
        np.array([[True, True, False], [True, True, True]]),
    )


def test_masked_and_out_of_range_markers_do_not_contribute() -> None:
    states = mx.array([[[1.0], [2.0], [3.0]]])
    pooled, mask = pool_first_subtokens(
        states,
        mx.array([[1, 2, 9]]),
        mx.array([[1, 0, 1]]),
        mx.array([1]),
    )
    mx.eval(pooled, mask)
    np.testing.assert_array_equal(np.asarray(pooled), [[[1.0]]])
    np.testing.assert_array_equal(np.asarray(mask), [[True]])


def test_pooling_rejects_inconsistent_shapes() -> None:
    with pytest.raises(WordPoolingError, match="words_mask"):
        pool_first_subtokens(
            mx.zeros((1, 2, 3)),
            mx.zeros((1, 3)),
            mx.ones((1, 2)),
            mx.array([2]),
        )
