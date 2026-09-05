from __future__ import annotations

import numpy as np
import pytest

from streamner_commit.mlx.decoder import (
    DecodedSpan,
    SpanDecoder,
    SpanDecoderError,
    sigmoid,
)


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_array_equal(values, [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.3, ((0, 0, "person"), (2, 2, "email"))),
        (0.5, ((0, 0, "person"), (2, 2, "email"))),
        (0.7, ((0, 0, "person"),)),
    ],
)
def test_explicit_span_decode_at_required_thresholds(
    threshold: float,
    expected: tuple[tuple[int, int, str], ...],
) -> None:
    decoder = SpanDecoder()
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.7],
        ]
    )
    logits = np.log(probabilities / (1.0 - probabilities))[None, :, :]
    spans = np.array([[[0, 0], [0, 1], [2, 2]]])
    decoded = decoder.decode(
        logits,
        spans,
        np.array([[True, True, True]]),
        ("person", "email"),
        threshold=threshold,
        flat_ner=True,
    )[0]

    assert tuple((item.start_word, item.end_word, item.label) for item in decoded) == expected


def test_threshold_comparison_is_strict() -> None:
    decoded = SpanDecoder().decode(
        np.array([[[0.0]]]),
        np.array([[[0, 0]]]),
        np.array([[True]]),
        ("person",),
        threshold=0.5,
    )
    assert decoded == ((),)


def test_greedy_search_is_stable_for_equal_scores() -> None:
    first = DecodedSpan(0, 1, "first", 0.8)
    second = DecodedSpan(0, 0, "second", 0.8)
    assert SpanDecoder().greedy_search((first, second)) == (first,)


def test_public_conversion_uses_half_open_character_ends() -> None:
    entity = SpanDecoder().to_public_entities(
        (DecodedSpan(0, 1, "person", 0.75),),
        "Ada Lovelace wrote",
        (0, 4, 13),
        (3, 12, 18),
    )[0]
    assert entity.to_dict() == {
        "start_char": 0,
        "end_char": 12,
        "label": "person",
        "text": "Ada Lovelace",
        "score": 0.75,
    }


def test_decoder_rejects_class_count_mismatch() -> None:
    with pytest.raises(SpanDecoderError, match="labels length"):
        SpanDecoder().decode(
            np.zeros((1, 1, 2)),
            np.array([[[0, 0]]]),
            np.array([[True]]),
            ("person",),
        )
