from __future__ import annotations

import re

import pytest

from streamner_commit.chunking import (
    char_offsets_to_word_span,
    chunk_text_by_words,
    chunk_word_char_spans,
    count_model_words,
    count_words,
    extract_word_span,
    word_char_spans,
    word_span_to_char_offsets,
)

NORMAL_TEXT_CASES = [
    "one two three four five",
    "one   two    three",
    "one\ttwo\t\tthree",
    "one\ntwo\r\nthree\rfour",
    "Hello, world! (again).",
    "   leading whitespace stays",
    "trailing whitespace stays \t\n",
    "  mixed\t whitespace\nremains  byte-for-byte \r\n",
    "naïve café 東京 مرحبا 👩🏽‍💻",
    "singleton",
]


@pytest.mark.parametrize("text", NORMAL_TEXT_CASES)
@pytest.mark.parametrize("words_per_chunk", [1, 2, 4, 8])
def test_chunking_reconstructs_and_preserves_word_boundaries(
    text: str, words_per_chunk: int
) -> None:
    chunks = chunk_text_by_words(text, words_per_chunk)

    assert "".join(chunks) == text
    assert "".join(chunks).encode() == text.encode()
    assert len(chunks) == (count_words(text) + words_per_chunk - 1) // words_per_chunk
    assert sum(count_words(chunk) for chunk in chunks) == count_words(text)
    assert [word for chunk in chunks for word in re.findall(r"\S+", chunk)] == re.findall(
        r"\S+", text
    )
    assert all(1 <= count_words(chunk) <= words_per_chunk for chunk in chunks)
    assert all(count_words(chunk) == words_per_chunk for chunk in chunks[:-1])


@pytest.mark.parametrize(
    ("text", "words_per_chunk", "expected"),
    [
        ("one two three", 1, ["one ", "two ", "three"]),
        ("one  two\tthree", 2, ["one  two\t", "three"]),
        ("  one two  ", 1, ["  one ", "two  "]),
        ("punctuation,stays attached!", 1, ["punctuation,stays ", "attached!"]),
        ("one", 8, ["one"]),
    ],
)
def test_chunk_boundary_assignment_is_deterministic(
    text: str, words_per_chunk: int, expected: list[str]
) -> None:
    assert chunk_text_by_words(text, words_per_chunk) == expected


@pytest.mark.parametrize(("text", "expected"), [("", []), (" ", [" "]), ("\t\r\n  ", ["\t\r\n  "])])
@pytest.mark.parametrize("words_per_chunk", [1, 2, 4, 8])
def test_empty_and_whitespace_only_inputs_reconstruct(
    text: str, expected: list[str], words_per_chunk: int
) -> None:
    chunks = chunk_text_by_words(text, words_per_chunk)

    assert chunks == expected
    assert "".join(chunks) == text
    assert sum(count_words(chunk) for chunk in chunks) == 0


@pytest.mark.parametrize("words_per_chunk", [0, -1, -8])
def test_chunk_size_must_be_positive(words_per_chunk: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_text_by_words("text", words_per_chunk)


@pytest.mark.parametrize("words_per_chunk", [True, 1.5, "2", None])
def test_chunk_size_must_be_an_integer(words_per_chunk: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        chunk_text_by_words("text", words_per_chunk)  # type: ignore[arg-type]


def test_word_char_spans_use_python_half_open_offsets() -> None:
    text = " \tJosé,\n東京  +1 test-token  "
    spans = word_char_spans(text)

    assert [text[start:end] for start, end in spans] == [
        "José",
        ",",
        "東京",
        "+",
        "1",
        "test-token",
    ]
    assert count_words(text) == 4
    assert count_model_words(text) == 6
    assert all(0 <= start < end <= len(text) for start, end in spans)


def test_chunk_units_and_model_words_are_explicitly_distinct() -> None:
    text = "Call +1 (202) 555-0147."

    assert [text[start:end] for start, end in chunk_word_char_spans(text)] == [
        "Call",
        "+1",
        "(202)",
        "555-0147.",
    ]
    assert [text[start:end] for start, end in word_char_spans(text)] == [
        "Call",
        "+",
        "1",
        "(",
        "202",
        ")",
        "555-0147",
        ".",
    ]


@pytest.mark.parametrize("text", ["", " ", "\t\r\n"])
def test_text_without_words_has_no_word_spans(text: str) -> None:
    assert word_char_spans(text) == []
    assert count_words(text) == 0


def test_word_span_offsets_and_extraction_preserve_internal_whitespace() -> None:
    text = "  alpha\t beta\n\ngamma!  "

    assert word_span_to_char_offsets(text, 0, 0) == (2, 7)
    assert word_span_to_char_offsets(text, 1, 2) == (9, 20)
    assert extract_word_span(text, 1, 2) == "beta\n\ngamma"

    start_char, end_char = word_span_to_char_offsets(text, 0, 3)
    span_text = extract_word_span(text, 0, 3)
    assert 0 <= 0 <= 3 < count_model_words(text)
    assert 0 <= start_char < end_char <= len(text)
    assert text[start_char:end_char] == span_text == "alpha\t beta\n\ngamma!"


def test_word_and_character_coordinates_round_trip_exhaustively() -> None:
    text = "  one\ttwo  three\nfour!  "
    num_words = count_model_words(text)

    for start_word in range(num_words):
        for end_word in range(start_word, num_words):
            start_char, end_char = word_span_to_char_offsets(text, start_word, end_word)
            assert char_offsets_to_word_span(text, start_char, end_char) == (
                start_word,
                end_word,
            )
            assert text[start_char:end_char] == extract_word_span(text, start_word, end_word)


@pytest.mark.parametrize(
    ("start_word", "end_word", "exception"),
    [
        (-1, 0, ValueError),
        (1, 0, ValueError),
        (0, 3, ValueError),
        (0, 0, ValueError),
        (True, 0, TypeError),
        (0, False, TypeError),
        (0.0, 0, TypeError),
    ],
)
def test_invalid_word_spans_are_rejected(
    start_word: object, end_word: object, exception: type[Exception]
) -> None:
    text = "one two three" if (start_word, end_word) != (0, 0) else ""
    with pytest.raises(exception):
        word_span_to_char_offsets(text, start_word, end_word)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start_char", "end_char", "exception"),
    [
        (-1, 3, ValueError),
        (0, 0, ValueError),
        (3, 2, ValueError),
        (0, 99, ValueError),
        (True, 3, TypeError),
        (0, False, TypeError),
        (0.0, 3, TypeError),
    ],
)
def test_invalid_character_ranges_are_rejected(
    start_char: object, end_char: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        char_offsets_to_word_span("one two", start_char, end_char)  # type: ignore[arg-type]


@pytest.mark.parametrize("start_char,end_char", [(1, 3), (0, 2), (0, 4), (3, 7)])
def test_ambiguous_character_offsets_are_rejected(start_char: int, end_char: int) -> None:
    with pytest.raises(ValueError, match="exact word boundaries"):
        char_offsets_to_word_span("one two", start_char, end_char)
