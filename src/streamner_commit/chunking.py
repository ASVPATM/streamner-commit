"""Whitespace-preserving chunking and word/character coordinate helpers.

Chunk units are maximal runs of non-whitespace characters, so punctuation stays
attached for the benchmark update schedule. Model-word coordinates separately
match GLiNER 0.2.28's ``WhitespaceTokenSplitter``, which tokenizes symbols.
"""

from __future__ import annotations

import re

_CHUNK_WORD_PATTERN = re.compile(r"\S+")
_MODEL_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def word_char_spans(text: str) -> list[tuple[int, int]]:
    """Return GLiNER model-word offsets using its pinned splitter pattern.

    The returned offsets refer directly to ``text``.  Therefore, for every
    ``(start, end)`` in the result, ``text[start:end]`` is exactly one model
    word. Symbols are separate model words even when attached to text.
    """

    return [(match.start(), match.end()) for match in _MODEL_WORD_PATTERN.finditer(text)]


def chunk_word_char_spans(text: str) -> list[tuple[int, int]]:
    """Return offsets for whitespace-delimited benchmark chunk units."""

    return [(match.start(), match.end()) for match in _CHUNK_WORD_PATTERN.finditer(text)]


def count_words(text: str) -> int:
    """Count whitespace-delimited benchmark chunk units.

    This compatibility name refers to update scheduling, not GLiNER's internal
    model-word coordinates. Use :func:`count_model_words` for span state.
    """

    return sum(1 for _ in _CHUNK_WORD_PATTERN.finditer(text))


def count_model_words(text: str) -> int:
    """Count model words exactly as the pinned GLiNER splitter does."""

    return sum(1 for _ in _MODEL_WORD_PATTERN.finditer(text))


def chunk_text_by_words(text: str, words_per_chunk: int) -> list[str]:
    """Split ``text`` into exact, word-aligned pieces.

    No character is inserted, removed, or normalized, so joining the returned
    chunks always reconstructs ``text``.  Separating whitespace is attached to
    the preceding chunk, while leading whitespace belongs to the first chunk.
    A nonempty whitespace-only input is returned as one chunk so reconstruction
    remains exact; downstream model code may skip that blank update.

    Any positive chunk size is accepted, including the benchmark sizes 1, 2, 4,
    and 8.  Empty input produces no chunks.
    """

    _validate_positive_int(words_per_chunk, name="words_per_chunk")

    matches = list(_CHUNK_WORD_PATTERN.finditer(text))
    if not matches:
        return [text] if text else []

    # Cut immediately before the next group's first word.  This keeps all
    # original whitespace and avoids producing a trailing whitespace-only chunk.
    boundaries = [0]
    boundaries.extend(
        matches[word_index].start()
        for word_index in range(words_per_chunk, len(matches), words_per_chunk)
    )
    boundaries.append(len(text))

    return [text[start:end] for start, end in zip(boundaries, boundaries[1:], strict=False)]


def word_span_to_char_offsets(text: str, start_word: int, end_word: int) -> tuple[int, int]:
    """Map an inclusive word span to half-open character offsets.

    ``start_word`` and ``end_word`` are zero-based and inclusive.  The returned
    character offsets exclude surrounding whitespace but preserve all original
    characters (including internal whitespace) between the two words.
    """

    _validate_word_indices(start_word, end_word)
    spans = word_char_spans(text)
    if end_word >= len(spans):
        raise ValueError(
            f"word span [{start_word}, {end_word}] is outside text with {len(spans)} words"
        )
    return spans[start_word][0], spans[end_word][1]


def char_offsets_to_word_span(text: str, start_char: int, end_char: int) -> tuple[int, int]:
    """Map exact half-open word-boundary offsets to inclusive word indices.

    The character offsets must start at the first character of a word and end
    immediately after a word.  Requiring exact boundaries prevents ambiguous
    mappings for offsets inside a word or in surrounding whitespace.
    """

    _validate_char_offsets(text, start_char, end_char)
    spans = word_char_spans(text)

    starts = {start: index for index, (start, _) in enumerate(spans)}
    ends = {end: index for index, (_, end) in enumerate(spans)}
    if start_char not in starts or end_char not in ends:
        raise ValueError("character offsets must coincide with exact word boundaries")

    start_word = starts[start_char]
    end_word = ends[end_char]
    if start_word > end_word:
        raise ValueError("character offsets do not describe a forward word span")
    return start_word, end_word


def extract_word_span(text: str, start_word: int, end_word: int) -> str:
    """Extract an inclusive word span without normalizing internal text."""

    start_char, end_char = word_span_to_char_offsets(text, start_word, end_word)
    return text[start_char:end_char]


def _validate_word_indices(start_word: int, end_word: int) -> None:
    for value, name in ((start_word, "start_word"), (end_word, "end_word")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if start_word < 0:
        raise ValueError("start_word must be nonnegative")
    if end_word < start_word:
        raise ValueError("end_word must be greater than or equal to start_word")


def _validate_char_offsets(text: str, start_char: int, end_char: int) -> None:
    for value, name in ((start_char, "start_char"), (end_char, "end_char")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if start_char < 0:
        raise ValueError("start_char must be nonnegative")
    if end_char <= start_char:
        raise ValueError("end_char must be greater than start_char")
    if end_char > len(text):
        raise ValueError("end_char must not exceed the text length")
