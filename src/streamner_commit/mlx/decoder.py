"""Reference-equivalent flat/nested span decoding without PyTorch or GLiNER."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from streamner_commit.types import PublicEntity


class SpanDecoderError(ValueError):
    """Decoder inputs are inconsistent or contain invalid coordinates."""


@dataclass(frozen=True, slots=True)
class DecodedSpan:
    """One decoded candidate in inclusive word coordinates."""

    start_word: int
    end_word: int
    label: str
    score: float


def sigmoid(logits: ArrayLike) -> NDArray[np.float64]:
    """Stable elementwise sigmoid used before strict threshold comparison."""

    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _overlaps(left: DecodedSpan, right: DecodedSpan, *, multi_label: bool) -> bool:
    same_boundary = left.start_word == right.start_word and left.end_word == right.end_word
    if same_boundary:
        return not multi_label
    return not (left.start_word > right.end_word or right.start_word > left.end_word)


def _overlaps_nested(left: DecodedSpan, right: DecodedSpan, *, multi_label: bool) -> bool:
    same_boundary = left.start_word == right.start_word and left.end_word == right.end_word
    if same_boundary:
        return not multi_label
    disjoint = left.start_word > right.end_word or right.start_word > left.end_word
    nested = (left.start_word <= right.start_word and left.end_word >= right.end_word) or (
        right.start_word <= left.start_word and right.end_word >= left.end_word
    )
    return not (disjoint or nested)


class SpanDecoder:
    """Decode explicit candidate spans with GLiNER v0.2.28 ordering semantics."""

    def greedy_search(
        self,
        spans: Sequence[DecodedSpan],
        *,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> tuple[DecodedSpan, ...]:
        """Select candidates by descending score and then sort by start position."""

        selected: list[DecodedSpan] = []
        overlap = _overlaps if flat_ner else _overlaps_nested
        # Python's sort is stable, preserving the row-major candidate order for ties.
        for candidate in sorted(spans, key=lambda item: -item.score):
            if not any(overlap(candidate, prior, multi_label=multi_label) for prior in selected):
                selected.append(candidate)
        selected.sort(key=lambda item: item.start_word)
        return tuple(selected)

    def decode(
        self,
        logits: ArrayLike,
        span_idx: ArrayLike,
        span_mask: ArrayLike,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> tuple[tuple[DecodedSpan, ...], ...]:
        """Apply sigmoid, strict thresholding, validity masks, and greedy search."""

        score_array = np.asarray(logits)
        if score_array.ndim == 4:
            batch_size, starts, widths, class_count = score_array.shape
            score_array = score_array.reshape(batch_size, starts * widths, class_count)
        elif score_array.ndim == 3:
            batch_size, _, class_count = score_array.shape
        else:
            raise SpanDecoderError("logits must have rank 3 or 4")
        boundaries = np.asarray(span_idx, dtype=np.int64).reshape(batch_size, -1, 2)
        valid_spans = np.asarray(span_mask, dtype=np.bool_).reshape(batch_size, -1)
        if boundaries.shape[:2] != score_array.shape[:2]:
            raise SpanDecoderError("span_idx candidate count must match logits")
        if valid_spans.shape != score_array.shape[:2]:
            raise SpanDecoderError("span_mask candidate count must match logits")
        if len(labels) != class_count:
            raise SpanDecoderError("labels length must equal the logits class dimension")
        if any(not isinstance(label, str) or not label for label in labels):
            raise SpanDecoderError("labels must be nonempty strings")
        if not 0.0 <= threshold <= 1.0:
            raise SpanDecoderError("threshold must be between zero and one")

        probabilities = sigmoid(score_array)
        outputs: list[tuple[DecodedSpan, ...]] = []
        for batch_index in range(batch_size):
            mask = valid_spans[batch_index, :, None] & (probabilities[batch_index] > threshold)
            candidates: list[DecodedSpan] = []
            # np.argwhere is C-order: candidate position first, class second.
            for span_position, class_index in np.argwhere(mask):
                start_word, end_word = boundaries[batch_index, span_position]
                if start_word < 0 or end_word < start_word:
                    raise SpanDecoderError("valid span coordinates must be inclusive and ordered")
                candidates.append(
                    DecodedSpan(
                        start_word=int(start_word),
                        end_word=int(end_word),
                        label=labels[int(class_index)],
                        score=float(probabilities[batch_index, span_position, class_index]),
                    )
                )
            outputs.append(
                self.greedy_search(
                    candidates,
                    flat_ner=flat_ner,
                    multi_label=multi_label,
                )
            )
        return tuple(outputs)

    def to_public_entities(
        self,
        decoded: Sequence[DecodedSpan],
        text: str,
        word_starts: Sequence[int],
        word_ends: Sequence[int],
    ) -> tuple[PublicEntity, ...]:
        """Translate inclusive word boundaries to half-open character entities."""

        if len(word_starts) != len(word_ends):
            raise SpanDecoderError("word start/end arrays must have equal length")
        entities: list[PublicEntity] = []
        for span in decoded:
            if span.end_word >= len(word_starts):
                raise SpanDecoderError("decoded span extends beyond available word coordinates")
            start_char = int(word_starts[span.start_word])
            end_char = int(word_ends[span.end_word])
            entities.append(
                PublicEntity(
                    start_char=start_char,
                    end_char=end_char,
                    label=span.label,
                    text=text[start_char:end_char],
                    score=span.score,
                )
            )
        return tuple(entities)


__all__ = [
    "DecodedSpan",
    "SpanDecoder",
    "SpanDecoderError",
    "sigmoid",
]
