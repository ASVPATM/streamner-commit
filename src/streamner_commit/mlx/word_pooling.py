"""Exact first-subtoken-to-word pooling for the locked StreamingSpan model."""

from __future__ import annotations

import mlx.core as mx


class WordPoolingError(ValueError):
    """Inputs violate the locked first-subtoken pooling contract."""


def pool_first_subtokens(
    token_states: mx.array,
    words_mask: mx.array,
    attention_mask: mx.array,
    text_lengths: mx.array,
) -> tuple[mx.array, mx.array]:
    """Select marked token states into 1-based word positions.

    ``words_mask`` follows GLiNER's ``first`` pooling convention: zero means the
    token is not selected, while ``n > 0`` maps that token to word ``n - 1``.
    The returned mask is determined by ``text_lengths``, matching the reference
    implementation rather than inferred nonzero output values.
    """

    if token_states.ndim != 3:
        raise WordPoolingError("token_states must have shape (batch, tokens, hidden)")
    batch_size, token_count, hidden_size = token_states.shape
    if words_mask.shape != (batch_size, token_count):
        raise WordPoolingError("words_mask must match the first two token-state dimensions")
    if attention_mask.shape != (batch_size, token_count):
        raise WordPoolingError("attention_mask must match words_mask")
    if text_lengths.size != batch_size:
        raise WordPoolingError("text_lengths must contain one length per batch row")

    lengths = text_lengths.reshape(batch_size, -1)[:, 0].astype(mx.int32)
    if batch_size == 0:
        raise WordPoolingError("empty batches are not supported")
    maximum = mx.max(lengths).item()
    if not isinstance(maximum, int):
        raise WordPoolingError("text lengths must be integers")
    max_text_length = maximum
    if max_text_length < 0:
        raise WordPoolingError("text lengths must be nonnegative")

    marked = (words_mask > 0) & (attention_mask > 0)
    in_range = words_mask <= max_text_length
    valid = marked & in_range
    target_words = mx.maximum(words_mask.astype(mx.int32) - 1, 0)
    batch_indices = mx.broadcast_to(
        mx.arange(batch_size, dtype=mx.int32)[:, None],
        (batch_size, token_count),
    )
    selected = token_states * valid[..., None]
    pooled = mx.zeros((batch_size, max_text_length, hidden_size), dtype=token_states.dtype)
    pooled = pooled.at[batch_indices.reshape(-1), target_words.reshape(-1)].add(
        selected.reshape(-1, hidden_size)
    )

    word_positions = mx.arange(max_text_length, dtype=lengths.dtype)[None, :]
    word_mask = word_positions < lengths[:, None]
    return pooled, word_mask


__all__ = ["WordPoolingError", "pool_first_subtokens"]
