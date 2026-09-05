"""Native MLX MarkerV2 span representations for the locked checkpoint."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpanRepresentationError(ValueError):
    """Span-representation inputs violate the MarkerV2 contract."""


class ProjectionMLP(nn.Module):
    """The checkpoint's Linear-ReLU-Dropout-Linear projection block."""

    def __init__(self, input_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(input_size, output_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size * 4, output_size),
        ]

    def __call__(self, inputs: mx.array) -> mx.array:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


def _gather_sequence(sequence: mx.array, indices: mx.array) -> mx.array:
    if sequence.ndim != 3 or indices.ndim != 2:
        raise SpanRepresentationError("sequence and indices must be rank 3 and rank 2")
    batch_size, _, hidden_size = sequence.shape
    if indices.shape[0] != batch_size:
        raise SpanRepresentationError("indices batch dimension must match sequence")
    expanded = mx.broadcast_to(indices[..., None], (*indices.shape, hidden_size))
    return mx.take_along_axis(sequence, expanded, axis=1)


class MarkerV2(nn.Module):
    """Project inclusive span endpoints plus the latest valid word context."""

    def __init__(self, hidden_size: int = 1024, max_width: int = 12, dropout: float = 0.3) -> None:
        super().__init__()
        if hidden_size <= 0 or max_width <= 0:
            raise ValueError("hidden_size and max_width must be positive")
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.project_start = ProjectionMLP(hidden_size, hidden_size, dropout)
        self.project_end = ProjectionMLP(hidden_size, hidden_size, dropout)
        self.project_context = ProjectionMLP(hidden_size, hidden_size, dropout)
        self.out_project = ProjectionMLP(hidden_size * 3, hidden_size, dropout)

    def __call__(
        self,
        word_states: mx.array,
        span_idx: mx.array,
        word_mask: mx.array | None = None,
    ) -> mx.array:
        if word_states.ndim != 3:
            raise SpanRepresentationError("word_states must have shape (batch, words, hidden)")
        batch_size, word_count, hidden_size = word_states.shape
        if hidden_size != self.hidden_size:
            raise SpanRepresentationError(
                f"expected hidden size {self.hidden_size}, got {hidden_size}"
            )
        if span_idx.ndim != 3 or span_idx.shape[0] != batch_size or span_idx.shape[2] != 2:
            raise SpanRepresentationError("span_idx must have shape (batch, spans, 2)")
        if span_idx.shape[1] % self.max_width:
            raise SpanRepresentationError("span count must be divisible by max_width")
        if word_count == 0:
            raise SpanRepresentationError("MarkerV2 requires at least one word slot")
        if word_mask is None:
            word_mask = mx.ones((batch_size, word_count), dtype=mx.bool_)
        elif word_mask.shape != (batch_size, word_count):
            raise SpanRepresentationError("word_mask must match word-state positions")
        word_mask = word_mask.astype(mx.bool_)

        indices = span_idx.astype(mx.int32)
        starts = _gather_sequence(self.project_start(word_states), indices[..., 0])
        ends = _gather_sequence(self.project_end(word_states), indices[..., 1])

        valid_counts = mx.sum(word_mask, axis=1).astype(mx.int32)
        last_positions = mx.maximum(valid_counts - 1, 0)[:, None]
        latest = _gather_sequence(self.project_context(word_states), last_positions)[:, 0]
        latest = latest * (valid_counts > 0)[:, None]
        context = mx.broadcast_to(latest[:, None, :], starts.shape)

        features = mx.maximum(mx.concatenate([starts, ends, context], axis=-1), 0)
        span_states = self.out_project(features)
        num_starts = span_idx.shape[1] // self.max_width
        return span_states.reshape(batch_size, num_starts, self.max_width, hidden_size)


def score_spans(span_states: mx.array, label_states: mx.array) -> mx.array:
    """Compute the checkpoint's span-by-label dot-product logits."""

    if span_states.ndim != 4:
        raise SpanRepresentationError("span_states must have shape (batch, starts, widths, hidden)")
    if label_states.ndim != 3:
        raise SpanRepresentationError("label_states must have shape (batch, labels, hidden)")
    if span_states.shape[0] != label_states.shape[0]:
        raise SpanRepresentationError("span and label batches must match")
    if span_states.shape[-1] != label_states.shape[-1]:
        raise SpanRepresentationError("span and label hidden dimensions must match")
    return mx.einsum("blkd,bcd->blkc", span_states, label_states)


__all__ = [
    "MarkerV2",
    "ProjectionMLP",
    "SpanRepresentationError",
    "score_spans",
]
