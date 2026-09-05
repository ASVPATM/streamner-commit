"""Prompt compaction and ordered label extraction for the cold MLX path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from streamner_commit.mlx.context_encoder import DenseDebertaV2Encoder


@dataclass(frozen=True, slots=True)
class PromptSlice:
    """Compacted prompt inputs, inclusive of the first valid separator."""

    hidden_states: mx.array
    input_ids: mx.array
    attention_mask: mx.array


@dataclass(frozen=True, slots=True)
class LabelEncoderOutput:
    """Intermediate and final values required by component parity tests."""

    prompt: PromptSlice
    contextualized_prompt: mx.array
    label_representations: mx.array
    label_mask: mx.array


def _require_batch_shapes(
    hidden_states: mx.array,
    input_ids: mx.array,
    attention_mask: mx.array,
) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden_size]")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
    if input_ids.shape != attention_mask.shape or input_ids.shape != hidden_states.shape[:2]:
        raise ValueError("hidden_states, input_ids, and attention_mask batch axes must match")


def slice_prompt(
    hidden_states: mx.array,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    separator_token_id: int,
) -> PromptSlice:
    """Compact each row from its first valid token through its first separator."""

    _require_batch_shapes(hidden_states, input_ids, attention_mask)
    if separator_token_id < 0:
        raise ValueError("separator_token_id must be nonnegative")
    valid = attention_mask > 0
    separator = (input_ids == separator_token_id) & valid
    has_separator = mx.any(separator, axis=1)
    if not bool(mx.all(has_separator).item()):
        missing = np_indices(~has_separator)
        raise ValueError(f"no separator token was found for batch rows {missing}")

    batch, sequence, width = hidden_states.shape
    positions = mx.broadcast_to(mx.arange(sequence, dtype=mx.int32)[None, :], (batch, sequence))
    first_valid = mx.min(mx.where(valid, positions, sequence), axis=1)
    first_separator = mx.min(mx.where(separator, positions, sequence), axis=1)
    prompt_lengths = first_separator - first_valid + 1
    if not bool(mx.all(prompt_lengths > 0).item()):
        invalid = np_indices(prompt_lengths <= 0)
        raise ValueError(f"separator token precedes the prompt for batch rows {invalid}")

    max_prompt_length = cast(int, mx.max(prompt_lengths).item())
    offsets = mx.arange(max_prompt_length, dtype=mx.int32)[None, :]
    source_positions = first_valid[:, None] + offsets
    within_prompt = offsets < prompt_lengths[:, None]
    safe_positions = mx.minimum(source_positions, sequence - 1)
    hidden_indices = mx.broadcast_to(safe_positions[..., None], (batch, max_prompt_length, width))
    prompt_hidden = mx.take_along_axis(hidden_states, hidden_indices, axis=1)
    prompt_ids = mx.take_along_axis(input_ids, safe_positions, axis=1)
    prompt_mask = mx.take_along_axis(attention_mask, safe_positions, axis=1)
    prompt_mask = mx.where(within_prompt, prompt_mask, 0)
    valid_prompt = within_prompt & (prompt_mask > 0)
    prompt_hidden = prompt_hidden * valid_prompt.astype(prompt_hidden.dtype)[..., None]
    prompt_ids = mx.where(valid_prompt, prompt_ids, 0)
    return PromptSlice(prompt_hidden, prompt_ids, prompt_mask)


def np_indices(mask: mx.array) -> list[int]:
    """Return one-dimensional true indices for deterministic validation messages."""

    values = cast(list[bool], mask.tolist())
    return [index for index, value in enumerate(values) if value]


def extract_label_representations(
    contextualized_prompt: mx.array,
    prompt_input_ids: mx.array,
    prompt_attention_mask: mx.array,
    *,
    label_token_id: int,
) -> tuple[mx.array, mx.array]:
    """Gather label-marker states in their stable left-to-right prompt order."""

    _require_batch_shapes(contextualized_prompt, prompt_input_ids, prompt_attention_mask)
    if label_token_id < 0:
        raise ValueError("label_token_id must be nonnegative")
    label_tokens = (prompt_input_ids == label_token_id) & (prompt_attention_mask > 0)
    counts = mx.sum(label_tokens, axis=1)
    max_labels = cast(int, mx.max(counts).item()) if counts.shape[0] else 0
    batch, prompt_length, width = contextualized_prompt.shape

    positions = mx.broadcast_to(
        mx.arange(prompt_length, dtype=mx.int32)[None, :],
        (batch, prompt_length),
    )
    ordered_positions = mx.sort(mx.where(label_tokens, positions, prompt_length), axis=1)
    ordered_positions = ordered_positions[:, :max_labels]
    safe_positions = mx.minimum(ordered_positions, max(prompt_length - 1, 0))
    indices = mx.broadcast_to(safe_positions[..., None], (batch, max_labels, width))
    representations = mx.take_along_axis(contextualized_prompt, indices, axis=1)
    label_mask = mx.arange(max_labels, dtype=prompt_attention_mask.dtype)[None, :] < counts[:, None]
    label_mask = label_mask.astype(prompt_attention_mask.dtype)
    representations = representations * label_mask.astype(representations.dtype)[..., None]
    return representations, label_mask


def encode_labels(
    encoder: DenseDebertaV2Encoder,
    hidden_states: mx.array,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    separator_token_id: int,
    label_token_id: int,
) -> LabelEncoderOutput:
    """Run exact prompt slicing, dense context encoding, and label extraction."""

    prompt = slice_prompt(
        hidden_states,
        input_ids,
        attention_mask,
        separator_token_id=separator_token_id,
    )
    contextualized = encoder(prompt.hidden_states, prompt.attention_mask)
    labels, label_mask = extract_label_representations(
        contextualized,
        prompt.input_ids,
        prompt.attention_mask,
        label_token_id=label_token_id,
    )
    return LabelEncoderOutput(prompt, contextualized, labels, label_mask)


__all__ = [
    "LabelEncoderOutput",
    "PromptSlice",
    "encode_labels",
    "extract_label_representations",
    "slice_prompt",
]
