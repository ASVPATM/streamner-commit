"""Pinned dense-input DeBERTa-v2 label context encoder for MLX.

This is deliberately not a general DeBERTa implementation.  It contains only the
inference path represented by the locked StreamingSpan checkpoint: two post-norm
layers, unbucketed c2p/p2c relative attention, and a shared relative embedding table.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from safetensors import safe_open

if TYPE_CHECKING:
    from streamner_commit.mlx.assets import AssetBundle


LABEL_ENCODER_COMPONENT = "label_encoder"
LABEL_ENCODER_PREFIX = "label_encoder."
LABEL_ENCODER_TENSOR_COUNT = 41


class LabelEncoderConfigurationError(ValueError):
    """The model config does not describe the locked label encoder."""


class LabelEncoderWeightError(ValueError):
    """The exported label-encoder weights are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class LabelEncoderConfig:
    """The exact architecture encoded by the pinned checkpoint."""

    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 2
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-7
    relative_attention: bool = True
    max_relative_positions: int = 512
    position_buckets: int = -1
    pos_att_type: tuple[str, ...] = ("p2c", "c2p")
    share_att_key: bool = False
    conv_kernel_size: int = 0
    norm_rel_ebd: str = "none"

    @property
    def attention_head_size(self) -> int:
        """Width of one attention head."""

        return self.hidden_size // self.num_attention_heads

    @property
    def relative_embedding_count(self) -> int:
        """Number of rows in the shared relative-position table."""

        return self.max_relative_positions * 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LabelEncoderConfig:
        """Validate checkpoint config values rather than inheriting library defaults."""

        if not isinstance(value, dict):
            raise LabelEncoderConfigurationError("labels_encoder_config must be an object")
        expected: dict[str, Any] = {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": 512,
            "pos_att_type": ["p2c", "c2p"],
        }
        for field, required in expected.items():
            actual = value.get(field)
            if actual != required:
                raise LabelEncoderConfigurationError(
                    f"labels_encoder_config.{field} must be {required!r}, got {actual!r}"
                )

        optional_defaults: dict[str, Any] = {
            "position_buckets": -1,
            "share_att_key": False,
            "conv_kernel_size": 0,
            "norm_rel_ebd": "none",
        }
        for field, required in optional_defaults.items():
            actual = value.get(field, required)
            if actual != required:
                raise LabelEncoderConfigurationError(
                    f"labels_encoder_config.{field} must be {required!r}, got {actual!r}"
                )
        return cls()


def build_relative_positions(query_length: int, key_length: int) -> mx.array:
    """Return unbucketed ``query_position - key_position`` indices."""

    if query_length < 0 or key_length < 0:
        raise ValueError("query_length and key_length must be nonnegative")
    query_positions = mx.arange(query_length, dtype=mx.int32)[:, None]
    key_positions = mx.arange(key_length, dtype=mx.int32)[None, :]
    return query_positions - key_positions


def _split_heads(value: mx.array, *, heads: int, head_size: int) -> mx.array:
    batch, length, width = value.shape
    if width != heads * head_size:
        raise ValueError("projected attention width does not match head configuration")
    return value.reshape(batch, length, heads, head_size).transpose(0, 2, 1, 3)


def _split_relative_heads(value: mx.array, *, heads: int, head_size: int) -> mx.array:
    length, width = value.shape
    if width != heads * head_size:
        raise ValueError("projected relative width does not match head configuration")
    return value.reshape(length, heads, head_size).transpose(1, 0, 2)


class DisentangledSelfAttention(nn.Module):
    """Checkpoint-specific DeBERTa-v2 c2p/p2c self-attention."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.pos_key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.pos_query_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _relative_bias(
        self,
        query: mx.array,
        key: mx.array,
        relative_positions: mx.array,
        relative_embeddings: mx.array,
    ) -> mx.array:
        config = self.config
        heads = config.num_attention_heads
        head_size = config.attention_head_size
        span = config.max_relative_positions
        scale = math.sqrt(head_size * 3)

        relative_embeddings = relative_embeddings[: span * 2]
        position_keys = _split_relative_heads(
            self.pos_key_proj(relative_embeddings),
            heads=heads,
            head_size=head_size,
        )
        position_queries = _split_relative_heads(
            self.pos_query_proj(relative_embeddings),
            heads=heads,
            head_size=head_size,
        )

        batch, _, query_length, _ = query.shape
        key_length = key.shape[-2]
        c2p_scores = mx.einsum("bhqd,hrd->bhqr", query, position_keys)
        c2p_indices = mx.clip(relative_positions + span, 0, span * 2 - 1)
        c2p_indices = mx.broadcast_to(
            c2p_indices[None, None, :, :],
            (batch, heads, query_length, key_length),
        )
        c2p_bias = mx.take_along_axis(c2p_scores, c2p_indices, axis=-1) / scale

        # The locked path is self-attention (query length equals key length).  This is
        # the precise HF p2c gather: gather on key rows, then transpose into q-by-k.
        if query_length != key_length:
            raise ValueError("the pinned label encoder supports self-attention only")
        p2c_scores = mx.einsum("bhkd,hrd->bhkr", key, position_queries)
        p2c_indices = mx.clip(-relative_positions + span, 0, span * 2 - 1)
        p2c_indices = mx.broadcast_to(
            p2c_indices[None, None, :, :],
            (batch, heads, key_length, key_length),
        )
        p2c_bias = mx.take_along_axis(p2c_scores, p2c_indices, axis=-1)
        p2c_bias = p2c_bias.swapaxes(-1, -2) / scale
        return c2p_bias + p2c_bias

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_positions: mx.array,
        relative_embeddings: mx.array,
    ) -> mx.array:
        config = self.config
        heads = config.num_attention_heads
        head_size = config.attention_head_size
        query = _split_heads(self.query_proj(hidden_states), heads=heads, head_size=head_size)
        key = _split_heads(self.key_proj(hidden_states), heads=heads, head_size=head_size)
        value = _split_heads(self.value_proj(hidden_states), heads=heads, head_size=head_size)

        scale = math.sqrt(head_size * 3)
        # HF divides K before bmm; retain that evaluation order for close FP32 parity.
        scores = query @ (key.swapaxes(-1, -2) / scale)
        scores = scores + self._relative_bias(
            query,
            key,
            relative_positions,
            relative_embeddings,
        )
        pair_mask = attention_mask.astype(mx.bool_)[:, None, :, None]
        pair_mask = pair_mask & attention_mask.astype(mx.bool_)[:, None, None, :]
        scores = mx.where(pair_mask, scores, mx.finfo(scores.dtype).min)
        probabilities = mx.softmax(scores, axis=-1)
        context = probabilities @ value
        batch, _, length, _ = context.shape
        return context.transpose(0, 2, 1, 3).reshape(batch, length, config.hidden_size)


class DebertaV2SelfOutput(nn.Module):
    """Attention projection followed by post-residual LayerNorm."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        return self.layer_norm(self.dense(hidden_states) + residual)


class DebertaV2Attention(nn.Module):
    """Disentangled attention and its post-norm output block."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.self_attn = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_positions: mx.array,
        relative_embeddings: mx.array,
    ) -> mx.array:
        attended = self.self_attn(
            hidden_states,
            attention_mask,
            relative_positions,
            relative_embeddings,
        )
        return self.output(attended, hidden_states)


class DebertaV2Intermediate(nn.Module):
    """Exact-GELU feed-forward expansion."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return nn.gelu(self.dense(hidden_states))


class DebertaV2Output(nn.Module):
    """Feed-forward contraction followed by post-residual LayerNorm."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        return self.layer_norm(self.dense(hidden_states) + residual)


class DebertaV2Layer(nn.Module):
    """One pinned post-norm encoder layer."""

    def __init__(self, config: LabelEncoderConfig) -> None:
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_positions: mx.array,
        relative_embeddings: mx.array,
    ) -> mx.array:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            relative_positions,
            relative_embeddings,
        )
        return self.output(self.intermediate(attention_output), attention_output)


class DenseDebertaV2Encoder(nn.Module):
    """Two-layer dense-input encoder with canonical checkpoint parameter names."""

    def __init__(self, config: LabelEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LabelEncoderConfig()
        self.rel_embeddings = nn.Embedding(
            self.config.relative_embedding_count,
            self.config.hidden_size,
        )
        self.layers = [DebertaV2Layer(self.config) for _ in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, 1024]")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states width must be {self.config.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match hidden_states batch and sequence axes")
        relative_positions = build_relative_positions(
            hidden_states.shape[1], hidden_states.shape[1]
        )
        output = hidden_states
        for layer in self.layers:
            output = layer(
                output,
                attention_mask,
                relative_positions,
                self.rel_embeddings.weight,
            )
        # The reference dense-context wrapper explicitly zeros padded rows.
        return output * attention_mask.astype(output.dtype)[..., None]


@dataclass(frozen=True, slots=True)
class NumericalDiagnostics:
    """Backend-neutral parity summary for one activation tensor."""

    shape_equal: bool
    max_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float


def numerical_diagnostics(actual: mx.array, expected: np.ndarray[Any, Any]) -> NumericalDiagnostics:
    """Evaluate an MLX array and summarize its difference from a NumPy oracle."""

    actual_numpy = np.asarray(actual)
    expected_numpy = np.asarray(expected)
    if actual_numpy.shape != expected_numpy.shape:
        return NumericalDiagnostics(False, math.inf, math.inf, float("nan"))
    actual_float = actual_numpy.astype(np.float64, copy=False).reshape(-1)
    expected_float = expected_numpy.astype(np.float64, copy=False).reshape(-1)
    difference = np.abs(actual_float - expected_float)
    denominator = float(np.linalg.norm(actual_float) * np.linalg.norm(expected_float))
    cosine = (
        float(np.dot(actual_float, expected_float) / denominator)
        if denominator > 0.0
        else float(actual_float.size == 0 or np.array_equal(actual_float, expected_float))
    )
    return NumericalDiagnostics(
        shape_equal=True,
        max_absolute_error=float(difference.max(initial=0.0)),
        mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
        cosine_similarity=cosine,
    )


def _sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_encoder_weights(
    encoder: DenseDebertaV2Encoder,
    bundle: AssetBundle,
) -> tuple[str, ...]:
    """Strictly load exactly the 41 safe, mapped label-encoder tensors."""

    records = tuple(
        record for record in bundle.tensors if record.component == LABEL_ENCODER_COMPONENT
    )
    if len(records) != LABEL_ENCODER_TENSOR_COUNT:
        raise LabelEncoderWeightError(
            f"expected {LABEL_ENCODER_TENSOR_COUNT} label-encoder tensors, got {len(records)}"
        )
    if any(record.transform != "none" for record in records):
        raise LabelEncoderWeightError("the pinned label encoder permits no weight transforms")

    local_parameters = tree_flatten(encoder.parameters(), destination={})
    expected_mlx_keys = {f"{LABEL_ENCODER_PREFIX}{key}" for key in local_parameters}
    manifest_mlx_keys = {record.mlx_key for record in records}
    if manifest_mlx_keys != expected_mlx_keys:
        missing = sorted(expected_mlx_keys - manifest_mlx_keys)
        extra = sorted(manifest_mlx_keys - expected_mlx_keys)
        raise LabelEncoderWeightError(
            f"label-encoder mapping mismatch; missing={missing}, extra={extra}"
        )
    if _sha256(bundle.weights_path) != bundle.weights_sha256:
        raise LabelEncoderWeightError("model.safetensors changed after asset validation")

    weights: list[tuple[str, mx.array]] = []
    with safe_open(bundle.weights_path, framework="numpy") as handle:
        available = set(handle.keys())
        for record in sorted(records, key=lambda item: item.mlx_key):
            if record.reference_name not in available:
                raise LabelEncoderWeightError(f"safe weights are missing {record.reference_name!r}")
            array = handle.get_tensor(record.reference_name)
            if tuple(array.shape) != record.shape or str(array.dtype) != record.dtype:
                raise LabelEncoderWeightError(
                    f"safe tensor metadata mismatch for {record.reference_name!r}"
                )
            local_key = record.mlx_key.removeprefix(LABEL_ENCODER_PREFIX)
            weights.append((local_key, mx.array(array)))

    try:
        encoder.load_weights(weights, strict=True)
    except ValueError as exc:
        raise LabelEncoderWeightError(f"cannot load label-encoder weights: {exc}") from exc
    mx.eval(encoder.parameters())
    return tuple(f"{LABEL_ENCODER_PREFIX}{key}" for key, _ in weights)


__all__ = [
    "LABEL_ENCODER_COMPONENT",
    "LABEL_ENCODER_PREFIX",
    "LABEL_ENCODER_TENSOR_COUNT",
    "DenseDebertaV2Encoder",
    "LabelEncoderConfig",
    "LabelEncoderConfigurationError",
    "LabelEncoderWeightError",
    "NumericalDiagnostics",
    "build_relative_positions",
    "load_label_encoder_weights",
    "numerical_diagnostics",
]
