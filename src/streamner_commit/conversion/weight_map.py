"""Exhaustive mapping for the locked StreamingSpan checkpoint tensors.

The mapping is intentionally architecture-specific.  A future checkpoint with
an additional head, projection, or buffer must fail classification rather than
silently losing a tensor during export.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

QWEN_PREFIX = "token_rep_layer.decoder_layer.model."
LABEL_PREFIX = "labels_encoder.encoder.encoder."
MARKER_PREFIX = "span_rep_layer.span_rep_layer."
PROMPT_PREFIX = "prompt_rep_layer."
COMPONENTS = frozenset({"qwen", "label_encoder", "marker_v2", "prompt_projection"})
TRANSFORMS = frozenset({"none"})

_QWEN_DIRECT = {
    "embed_tokens.weight",
    "norm.weight",
}
_QWEN_LAYER = re.compile(
    r"layers\.(?P<layer>\d+)\."
    r"(?:(?:self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj|q_norm|k_norm)\.weight)"
    r"|(?:mlp\.(?:gate_proj|up_proj|down_proj)\.weight)"
    r"|(?:(?:input_layernorm|post_attention_layernorm)\.weight))\Z"
)
_LABEL_LAYER = re.compile(
    r"layer\.(?P<layer>\d+)\."
    r"(?P<body>"
    r"attention\.self\."
    r"(?:query_proj|key_proj|value_proj|pos_key_proj|pos_query_proj)\."
    r"(?:weight|bias)"
    r"|attention\.output\.(?:dense|LayerNorm)\.(?:weight|bias)"
    r"|intermediate\.dense\.(?:weight|bias)"
    r"|output\.(?:dense|LayerNorm)\.(?:weight|bias)"
    r")\Z"
)
_PROJECTION = re.compile(r"(?P<layer>0|3)\.(?P<parameter>weight|bias)\Z")
_MARKER = re.compile(
    r"(?P<projection>project_start|project_end|project_context|out_project)\."
    r"(?P<layer>0|3)\.(?P<parameter>weight|bias)\Z"
)


class WeightMappingError(ValueError):
    """Raised when a checkpoint key cannot be mapped exactly once."""


@dataclass(frozen=True, slots=True)
class TensorMapping:
    """One reference tensor's canonical MLX destination."""

    reference_name: str
    component: str
    mlx_key: str
    transform: str = "none"

    def __post_init__(self) -> None:
        for field_name in ("reference_name", "component", "mlx_key", "transform"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.transform != "none":
            raise ValueError("the locked checkpoint requires no tensor transforms")

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible mapping row."""

        return {
            "reference_name": self.reference_name,
            "component": self.component,
            "mlx_key": self.mlx_key,
            "transform": self.transform,
        }


def _mapping(reference_name: str, component: str, mlx_key: str) -> TensorMapping:
    return TensorMapping(
        reference_name=reference_name,
        component=component,
        mlx_key=mlx_key,
    )


def _map_qwen(reference_name: str, suffix: str) -> TensorMapping:
    if suffix not in _QWEN_DIRECT and _QWEN_LAYER.fullmatch(suffix) is None:
        raise WeightMappingError(f"unrecognized Qwen tensor: {reference_name}")
    return _mapping(reference_name, "qwen", f"qwen.{suffix}")


def _map_label_encoder(reference_name: str, suffix: str) -> TensorMapping:
    if suffix == "rel_embeddings.weight":
        return _mapping(
            reference_name,
            "label_encoder",
            "label_encoder.rel_embeddings.weight",
        )

    match = _LABEL_LAYER.fullmatch(suffix)
    if match is None:
        raise WeightMappingError(f"unrecognized label-encoder tensor: {reference_name}")
    body = match.group("body")
    body = body.replace("attention.self.", "attention.self_attn.")
    body = body.replace(".LayerNorm.", ".layer_norm.")
    mlx_key = f"label_encoder.layers.{match.group('layer')}.{body}"
    return _mapping(reference_name, "label_encoder", mlx_key)


def _map_prompt_projection(reference_name: str, suffix: str) -> TensorMapping:
    match = _PROJECTION.fullmatch(suffix)
    if match is None:
        raise WeightMappingError(f"unrecognized prompt-projection tensor: {reference_name}")
    mlx_key = f"prompt_projection.layers.{match.group('layer')}.{match.group('parameter')}"
    return _mapping(reference_name, "prompt_projection", mlx_key)


def _map_marker(reference_name: str, suffix: str) -> TensorMapping:
    match = _MARKER.fullmatch(suffix)
    if match is None:
        raise WeightMappingError(f"unrecognized markerV2 tensor: {reference_name}")
    mlx_key = (
        f"marker.{match.group('projection')}.layers.{match.group('layer')}."
        f"{match.group('parameter')}"
    )
    return _mapping(reference_name, "marker_v2", mlx_key)


def map_reference_key(reference_name: str) -> TensorMapping:
    """Map one exact reference key or fail closed.

    No generic prefix stripping is allowed: each component also validates its
    complete suffix grammar, catching newly introduced weights and misspellings.
    """

    if not isinstance(reference_name, str) or not reference_name:
        raise WeightMappingError("reference tensor name must be a non-empty string")
    if reference_name.startswith(QWEN_PREFIX):
        return _map_qwen(reference_name, reference_name.removeprefix(QWEN_PREFIX))
    if reference_name.startswith(LABEL_PREFIX):
        return _map_label_encoder(
            reference_name,
            reference_name.removeprefix(LABEL_PREFIX),
        )
    if reference_name.startswith(MARKER_PREFIX):
        return _map_marker(reference_name, reference_name.removeprefix(MARKER_PREFIX))
    if reference_name.startswith(PROMPT_PREFIX):
        return _map_prompt_projection(
            reference_name,
            reference_name.removeprefix(PROMPT_PREFIX),
        )
    raise WeightMappingError(f"unclassified reference tensor: {reference_name}")


def classify_reference_inventory(
    reference_names: Iterable[str],
) -> tuple[tuple[TensorMapping, ...], MappingProxyType[str, int]]:
    """Classify an inventory and reject duplicate source or destination keys."""

    mappings: list[TensorMapping] = []
    source_names: set[str] = set()
    destination_names: set[str] = set()
    component_counts: Counter[str] = Counter()
    for reference_name in reference_names:
        if reference_name in source_names:
            raise WeightMappingError(f"duplicate reference tensor: {reference_name}")
        mapping = map_reference_key(reference_name)
        if mapping.mlx_key in destination_names:
            raise WeightMappingError(f"duplicate MLX destination: {mapping.mlx_key}")
        source_names.add(reference_name)
        destination_names.add(mapping.mlx_key)
        component_counts[mapping.component] += 1
        mappings.append(mapping)

    mappings.sort(key=lambda item: item.reference_name)
    return tuple(mappings), MappingProxyType(dict(sorted(component_counts.items())))
