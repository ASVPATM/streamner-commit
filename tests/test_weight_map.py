from __future__ import annotations

import pytest

from streamner_commit.conversion.weight_map import (
    WeightMappingError,
    classify_reference_inventory,
    map_reference_key,
)


@pytest.mark.parametrize(
    ("reference_name", "component", "mlx_key"),
    [
        (
            "token_rep_layer.decoder_layer.model.embed_tokens.weight",
            "qwen",
            "qwen.embed_tokens.weight",
        ),
        (
            "token_rep_layer.decoder_layer.model.layers.27.self_attn.q_proj.weight",
            "qwen",
            "qwen.layers.27.self_attn.q_proj.weight",
        ),
        (
            "token_rep_layer.decoder_layer.model.layers.0.mlp.down_proj.weight",
            "qwen",
            "qwen.layers.0.mlp.down_proj.weight",
        ),
        (
            "labels_encoder.encoder.encoder.rel_embeddings.weight",
            "label_encoder",
            "label_encoder.rel_embeddings.weight",
        ),
        (
            "labels_encoder.encoder.encoder.layer.1.attention.self.pos_query_proj.bias",
            "label_encoder",
            "label_encoder.layers.1.attention.self_attn.pos_query_proj.bias",
        ),
        (
            "labels_encoder.encoder.encoder.layer.0.attention.output.LayerNorm.weight",
            "label_encoder",
            "label_encoder.layers.0.attention.output.layer_norm.weight",
        ),
        (
            "labels_encoder.encoder.encoder.layer.1.output.LayerNorm.bias",
            "label_encoder",
            "label_encoder.layers.1.output.layer_norm.bias",
        ),
        (
            "prompt_rep_layer.3.bias",
            "prompt_projection",
            "prompt_projection.layers.3.bias",
        ),
        (
            "span_rep_layer.span_rep_layer.project_context.0.weight",
            "marker_v2",
            "marker.project_context.layers.0.weight",
        ),
        (
            "span_rep_layer.span_rep_layer.out_project.3.bias",
            "marker_v2",
            "marker.out_project.layers.3.bias",
        ),
    ],
)
def test_map_reference_key(
    reference_name: str,
    component: str,
    mlx_key: str,
) -> None:
    mapping = map_reference_key(reference_name)
    assert mapping.reference_name == reference_name
    assert mapping.component == component
    assert mapping.mlx_key == mlx_key
    assert mapping.transform == "none"
    assert mapping.to_dict()["mlx_key"] == mlx_key


@pytest.mark.parametrize(
    "reference_name",
    [
        "",
        "model.token_rep_layer.decoder_layer.model.norm.weight",
        "token_rep_layer.decoder_layer.model.lm_head.weight",
        "token_rep_layer.decoder_layer.model.layers.0.self_attn.q_proj.bias",
        "labels_encoder.encoder.encoder.embeddings.word_embeddings.weight",
        "labels_encoder.encoder.encoder.layer.0.attention.self.unknown.weight",
        "prompt_rep_layer.1.weight",
        "span_rep_layer.context_encoder.weight",
        "span_rep_layer.span_rep_layer.project_start.2.weight",
        "token_projection.weight",
    ],
)
def test_map_reference_key_rejects_unclassified_tensors(reference_name: str) -> None:
    with pytest.raises(WeightMappingError):
        map_reference_key(reference_name)


def test_classify_inventory_is_sorted_and_counts_components() -> None:
    names = [
        "prompt_rep_layer.0.weight",
        "token_rep_layer.decoder_layer.model.norm.weight",
        "labels_encoder.encoder.encoder.rel_embeddings.weight",
        "span_rep_layer.span_rep_layer.project_end.3.bias",
    ]
    mappings, counts = classify_reference_inventory(reversed(names))
    assert [mapping.reference_name for mapping in mappings] == sorted(names)
    assert dict(counts) == {
        "label_encoder": 1,
        "marker_v2": 1,
        "prompt_projection": 1,
        "qwen": 1,
    }


def test_classify_inventory_rejects_duplicate_reference_names() -> None:
    name = "prompt_rep_layer.0.weight"
    with pytest.raises(WeightMappingError, match="duplicate reference"):
        classify_reference_inventory([name, name])
