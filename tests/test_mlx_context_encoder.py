from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.context_encoder import (
    LABEL_ENCODER_PREFIX,
    LABEL_ENCODER_TENSOR_COUNT,
    DenseDebertaV2Encoder,
    LabelEncoderConfig,
    LabelEncoderConfigurationError,
    build_relative_positions,
    load_label_encoder_weights,
    numerical_diagnostics,
)
from streamner_commit.mlx.labels_encoder import encode_labels

REFERENCE_ROOT = Path("artifacts/reference") / REFERENCE_REVISION


def _locked_config() -> dict[str, Any]:
    return {
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


def test_locked_configuration_and_relative_position_contract() -> None:
    config = LabelEncoderConfig.from_dict(_locked_config())

    assert config.attention_head_size == 64
    assert config.relative_embedding_count == 1024
    np.testing.assert_array_equal(
        np.asarray(build_relative_positions(3, 4)),
        np.array([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]], dtype=np.int32),
    )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("num_hidden_layers", 12),
        ("max_relative_positions", 128),
        ("pos_att_type", ["c2p"]),
        ("hidden_act", "gelu_new"),
    ],
)
def test_generic_deberta_defaults_are_rejected(field: str, wrong: object) -> None:
    value = _locked_config()
    value[field] = wrong

    with pytest.raises(LabelEncoderConfigurationError, match=field):
        LabelEncoderConfig.from_dict(value)


def test_model_has_exact_canonical_parameter_inventory() -> None:
    encoder = DenseDebertaV2Encoder()
    parameters = tree_flatten(encoder.parameters(), destination={})

    assert len(parameters) == LABEL_ENCODER_TENSOR_COUNT
    assert "rel_embeddings.weight" in parameters
    assert "layers.0.attention.self_attn.pos_key_proj.weight" in parameters
    assert "layers.1.attention.self_attn.pos_query_proj.bias" in parameters
    assert "layers.1.output.layer_norm.bias" in parameters
    assert not any("embedding" in key for key in parameters if key != "rel_embeddings.weight")
    assert not any("conv" in key for key in parameters)


def test_encoder_rejects_incompatible_dense_inputs() -> None:
    encoder = DenseDebertaV2Encoder()

    with pytest.raises(ValueError, match="width must be 1024"):
        encoder(mx.zeros((1, 2, 12)), mx.ones((1, 2)))
    with pytest.raises(ValueError, match="attention_mask"):
        encoder(mx.zeros((1, 2, 1024)), mx.ones((1, 3)))


@pytest.mark.skipif(not REFERENCE_ROOT.is_dir(), reason="ignored reference assets not exported")
def test_real_checkpoint_label_context_parity() -> None:
    bundle = load_asset_bundle(
        REFERENCE_ROOT,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    config = LabelEncoderConfig.from_dict(bundle.config["labels_encoder_config"])
    encoder = DenseDebertaV2Encoder(config)

    loaded_keys = load_label_encoder_weights(encoder, bundle)

    expected_keys = {
        record.mlx_key for record in bundle.tensors if record.component == "label_encoder"
    }
    assert len(loaded_keys) == LABEL_ENCODER_TENSOR_COUNT
    assert set(loaded_keys) == expected_keys
    assert all(key.startswith(LABEL_ENCODER_PREFIX) for key in loaded_keys)

    with np.load(REFERENCE_ROOT / "parity" / "parity_arrays.npz") as fixture:
        metadata = json.loads(
            (REFERENCE_ROOT / "parity" / "parity_metadata.json").read_text(encoding="utf-8")
        )
        special_tokens = metadata["special_tokens"]
        output = encode_labels(
            encoder,
            mx.array(fixture["label_encoder_input_hidden_states"]),
            mx.array(fixture["input_ids"]),
            mx.array(fixture["label_attention_mask"]),
            separator_token_id=special_tokens["separator_token_id"],
            label_token_id=special_tokens["label_token_id"],
        )
        mx.eval(output.contextualized_prompt, output.label_representations)

        np.testing.assert_array_equal(output.prompt.input_ids, fixture["prompt_input_ids"])
        np.testing.assert_array_equal(
            output.prompt.attention_mask,
            fixture["prompt_attention_mask"],
        )
        np.testing.assert_array_equal(output.label_mask, fixture["label_mask"])
        prompt = numerical_diagnostics(
            output.prompt.hidden_states,
            fixture["prompt_input_hidden_states"],
        )
        context = numerical_diagnostics(
            output.contextualized_prompt,
            fixture["contextualized_prompt_hidden_states"],
        )
        labels = numerical_diagnostics(
            output.label_representations,
            fixture["label_representations_pre_projection"],
        )

    assert prompt.shape_equal
    assert prompt.max_absolute_error == 0.0
    assert context.shape_equal
    assert context.max_absolute_error < 0.03
    assert context.mean_absolute_error < 0.001
    assert context.cosine_similarity > 0.999999
    assert labels.shape_equal
    assert labels.max_absolute_error < 0.01
    assert labels.mean_absolute_error < 0.001
    assert labels.cosine_similarity > 0.999999
