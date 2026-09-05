from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    AssetBundle,
    TensorRecord,
    load_asset_bundle,
)
from streamner_commit.mlx.qwen_adapter import (
    QWEN_TENSOR_COUNT,
    NumericalDiagnostics,
    QwenAdapter,
    QwenAdapterError,
    QwenConfiguration,
    numerical_diagnostics,
    qwen_weight_spec,
)

ASSET_ROOT = Path("artifacts/reference") / REFERENCE_REVISION
PARITY_ARRAYS = ASSET_ROOT / "parity" / "parity_arrays.npz"
RUN_REAL_PARITY = os.environ.get("STREAMNER_RUN_MLX_PARITY") == "1"


def _decoder_config() -> dict[str, Any]:
    return {
        "decoder_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 40_960,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "vocab_size": 151_671,
        }
    }


def _qwen_records() -> tuple[TensorRecord, ...]:
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (151_671, 1024),
        "norm.weight": (1024,),
    }
    for layer in range(28):
        prefix = f"layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (1024,),
                f"{prefix}.post_attention_layernorm.weight": (1024,),
                f"{prefix}.mlp.gate_proj.weight": (3072, 1024),
                f"{prefix}.mlp.up_proj.weight": (3072, 1024),
                f"{prefix}.mlp.down_proj.weight": (1024, 3072),
                f"{prefix}.self_attn.q_proj.weight": (2048, 1024),
                f"{prefix}.self_attn.k_proj.weight": (1024, 1024),
                f"{prefix}.self_attn.v_proj.weight": (1024, 1024),
                f"{prefix}.self_attn.o_proj.weight": (1024, 2048),
                f"{prefix}.self_attn.q_norm.weight": (128,),
                f"{prefix}.self_attn.k_norm.weight": (128,),
            }
        )
    assert len(shapes) == QWEN_TENSOR_COUNT
    return tuple(
        TensorRecord(
            reference_name=f"token_rep_layer.decoder_layer.model.{bare_key}",
            mlx_key=f"qwen.{bare_key}",
            component="qwen",
            transform="none",
            shape=shape,
            dtype="float32",
            numel=int(np.prod(shape)),
        )
        for bare_key, shape in sorted(shapes.items())
    )


def _bundle(records: tuple[TensorRecord, ...] | None = None) -> AssetBundle:
    return AssetBundle(
        root=Path("/nonexistent"),
        model_id=REFERENCE_MODEL_ID,
        revision=REFERENCE_REVISION,
        weights_path=Path("/nonexistent/model.safetensors"),
        config_path=Path("/nonexistent/config.json"),
        tensor_manifest_path=Path("/nonexistent/tensor_manifest.json"),
        tokenizer_files=(),
        tokenizer_special_tokens={},
        tensors=records if records is not None else _qwen_records(),
        tensor_count=QWEN_TENSOR_COUNT,
        parameter_count=sum(record.numel for record in (records or _qwen_records())),
        weights_sha256="0" * 64,
        config=_decoder_config(),
        export_manifest={},
        tensor_manifest={},
    )


def test_configuration_reads_explicit_head_width_and_nested_rope_theta() -> None:
    configuration = QwenConfiguration.from_model_config(_decoder_config())

    assert configuration.head_dim == 128
    assert configuration.num_attention_heads * configuration.head_dim == 2048
    assert configuration.rope_theta == 1_000_000.0
    assert configuration.vocab_size == 151_671


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("decoder_config", "head_dim"), 64, "head_dim"),
        (("decoder_config", "vocab_size"), 151_936, "vocab_size"),
        (
            ("decoder_config", "rope_parameters", "rope_theta"),
            10_000,
            "rope_theta",
        ),
        (
            ("decoder_config", "rope_parameters", "rope_type"),
            "linear",
            "rope_type",
        ),
        (("decoder_config", "attention_bias"), True, "attention_bias"),
    ],
)
def test_configuration_fails_closed_on_architecture_drift(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    config = copy.deepcopy(_decoder_config())
    target: dict[str, Any] = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(QwenAdapterError, match=message):
        QwenConfiguration.from_model_config(config)


def test_weight_spec_is_exact_direct_bare_mapping() -> None:
    specifications = qwen_weight_spec(_bundle())

    assert len(specifications) == QWEN_TENSOR_COUNT
    assert len({specification.bare_key for specification in specifications}) == len(specifications)
    assert all(not specification.bare_key.startswith("qwen.") for specification in specifications)
    assert all(not specification.bare_key.startswith("model.") for specification in specifications)
    assert all("lm_head" not in specification.bare_key for specification in specifications)
    q_projection = next(
        specification
        for specification in specifications
        if specification.bare_key == "layers.0.self_attn.q_proj.weight"
    )
    output_projection = next(
        specification
        for specification in specifications
        if specification.bare_key == "layers.0.self_attn.o_proj.weight"
    )
    assert q_projection.shape == (2048, 1024)
    assert output_projection.shape == (1024, 2048)


def test_weight_spec_rejects_missing_tensor() -> None:
    with pytest.raises(QwenAdapterError, match="310 tensors"):
        qwen_weight_spec(_bundle(_qwen_records()[:-1]))


def test_weight_spec_rejects_noncanonical_destination_without_transposing() -> None:
    records = list(_qwen_records())
    original = records[0]
    records[0] = TensorRecord(
        reference_name=original.reference_name,
        mlx_key="qwen.lm_head.weight",
        component=original.component,
        transform=original.transform,
        shape=tuple(reversed(original.shape)),
        dtype=original.dtype,
        numel=original.numel,
    )

    with pytest.raises(QwenAdapterError, match="canonical map"):
        qwen_weight_spec(_bundle(tuple(records)))


def test_numerical_diagnostics_reports_all_required_statistics() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    candidate = np.array([[1.0, 2.5], [2.5, 4.0]], dtype=np.float32)

    diagnostics = numerical_diagnostics(reference, candidate)

    assert diagnostics == NumericalDiagnostics(
        reference_shape=(2, 2),
        candidate_shape=(2, 2),
        shape_equal=True,
        max_absolute_error=0.5,
        mean_absolute_error=0.25,
        cosine_similarity=pytest.approx(0.9916316520429012),
    )
    assert diagnostics.to_dict()["shape_equal"] is True


def test_numerical_diagnostics_does_not_broadcast_shape_mismatch() -> None:
    diagnostics = numerical_diagnostics(np.ones((2, 1)), np.ones((2,)))

    assert diagnostics.shape_equal is False
    assert diagnostics.max_absolute_error is None
    assert diagnostics.mean_absolute_error is None
    assert diagnostics.cosine_similarity is None


def test_numerical_diagnostics_handles_zero_vectors_and_rejects_nonfinite() -> None:
    assert numerical_diagnostics(np.zeros(2), np.zeros(2)).cosine_similarity == 1.0
    assert numerical_diagnostics(np.zeros(2), np.ones(2)).cosine_similarity == 0.0
    with pytest.raises(QwenAdapterError, match="finite"):
        numerical_diagnostics(np.array([np.nan]), np.array([0.0]))


@pytest.mark.skipif(
    not RUN_REAL_PARITY,
    reason="set STREAMNER_RUN_MLX_PARITY=1 to run the real Metal parity check",
)
def test_real_qwen_hidden_states_and_native_cache_match_reference() -> None:
    if not ASSET_ROOT.is_dir() or not PARITY_ARRAYS.is_file():
        pytest.skip("exported reference assets and parity fixture are unavailable")
    bundle = load_asset_bundle(
        ASSET_ROOT,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    adapter = QwenAdapter.from_asset_bundle(bundle)
    with np.load(PARITY_ARRAYS, allow_pickle=False) as fixture:
        input_ids = fixture["input_ids"].copy()
        reference_hidden = fixture["qwen_final_hidden_states"].copy()

    cold_hidden = np.asarray(adapter.cold_hidden_states(input_ids))
    diagnostics = numerical_diagnostics(reference_hidden, cold_hidden)
    assert diagnostics.shape_equal
    assert diagnostics.cosine_similarity is not None
    assert diagnostics.cosine_similarity > 0.99999
    assert diagnostics.mean_absolute_error is not None
    assert diagnostics.mean_absolute_error < 0.01
    assert diagnostics.max_absolute_error is not None
    assert diagnostics.max_absolute_error < 0.25

    cache = adapter.create_cache()
    first = np.asarray(adapter.cached_hidden_states(input_ids[:, :6], cache))
    second = np.asarray(adapter.cached_hidden_states(input_ids[:, 6:], cache))
    assert adapter.cache_offsets(cache) == (input_ids.shape[1],) * 28
    cache_diagnostics = numerical_diagnostics(
        cold_hidden,
        np.concatenate((first, second), axis=1),
    )
    assert cache_diagnostics.cosine_similarity is not None
    assert cache_diagnostics.cosine_similarity > 0.999999
    assert cache_diagnostics.mean_absolute_error is not None
    assert cache_diagnostics.mean_absolute_error < 0.001
    assert cache_diagnostics.max_absolute_error is not None
    assert cache_diagnostics.max_absolute_error < 0.01
