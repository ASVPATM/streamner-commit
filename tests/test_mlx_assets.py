from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    AssetValidationError,
    load_asset_bundle,
)

REFERENCE_NAME = "prompt_rep_layer.0.weight"
MLX_KEY = "prompt_projection.layers.0.weight"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(root: Path, name: str, role: str) -> dict[str, Any]:
    path = root / name
    return {
        "name": Path(name).name,
        "path": name,
        "role": role,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _rewrite_export_manifest(root: Path) -> None:
    tensor_manifest = json.loads((root / "tensor_manifest.json").read_text(encoding="utf-8"))
    records = [
        _file_record(root, "config.json", "config"),
        _file_record(root, "model.safetensors", "weights"),
        _file_record(root, "special_tokens_map.json", "tokenizer"),
        _file_record(root, "tensor_manifest.json", "tensor_manifest"),
        _file_record(root, "tokenizer.json", "tokenizer"),
        _file_record(root, "tokenizer_config.json", "tokenizer"),
    ]
    export_manifest = {
        "schema_version": 1,
        "model": {"id": REFERENCE_MODEL_ID, "revision": REFERENCE_REVISION},
        "files": sorted(records, key=lambda row: row["path"]),
        "tensor_manifest": "tensor_manifest.json",
        "totals": tensor_manifest["totals"],
        "tokenizer": {
            "files": [
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            "special_tokens": {
                "eos_token": "[EOS]",
                "eos_token_id": 3,
                "label_marker_token": "<<LABEL>>",
                "label_marker_token_id": 4,
                "pad_token": "[PAD]",
                "pad_token_id": 2,
                "separator_marker_token": "<<SEP>>",
                "separator_marker_token_id": 5,
                "unk_token": "[UNK]",
                "unk_token_id": 0,
            },
        },
    }
    _write_json(root / "export_manifest.json", export_manifest)


def _make_bundle(root: Path) -> Path:
    root.mkdir()
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_file({REFERENCE_NAME: weights}, root / "model.safetensors")
    _write_json(root / "config.json", {"model_type": "tiny-test"})

    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "hello": 1,
                "[PAD]": 2,
                "[EOS]": 3,
                "<<LABEL>>": 4,
                "<<SEP>>": 5,
            },
            "[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    _write_json(
        root / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
        },
    )
    _write_json(
        root / "special_tokens_map.json",
        {"unk_token": "[UNK]", "pad_token": "[PAD]", "eos_token": "[EOS]"},
    )

    weights_path = root / "model.safetensors"
    tensor_manifest = {
        "schema_version": 1,
        "weights": {
            "path": "model.safetensors",
            "sha256": _sha256(weights_path),
            "size_bytes": weights_path.stat().st_size,
        },
        "totals": {"tensor_count": 1, "parameter_count": 6},
        "tensors": [
            {
                "reference_name": REFERENCE_NAME,
                "component": "prompt_projection",
                "mlx_key": MLX_KEY,
                "transform": "none",
                "shape": [2, 3],
                "dtype": "float32",
                "numel": 6,
            }
        ],
    }
    _write_json(root / "tensor_manifest.json", tensor_manifest)
    _rewrite_export_manifest(root)
    return root


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_valid_bundle_loads_without_torch_or_gliner(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    bundle = load_asset_bundle(
        root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
    )

    assert bundle.tensor_count == 1
    assert bundle.parameter_count == 6
    assert bundle.config == {"model_type": "tiny-test"}
    assert bundle.tensor_by_reference_name[REFERENCE_NAME].mlx_key == MLX_KEY
    assert bundle.tensor_by_mlx_key[MLX_KEY].shape == (2, 3)
    np.testing.assert_array_equal(
        bundle.load_weights_numpy()[REFERENCE_NAME],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    assert "torch" not in sys.modules
    assert "gliner" not in sys.modules


def test_exported_tokenizer_loads_locally(tmp_path: Path) -> None:
    bundle = load_asset_bundle(_make_bundle(tmp_path / "assets"))

    tokenizer = bundle.load_tokenizer()

    assert tokenizer.encode("hello", add_special_tokens=False) == [1]
    assert tokenizer.pad_token_id == 2
    assert tokenizer.eos_token_id == 3


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("expected_revision", "wrong", "revision mismatch"),
        ("expected_model_id", "wrong", "model ID mismatch"),
    ],
)
def test_expected_model_identity_is_enforced(
    tmp_path: Path, argument: str, value: str, message: str
) -> None:
    root = _make_bundle(tmp_path / "assets")
    kwargs: dict[str, Any] = {argument: value}

    with pytest.raises(AssetValidationError, match=message):
        load_asset_bundle(root, **kwargs)


def test_strict_reference_rejects_tiny_fixture_counts(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")

    with pytest.raises(AssetValidationError, match="371 tensors"):
        load_asset_bundle(root, strict_reference=True)


@pytest.mark.parametrize("malicious_path", ["../tensor_manifest.json", "/tmp/manifest.json"])
def test_manifest_locator_rejects_traversal_and_absolute_paths(
    tmp_path: Path, malicious_path: str
) -> None:
    root = _make_bundle(tmp_path / "assets")
    export = _load_json(root / "export_manifest.json")
    export["tensor_manifest"] = malicious_path
    _write_json(root / "export_manifest.json", export)

    with pytest.raises(AssetValidationError, match="absolute|traverse"):
        load_asset_bundle(root)


def test_manifest_rejects_symlink_that_escapes_root(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    outside = tmp_path / "outside.json"
    _write_json(outside, {})
    (root / "escaped.json").symlink_to(outside)
    export = _load_json(root / "export_manifest.json")
    export["tensor_manifest"] = "escaped.json"
    _write_json(root / "export_manifest.json", export)

    with pytest.raises(AssetValidationError, match="escapes"):
        load_asset_bundle(root)


def test_pickle_like_file_is_rejected_before_loading(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    unsafe = root / "model.bin"
    unsafe.write_bytes(b"not a safe tensor")
    export = _load_json(root / "export_manifest.json")
    weights_record = next(record for record in export["files"] if record["role"] == "weights")
    weights_record.update(
        {
            "name": unsafe.name,
            "path": unsafe.name,
            "sha256": _sha256(unsafe),
            "size_bytes": unsafe.stat().st_size,
        }
    )
    _write_json(root / "export_manifest.json", export)

    with pytest.raises(AssetValidationError, match="unsafe pickle-like"):
        load_asset_bundle(root)


def test_payload_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    (root / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(AssetValidationError, match="mismatch for config.json"):
        load_asset_bundle(root)


def test_tensor_total_must_match_rows(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    manifest = _load_json(root / "tensor_manifest.json")
    manifest["totals"]["tensor_count"] = 2
    _write_json(root / "tensor_manifest.json", manifest)
    _rewrite_export_manifest(root)

    with pytest.raises(AssetValidationError, match="tensor count mismatch"):
        load_asset_bundle(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", [3, 2], "shape mismatch"),
        ("dtype", "float16", "dtype mismatch"),
    ],
)
def test_safe_header_must_match_manifest_metadata(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    root = _make_bundle(tmp_path / "assets")
    manifest = _load_json(root / "tensor_manifest.json")
    manifest["tensors"][0][field] = value
    _write_json(root / "tensor_manifest.json", manifest)
    _rewrite_export_manifest(root)

    with pytest.raises(AssetValidationError, match=message):
        load_asset_bundle(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_name", "unclassified.weight", "unclassified"),
        ("component", "unknown", "unclassified"),
        ("component", "qwen", "canonical weight map"),
        ("mlx_key", "prompt_projection.wrong", "canonical weight map"),
        ("transform", "transpose", "expects 'none'"),
    ],
)
def test_every_tensor_requires_canonical_classification(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    root = _make_bundle(tmp_path / "assets")
    manifest = _load_json(root / "tensor_manifest.json")
    manifest["tensors"][0][field] = value
    _write_json(root / "tensor_manifest.json", manifest)
    _rewrite_export_manifest(root)

    with pytest.raises(AssetValidationError, match=message):
        load_asset_bundle(root)


def test_tensor_inventory_must_match_safe_file_keys(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    manifest = _load_json(root / "tensor_manifest.json")
    manifest["tensors"].append(
        {
            "reference_name": "prompt_rep_layer.0.bias",
            "component": "prompt_projection",
            "mlx_key": "prompt_projection.layers.0.bias",
            "transform": "none",
            "shape": [2],
            "dtype": "float32",
            "numel": 2,
        }
    )
    manifest["tensors"].sort(key=lambda row: row["reference_name"])
    manifest["totals"] = {"tensor_count": 2, "parameter_count": 8}
    _write_json(root / "tensor_manifest.json", manifest)
    _rewrite_export_manifest(root)

    with pytest.raises(AssetValidationError, match="keys disagree"):
        load_asset_bundle(root)


def test_weights_are_rehashed_when_arrays_are_requested(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    bundle = load_asset_bundle(root)
    with (root / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(AssetValidationError, match="changed after validation"):
        bundle.load_weights_numpy()


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    (root / "export_manifest.json").write_text(
        '{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8"
    )

    with pytest.raises(AssetValidationError, match="duplicate key"):
        load_asset_bundle(root)
