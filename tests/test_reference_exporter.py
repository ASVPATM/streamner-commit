from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import load_file

from streamner_commit.reference import exporter

REVISION = "a" * 40


def _state_dict() -> dict[str, np.ndarray]:
    return {
        "token_rep_layer.decoder_layer.model.norm.weight": np.array([1.0, 2.0], dtype=np.float32),
        "labels_encoder.encoder.encoder.rel_embeddings.weight": np.array(
            [[3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        ),
        "span_rep_layer.span_rep_layer.project_start.0.weight": np.array(
            [[7.0, 8.0], [9.0, 10.0]], dtype=np.float32
        ),
        "prompt_rep_layer.0.bias": np.array([11.0, 12.0], dtype=np.float32),
    }


def _tokenizer() -> exporter.TokenizerExport:
    return exporter.TokenizerExport(
        assets={
            "tokenizer.json": b'{"version":"1.0"}\n',
            "tokenizer_config.json": b'{"padding_side":"right"}\n',
            "chat_template.jinja": b"{{ messages }}\n",
        },
        special_tokens_map={
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        },
        metadata={
            "eos_token": "<eos>",
            "eos_token_id": 2,
            "pad_token": "<pad>",
            "pad_token_id": 0,
            "label_marker_token": "<<LABEL>>",
            "label_marker_token_id": 3,
            "separator_marker_token": "<<SEP>>",
            "separator_marker_token_id": 4,
            "padding_side": "right",
            "vocabulary_size": 5,
        },
    )


def _export(tmp_path: Path) -> exporter.ReferenceExportResult:
    return exporter.export_reference_state(
        _state_dict(),
        config={"model_type": "gliner_streaming_span", "hidden_size": 2},
        tokenizer=_tokenizer(),
        output_root=tmp_path / "reference",
        model_id="example/private-model",
        revision=REVISION,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_module_reload_does_not_import_heavy_reference_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    attempted: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", maxsplit=1)[0] in {
            "gliner",
            "torch",
            "transformers",
            "huggingface_hub",
        }:
            attempted.append(name)
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(exporter)
    assert attempted == []


def test_tiny_export_is_safe_complete_classified_and_path_free(tmp_path: Path) -> None:
    result = _export(tmp_path)

    assert result.output_dir == tmp_path / "reference" / REVISION
    assert result.tensor_count == 4
    assert result.parameter_count == 12
    assert result.weights_path.is_file()
    assert result.export_manifest_path.is_file()

    tensors = load_file(result.weights_path)
    expected = _state_dict()
    assert list(tensors) == sorted(expected)
    for name, values in expected.items():
        np.testing.assert_array_equal(tensors[name], values)

    tensor_manifest = _read_json(result.tensor_manifest_path)
    assert tensor_manifest["schema_version"] == 1
    assert tensor_manifest["totals"] == {
        "parameter_count": 12,
        "tensor_count": 4,
    }
    assert tensor_manifest["weights"] == {
        "path": "model.safetensors",
        "sha256": exporter.sha256_file(result.weights_path),
        "size_bytes": result.weights_path.stat().st_size,
    }
    rows = tensor_manifest["tensors"]
    assert [row["reference_name"] for row in rows] == sorted(expected)
    assert {row["component"] for row in rows} == {
        "qwen",
        "label_encoder",
        "marker_v2",
        "prompt_projection",
    }
    assert {row["transform"] for row in rows} == {"none"}
    assert all(row["dtype"] == "float32" for row in rows)
    assert all(row["mlx_key"] for row in rows)

    export_manifest = _read_json(result.export_manifest_path)
    assert export_manifest["schema_version"] == 1
    assert export_manifest["model"] == {
        "id": "example/private-model",
        "revision": REVISION,
    }
    assert export_manifest["tensor_manifest"] == "tensor_manifest.json"
    assert export_manifest["totals"] == tensor_manifest["totals"]
    assert export_manifest["tokenizer"]["files"] == [
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    file_rows = export_manifest["files"]
    assert {row["role"] for row in file_rows} == {
        "weights",
        "config",
        "tensor_manifest",
        "tokenizer",
    }
    assert "export_manifest.json" not in {row["path"] for row in file_rows}
    for row in file_rows:
        relative = Path(row["path"])
        assert not relative.is_absolute()
        assert relative.name == row["path"]
        payload_path = result.output_dir / relative
        assert row["sha256"] == exporter.sha256_file(payload_path)
        assert row["size_bytes"] == payload_path.stat().st_size

    serialized_manifests = result.tensor_manifest_path.read_text(
        encoding="utf-8"
    ) + result.export_manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized_manifests


def test_repeated_export_is_byte_identical_and_leaves_no_temp_files(tmp_path: Path) -> None:
    first = _export(tmp_path)
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()}
    second = _export(tmp_path)
    after = {path.name: path.read_bytes() for path in second.output_dir.iterdir() if path.is_file()}

    assert after == before
    assert not [path for path in second.output_dir.iterdir() if path.name.startswith(".")]


def test_relative_output_root_does_not_leak_into_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = exporter.export_reference_state(
        _state_dict(),
        config={},
        tokenizer=_tokenizer(),
        output_root=Path("nested/reference"),
        model_id="example/model",
        revision=REVISION,
    )
    manifest = _read_json(result.export_manifest_path)
    assert all("/" not in row["path"] for row in manifest["files"])


def test_unclassified_tensor_fails_closed_before_writing_weights(tmp_path: Path) -> None:
    state = _state_dict()
    state["mystery.weight"] = np.ones((1,), dtype=np.float32)

    with pytest.raises(exporter.ReferenceExportError, match="unclassified"):
        exporter.export_reference_state(
            state,
            config={},
            tokenizer=_tokenizer(),
            output_root=tmp_path,
            model_id="example/model",
            revision=REVISION,
        )
    assert not (tmp_path / REVISION / "model.safetensors").exists()


@pytest.mark.parametrize(
    "bad_name",
    ["../tokenizer.json", "nested/tokenizer.json", "/tokenizer.json", ".."],
)
def test_unsafe_tokenizer_asset_names_are_rejected(tmp_path: Path, bad_name: str) -> None:
    valid = _tokenizer()
    assets = dict(valid.assets)
    assets[bad_name] = b"bad"
    token_data = exporter.TokenizerExport(
        assets=assets,
        special_tokens_map=valid.special_tokens_map,
        metadata=valid.metadata,
    )

    with pytest.raises(exporter.ReferenceExportError, match="unsafe tokenizer"):
        exporter.export_reference_state(
            _state_dict(),
            config={},
            tokenizer=token_data,
            output_root=tmp_path,
            model_id="example/model",
            revision=REVISION,
        )


def test_non_tensor_state_value_is_rejected(tmp_path: Path) -> None:
    state: dict[str, object] = _state_dict()
    state["prompt_rep_layer.0.bias"] = [1.0, 2.0]
    with pytest.raises(exporter.ReferenceExportError, match="supported tensor"):
        exporter.export_reference_state(
            state,
            config={},
            tokenizer=_tokenizer(),
            output_root=tmp_path,
            model_id="example/model",
            revision=REVISION,
        )


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39, "../" + "a" * 40])
def test_revision_must_be_an_immutable_safe_sha(tmp_path: Path, revision: str) -> None:
    with pytest.raises(ValueError, match="commit SHA"):
        exporter.export_reference_state(
            _state_dict(),
            config={},
            tokenizer=_tokenizer(),
            output_root=tmp_path,
            model_id="example/model",
            revision=revision,
        )
