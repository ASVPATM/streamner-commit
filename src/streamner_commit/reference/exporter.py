"""Deterministic, safe export of the locked reference checkpoint.

This module is import-safe in the main MLX environment: PyTorch, GLiNER,
Transformers, and Hugging Face Hub are imported only by the production export
entry point.  The pure export helper accepts tiny NumPy-backed fixtures for
unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TypedDict

import numpy as np

from streamner_commit.backends.reference_gliner import (
    DEFAULT_MODEL_ID,
    require_pinned_gliner,
)
from streamner_commit.conversion.weight_map import classify_reference_inventory

LOCKED_MODEL_REVISION = "e871777dc4b3b688747a0433fff8d94a36fcc7b0"
SCHEMA_VERSION = 1
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
TENSOR_MANIFEST_FILENAME = "tensor_manifest.json"
EXPORT_MANIFEST_FILENAME = "export_manifest.json"
SPECIAL_TOKENS_FILENAME = "special_tokens_map.json"

_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_RESERVED_FILENAMES = frozenset(
    {
        WEIGHTS_FILENAME,
        CONFIG_FILENAME,
        TENSOR_MANIFEST_FILENAME,
        EXPORT_MANIFEST_FILENAME,
    }
)
_TOKENIZER_SOURCE_FILENAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
_SNAPSHOT_ALLOW_PATTERNS = (
    "gliner_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    *_TOKENIZER_SOURCE_FILENAMES,
)

_LOCKED_TENSOR_COUNT = 371
_LOCKED_PARAMETER_COUNT = 676_575_232
_LOCKED_COMPONENT_COUNTS = {
    "label_encoder": 41,
    "marker_v2": 16,
    "prompt_projection": 4,
    "qwen": 310,
}
_LOCKED_COMPONENT_PARAMETERS = {
    "label_encoder": 30_439_424,
    "marker_v2": 41_963_520,
    "prompt_projection": 8_393_728,
    "qwen": 595_778_560,
}


class ReferenceExportError(RuntimeError):
    """Raised when a reference export cannot be proven complete and safe."""


class _TensorRow(TypedDict):
    reference_name: str
    shape: list[int]
    dtype: str
    numel: int
    component: str
    mlx_key: str
    transform: str


@dataclass(frozen=True, slots=True)
class TokenizerExport:
    """Tokenizer bytes plus explicit special-token metadata."""

    assets: Mapping[str, bytes]
    special_tokens_map: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ReferenceExportResult:
    """Paths and inventory totals from one completed export."""

    output_dir: Path
    weights_path: Path
    tensor_manifest_path: Path
    export_manifest_path: Path
    tensor_count: int
    parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": self.output_dir.as_posix(),
            "weights_path": self.weights_path.as_posix(),
            "tensor_manifest_path": self.tensor_manifest_path.as_posix(),
            "export_manifest_path": self.export_manifest_path.as_posix(),
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
        }


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it whole."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ReferenceExportError(f"export metadata is not JSON-safe: {error}") from error
    return (text + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_safetensors(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    module = import_module("safetensors.numpy")
    save_file = getattr(module, "save_file", None)
    if not callable(save_file):
        raise ReferenceExportError("safetensors.numpy.save_file is unavailable")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary_path), metadata={"format": "pt"})
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception as error:
        raise ReferenceExportError(f"could not write safe weights: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("model revision must be a lowercase 40-character commit SHA")
    return revision


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-blank string")
    return model_id


def _validate_asset_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ReferenceExportError(f"unsafe tokenizer asset name: {name!r}")
    if name in _RESERVED_FILENAMES:
        raise ReferenceExportError(f"tokenizer asset uses reserved filename: {name!r}")
    return name


def _as_numpy_tensor(name: str, value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        current = value
        for operation in ("detach", "cpu", "contiguous"):
            method = getattr(current, operation, None)
            if not callable(method):
                raise ReferenceExportError(f"state-dict entry {name!r} is not a supported tensor")
            current = method()
        to_numpy = getattr(current, "numpy", None)
        if not callable(to_numpy):
            raise ReferenceExportError(f"state-dict entry {name!r} cannot be converted to NumPy")
        array = np.asarray(to_numpy())

    if array.dtype.hasobject or array.dtype.kind in {"U", "S", "V"}:
        raise ReferenceExportError(f"state-dict entry {name!r} has unsupported dtype {array.dtype}")
    return np.ascontiguousarray(array)


def _file_record(path: Path, *, role: str) -> dict[str, object]:
    # The export is a flat, self-contained directory. Never persist the caller's
    # output-root path (absolute or relative) into the portable manifest.
    relative_path = path.name
    if relative_path in {"", ".", ".."}:
        raise ReferenceExportError(f"manifest file path is not a basename: {relative_path!r}")
    return {
        "name": path.name,
        "path": relative_path,
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_locked_inventory(
    *,
    model_id: str,
    revision: str,
    rows: list[_TensorRow],
) -> None:
    if model_id != DEFAULT_MODEL_ID or revision != LOCKED_MODEL_REVISION:
        return

    tensor_count = len(rows)
    parameter_count = sum(row["numel"] for row in rows)
    component_counts = Counter(str(row["component"]) for row in rows)
    component_parameters: Counter[str] = Counter()
    for row in rows:
        component_parameters[row["component"]] += row["numel"]
    dtypes = {str(row["dtype"]) for row in rows}

    errors: list[str] = []
    if tensor_count != _LOCKED_TENSOR_COUNT:
        errors.append(f"tensor count {tensor_count} != {_LOCKED_TENSOR_COUNT}")
    if parameter_count != _LOCKED_PARAMETER_COUNT:
        errors.append(f"parameter count {parameter_count} != {_LOCKED_PARAMETER_COUNT}")
    if dict(sorted(component_counts.items())) != _LOCKED_COMPONENT_COUNTS:
        errors.append(f"component counts {dict(sorted(component_counts.items()))!r}")
    if dict(sorted(component_parameters.items())) != _LOCKED_COMPONENT_PARAMETERS:
        errors.append(f"component parameter counts {dict(sorted(component_parameters.items()))!r}")
    if dtypes != {"float32"}:
        errors.append(f"dtypes {sorted(dtypes)!r} != ['float32']")
    if errors:
        raise ReferenceExportError(
            "locked checkpoint inventory differs from the reviewed architecture: "
            + "; ".join(errors)
        )


def export_reference_state(
    state_dict: Mapping[str, object],
    *,
    config: Mapping[str, object],
    tokenizer: TokenizerExport,
    output_root: Path,
    model_id: str,
    revision: str,
) -> ReferenceExportResult:
    """Write one classified state dict and its tokenizer as a safe export.

    The helper is independent of GLiNER and PyTorch. Every write is replaced
    atomically, and ``export_manifest.json`` is written last as the completion
    marker. Re-running with identical inputs produces byte-identical payloads.
    """

    normalized_model_id = _validate_model_id(model_id)
    normalized_revision = _validate_revision(revision)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ReferenceExportError("state_dict must be a non-empty mapping")
    if not isinstance(config, Mapping):
        raise ReferenceExportError("config must be a mapping")
    if not isinstance(tokenizer, TokenizerExport):
        raise TypeError("tokenizer must be a TokenizerExport")

    reference_names = list(state_dict)
    if not all(isinstance(name, str) for name in reference_names):
        raise ReferenceExportError("every state-dict key must be a string")
    try:
        mappings, _component_counts = classify_reference_inventory(reference_names)
    except ValueError as error:
        raise ReferenceExportError(str(error)) from error

    arrays: dict[str, np.ndarray] = {}
    rows: list[_TensorRow] = []
    for mapping in mappings:
        array = _as_numpy_tensor(
            mapping.reference_name,
            state_dict[mapping.reference_name],
        )
        arrays[mapping.reference_name] = array
        rows.append(
            {
                "reference_name": mapping.reference_name,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "numel": int(array.size),
                "component": mapping.component,
                "mlx_key": mapping.mlx_key,
                "transform": mapping.transform,
            }
        )

    _validate_locked_inventory(
        model_id=normalized_model_id,
        revision=normalized_revision,
        rows=rows,
    )

    assets: dict[str, bytes] = {}
    for raw_name, payload in tokenizer.assets.items():
        name = _validate_asset_name(raw_name)
        if not isinstance(payload, bytes):
            raise ReferenceExportError(f"tokenizer asset {name!r} must contain bytes")
        assets[name] = payload
    if "tokenizer.json" not in assets or "tokenizer_config.json" not in assets:
        raise ReferenceExportError(
            "tokenizer assets must include tokenizer.json and tokenizer_config.json"
        )
    assets.setdefault(
        SPECIAL_TOKENS_FILENAME,
        _json_bytes(dict(tokenizer.special_tokens_map)),
    )

    output_dir = Path(output_root) / normalized_revision
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_FILENAME
    config_path = output_dir / CONFIG_FILENAME
    tensor_manifest_path = output_dir / TENSOR_MANIFEST_FILENAME
    export_manifest_path = output_dir / EXPORT_MANIFEST_FILENAME

    _atomic_save_safetensors(weights_path, arrays)
    _atomic_write_bytes(config_path, _json_bytes(dict(config)))
    tokenizer_paths: list[Path] = []
    for name, payload in sorted(assets.items()):
        path = output_dir / name
        _atomic_write_bytes(path, payload)
        tokenizer_paths.append(path)

    tensor_count = len(rows)
    parameter_count = sum(row["numel"] for row in rows)
    totals = {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
    }
    tensor_manifest = {
        "schema_version": SCHEMA_VERSION,
        "weights": {
            "path": WEIGHTS_FILENAME,
            "sha256": sha256_file(weights_path),
            "size_bytes": weights_path.stat().st_size,
        },
        "totals": totals,
        "tensors": rows,
    }
    _atomic_write_bytes(tensor_manifest_path, _json_bytes(tensor_manifest))

    file_records = [
        _file_record(weights_path, role="weights"),
        _file_record(config_path, role="config"),
        _file_record(tensor_manifest_path, role="tensor_manifest"),
        *(_file_record(path, role="tokenizer") for path in tokenizer_paths),
    ]
    file_records.sort(key=lambda row: str(row["path"]))
    export_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "id": normalized_model_id,
            "revision": normalized_revision,
        },
        "files": file_records,
        "tensor_manifest": TENSOR_MANIFEST_FILENAME,
        "totals": totals,
        "tokenizer": {
            "files": [path.name for path in sorted(tokenizer_paths)],
            "special_tokens": dict(tokenizer.metadata),
        },
    }
    _atomic_write_bytes(export_manifest_path, _json_bytes(export_manifest))

    return ReferenceExportResult(
        output_dir=output_dir,
        weights_path=weights_path,
        tensor_manifest_path=tensor_manifest_path,
        export_manifest_path=export_manifest_path,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
    )


def _load_checkpoint(snapshot: Path) -> Mapping[str, object]:
    safe_path = snapshot / WEIGHTS_FILENAME
    pytorch_path = snapshot / "pytorch_model.bin"
    if safe_path.is_file():
        module = import_module("safetensors.numpy")
        load_file = getattr(module, "load_file", None)
        if not callable(load_file):
            raise ReferenceExportError("safetensors.numpy.load_file is unavailable")
        loaded = load_file(str(safe_path))
    elif pytorch_path.is_file():
        torch = import_module("torch")
        load = getattr(torch, "load", None)
        if not callable(load):
            raise ReferenceExportError("torch.load is unavailable")
        loaded = load(pytorch_path, map_location="cpu", weights_only=True)
    else:
        raise ReferenceExportError("snapshot has neither model.safetensors nor pytorch_model.bin")
    if not isinstance(loaded, Mapping) or not loaded:
        raise ReferenceExportError("reference checkpoint is not a non-empty state dict")
    return loaded


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceExportError(f"could not read {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ReferenceExportError(f"{path.name} must contain a JSON object")
    return payload


def _tokenizer_export(snapshot: Path, config: Mapping[str, object]) -> TokenizerExport:
    transformers = import_module("transformers")
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None or not callable(getattr(auto_tokenizer, "from_pretrained", None)):
        raise ReferenceExportError("Transformers AutoTokenizer is unavailable")
    tokenizer = auto_tokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )

    vocabulary_size = len(tokenizer)
    configured_vocabulary_size = config.get("vocab_size")
    if configured_vocabulary_size != vocabulary_size:
        raise ReferenceExportError(
            "tokenizer size does not match config.vocab_size: "
            f"{vocabulary_size} != {configured_vocabulary_size}"
        )
    if tokenizer.padding_side != "right":
        raise ReferenceExportError("the locked streaming tokenizer must use right padding")

    label_token = config.get("label_token")
    separator_token = config.get("sep_token")
    if not isinstance(label_token, str) or not isinstance(separator_token, str):
        raise ReferenceExportError("config is missing label_token or sep_token")
    label_token_id = tokenizer.convert_tokens_to_ids(label_token)
    separator_token_id = tokenizer.convert_tokens_to_ids(separator_token)

    expected_ids = {
        "class_token_index": label_token_id,
        "sep_token_index": separator_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for config_key, actual in expected_ids.items():
        expected = config.get(config_key)
        if expected != actual:
            raise ReferenceExportError(
                f"tokenizer {config_key} mismatch: {actual!r} != {expected!r}"
            )

    assets: dict[str, bytes] = {}
    for name in _TOKENIZER_SOURCE_FILENAMES:
        source = snapshot / name
        if source.is_file():
            assets[name] = source.read_bytes()

    special_tokens_map = dict(tokenizer.special_tokens_map)
    additional_special_tokens = getattr(tokenizer, "additional_special_tokens", ())
    additional_special_token_ids = getattr(
        tokenizer,
        "additional_special_tokens_ids",
        (),
    )
    metadata = {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "additional_special_tokens": list(additional_special_tokens),
        "additional_special_token_ids": list(additional_special_token_ids),
        "all_special_tokens": list(tokenizer.all_special_tokens),
        "all_special_ids": list(tokenizer.all_special_ids),
        "label_marker_token": label_token,
        "label_marker_token_id": label_token_id,
        "separator_marker_token": separator_token,
        "separator_marker_token_id": separator_token_id,
        "padding_side": tokenizer.padding_side,
        "vocabulary_size": vocabulary_size,
    }
    return TokenizerExport(
        assets=assets,
        special_tokens_map=special_tokens_map,
        metadata=metadata,
    )


def export_reference_assets(
    *,
    output_root: Path,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = LOCKED_MODEL_REVISION,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
) -> ReferenceExportResult:
    """Resolve the locked snapshot and export it without unsafe deserialization.

    Network access is disabled by default. Legacy PyTorch checkpoints are read
    with ``weights_only=True`` in the pinned reference environment, immediately
    converted to host arrays, and written as Safetensors.
    """

    normalized_model_id = _validate_model_id(model_id)
    normalized_revision = _validate_revision(revision)
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a boolean")
    require_pinned_gliner()

    hub = import_module("huggingface_hub")
    snapshot_download = getattr(hub, "snapshot_download", None)
    if not callable(snapshot_download):
        raise ReferenceExportError("huggingface_hub.snapshot_download is unavailable")
    options: dict[str, object] = {
        "repo_id": normalized_model_id,
        "revision": normalized_revision,
        "local_files_only": local_files_only,
        "allow_patterns": list(_SNAPSHOT_ALLOW_PATTERNS),
    }
    if cache_dir is not None:
        options["cache_dir"] = str(cache_dir)
    try:
        snapshot = Path(snapshot_download(**options))
    except Exception as error:
        mode = "local cache" if local_files_only else "Hub/cache"
        raise ReferenceExportError(
            f"could not resolve locked snapshot from {mode}: {error}"
        ) from error

    config_path = snapshot / "gliner_config.json"
    if not config_path.is_file():
        raise ReferenceExportError("snapshot is missing gliner_config.json")
    config = _load_json_object(config_path)
    tokenizer = _tokenizer_export(snapshot, config)
    state_dict = _load_checkpoint(snapshot)
    return export_reference_state(
        state_dict,
        config=config,
        tokenizer=tokenizer,
        output_root=output_root,
        model_id=normalized_model_id,
        revision=normalized_revision,
    )


__all__ = [
    "CONFIG_FILENAME",
    "EXPORT_MANIFEST_FILENAME",
    "LOCKED_MODEL_REVISION",
    "ReferenceExportError",
    "ReferenceExportResult",
    "SPECIAL_TOKENS_FILENAME",
    "TENSOR_MANIFEST_FILENAME",
    "TokenizerExport",
    "WEIGHTS_FILENAME",
    "export_reference_assets",
    "export_reference_state",
    "sha256_file",
]
