"""Validate and load exported reference assets without PyTorch or GLiNER.

The reference environment owns conversion from the original pickle checkpoint to
``model.safetensors``.  This module is the trust boundary in the main environment: it
accepts only schema-versioned, hash-checked files contained by one export directory and
checks the complete tensor inventory against the safe tensor header before exposing it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import load_file

from streamner_commit.conversion.weight_map import (
    COMPONENTS,
    TRANSFORMS,
    WeightMappingError,
    map_reference_key,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


SCHEMA_VERSION = 1
REFERENCE_MODEL_ID = "knowledgator/gliner-stream-pii-v1.0"
REFERENCE_REVISION = "e871777dc4b3b688747a0433fff8d94a36fcc7b0"
REFERENCE_TENSOR_COUNT = 371
REFERENCE_PARAMETER_COUNT = 676_575_232

_UNSAFE_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}
_UNCLASSIFIED_VALUES = {"", "none", "null", "todo", "unknown", "unclassified"}
_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}
_DTYPE_ALIASES = {
    **_SAFETENSORS_DTYPES,
    **{value: value for value in _SAFETENSORS_DTYPES.values()},
    "boolean": "bool",
    "torch.bool": "bool",
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "numpy.bool_": "bool",
    "numpy.float16": "float16",
    "numpy.float32": "float32",
    "numpy.float64": "float64",
    "numpy.int8": "int8",
    "numpy.int16": "int16",
    "numpy.int32": "int32",
    "numpy.int64": "int64",
}


class AssetValidationError(ValueError):
    """An exported asset bundle failed an integrity or schema check."""


@dataclass(frozen=True, slots=True)
class TensorRecord:
    """One fully classified reference tensor."""

    reference_name: str
    mlx_key: str
    component: str
    transform: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


@dataclass(frozen=True, slots=True)
class AssetBundle:
    """Validated paths and metadata consumed by the MLX port."""

    root: Path
    model_id: str
    revision: str
    weights_path: Path
    config_path: Path
    tensor_manifest_path: Path
    tokenizer_files: tuple[Path, ...]
    tokenizer_special_tokens: dict[str, Any]
    tensors: tuple[TensorRecord, ...]
    tensor_count: int
    parameter_count: int
    weights_sha256: str
    config: dict[str, Any]
    export_manifest: dict[str, Any]
    tensor_manifest: dict[str, Any]

    @property
    def tensor_by_reference_name(self) -> dict[str, TensorRecord]:
        """Return a fresh reference-name index suitable for runtime mapping."""

        return {record.reference_name: record for record in self.tensors}

    @property
    def tensor_by_mlx_key(self) -> dict[str, TensorRecord]:
        """Return a fresh MLX-key index suitable for parameter assignment."""

        return {record.mlx_key: record for record in self.tensors}

    def load_weights_numpy(self) -> dict[str, np.ndarray[Any, Any]]:
        """Load the validated safe weights as NumPy arrays.

        Integrity is checked again so a file modified after bundle construction is not
        silently consumed.  ``safetensors.numpy`` never invokes pickle.
        """

        actual_hash = _sha256(self.weights_path)
        if actual_hash != self.weights_sha256:
            raise AssetValidationError(
                "model.safetensors changed after validation: "
                f"expected SHA-256 {self.weights_sha256}, got {actual_hash}"
            )
        try:
            arrays = load_file(self.weights_path)
        except SafetensorError as exc:
            raise AssetValidationError(f"cannot load safe weights: {exc}") from exc
        _validate_loaded_arrays(arrays, self.tensors)
        return arrays

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Load the exported tokenizer locally with remote code disabled."""

        # Transformers is deliberately lazy: importing the asset module itself remains a
        # small, torch-free operation and callers opt into tokenizer construction.
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.root,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise AssetValidationError(f"cannot load exported tokenizer: {exc}") from exc

        marker_fields = {
            "label_marker_token",
            "label_marker_token_id",
            "separator_marker_token",
            "separator_marker_token_id",
        }
        for field, expected in self.tokenizer_special_tokens.items():
            if field in marker_fields:
                continue
            if field == "vocabulary_size":
                actual_size = len(tokenizer)
                if actual_size != expected:
                    raise AssetValidationError(
                        "tokenizer vocabulary-size mismatch: "
                        f"expected {expected}, got {actual_size}"
                    )
                continue
            if field == "additional_special_tokens":
                if not isinstance(expected, list) or any(
                    not isinstance(token, str) for token in expected
                ):
                    raise AssetValidationError(
                        "tokenizer additional_special_tokens must be a string list"
                    )
                continue
            if field == "additional_special_token_ids":
                tokens = self.tokenizer_special_tokens.get("additional_special_tokens")
                if not isinstance(tokens, list):
                    raise AssetValidationError(
                        "tokenizer metadata must pair additional special tokens and IDs"
                    )
                actual_ids = tokenizer.convert_tokens_to_ids(tokens)
                if actual_ids != expected:
                    raise AssetValidationError(
                        "tokenizer additional-special-token IDs disagree with metadata"
                    )
                continue
            if not hasattr(tokenizer, field):
                raise AssetValidationError(
                    f"tokenizer does not expose recorded special-token field {field!r}"
                )
            actual = getattr(tokenizer, field)
            if actual != expected:
                raise AssetValidationError(
                    f"tokenizer special token mismatch for {field}: "
                    f"expected {expected}, got {actual}"
                )

        for marker in ("label_marker", "separator_marker"):
            token_field = f"{marker}_token"
            id_field = f"{marker}_token_id"
            has_token = token_field in self.tokenizer_special_tokens
            has_id = id_field in self.tokenizer_special_tokens
            if has_token != has_id:
                raise AssetValidationError(
                    f"tokenizer metadata must record both {token_field} and {id_field}"
                )
            if has_token:
                expected_token = self.tokenizer_special_tokens[token_field]
                expected_id = self.tokenizer_special_tokens[id_field]
                if not isinstance(expected_token, str) or not isinstance(expected_id, int):
                    raise AssetValidationError(
                        f"tokenizer marker metadata for {marker} has invalid types"
                    )
                actual_id = tokenizer.convert_tokens_to_ids(expected_token)
                if actual_id != expected_id:
                    raise AssetValidationError(
                        f"tokenizer marker mismatch for {expected_token!r}: "
                        f"expected ID {expected_id}, got {actual_id}"
                    )
        return tokenizer


def load_asset_bundle(
    root: str | Path,
    *,
    expected_revision: str | None = None,
    expected_model_id: str | None = None,
    strict_reference: bool = False,
) -> AssetBundle:
    """Validate an exported asset directory and return a runtime-facing bundle.

    ``strict_reference=True`` additionally enforces the frozen checkpoint totals (371
    tensors and 676,575,232 parameters).  Tiny unit fixtures use the same schema with
    this flag disabled; production callers should enable it and pass the revision from
    ``configs/model_lock.json``.
    """

    export_root = Path(root).expanduser().resolve()
    if not export_root.is_dir():
        raise AssetValidationError(f"asset root is not a directory: {export_root}")

    export_manifest_path = _contained_file(export_root, "export_manifest.json", "export manifest")
    export = _load_json_object(export_manifest_path, "export manifest")
    _require_schema_version(export, "export manifest")

    model = _require_object(export, "model", "export manifest")
    model_id = _require_nonempty_string(model, "id", "export manifest.model")
    revision = _require_nonempty_string(model, "revision", "export manifest.model")
    if expected_revision is not None and revision != expected_revision:
        raise AssetValidationError(
            f"model revision mismatch: expected {expected_revision}, got {revision}"
        )
    if expected_model_id is not None and model_id != expected_model_id:
        raise AssetValidationError(
            f"model ID mismatch: expected {expected_model_id}, got {model_id}"
        )

    file_records = _validate_file_records(export_root, export)
    role_paths = _index_role_paths(file_records)
    weights_path = _single_role_path(role_paths, "weights")
    config_path = _single_role_path(role_paths, "config")
    listed_tensor_manifest_path = _single_role_path(role_paths, "tensor_manifest")

    manifest_locator = _require_nonempty_string(export, "tensor_manifest", "export manifest")
    tensor_manifest_path = _contained_file(
        export_root, manifest_locator, "export manifest.tensor_manifest"
    )
    if tensor_manifest_path != listed_tensor_manifest_path:
        raise AssetValidationError(
            "tensor manifest locator does not match the file with role 'tensor_manifest'"
        )

    tokenizer = _require_object(export, "tokenizer", "export manifest")
    tokenizer_names = _require_string_list(tokenizer, "files", "export manifest.tokenizer")
    tokenizer_paths = tuple(
        _contained_file(export_root, name, "export manifest.tokenizer.files")
        for name in tokenizer_names
    )
    listed_tokenizer_paths = tuple(role_paths.get("tokenizer", ()))
    if set(tokenizer_paths) != set(listed_tokenizer_paths):
        raise AssetValidationError("tokenizer.files must exactly match files with role 'tokenizer'")
    if not tokenizer_paths:
        raise AssetValidationError("export manifest must list at least one tokenizer file")
    special_tokens = _require_object(tokenizer, "special_tokens", "export manifest.tokenizer")

    config = _load_json_object(config_path, "model config")
    tensor_manifest = _load_json_object(tensor_manifest_path, "tensor manifest")
    _require_schema_version(tensor_manifest, "tensor manifest")

    weights = _require_object(tensor_manifest, "weights", "tensor manifest")
    tensor_weights_path = _contained_file(
        export_root,
        _require_nonempty_string(weights, "path", "tensor manifest.weights"),
        "tensor manifest.weights.path",
    )
    if tensor_weights_path != weights_path:
        raise AssetValidationError("tensor manifest weights path does not match files inventory")
    weights_hash = _require_sha256(weights, "sha256", "tensor manifest.weights")
    weights_size = _require_nonnegative_int(weights, "size_bytes", "tensor manifest.weights")
    if weights_path.suffix != ".safetensors":
        raise AssetValidationError("weights must use safe .safetensors serialization")
    _cross_check_file_record(file_records, weights_path, weights_hash, weights_size)

    records = _parse_tensor_records(tensor_manifest)
    totals = _require_object(tensor_manifest, "totals", "tensor manifest")
    tensor_count = _require_nonnegative_int(totals, "tensor_count", "tensor manifest.totals")
    parameter_count = _require_nonnegative_int(totals, "parameter_count", "tensor manifest.totals")
    if tensor_count != len(records):
        raise AssetValidationError(
            f"tensor count mismatch: manifest total {tensor_count}, rows {len(records)}"
        )
    row_parameter_count = sum(record.numel for record in records)
    if parameter_count != row_parameter_count:
        raise AssetValidationError(
            "parameter count mismatch: "
            f"manifest total {parameter_count}, rows {row_parameter_count}"
        )

    export_totals = _require_object(export, "totals", "export manifest")
    export_tensor_count = _require_nonnegative_int(
        export_totals, "tensor_count", "export manifest.totals"
    )
    if export_tensor_count != tensor_count:
        raise AssetValidationError("export and tensor manifest tensor totals disagree")
    if (
        _require_nonnegative_int(export_totals, "parameter_count", "export manifest.totals")
        != parameter_count
    ):
        raise AssetValidationError("export and tensor manifest parameter totals disagree")

    if strict_reference:
        if model_id != REFERENCE_MODEL_ID:
            raise AssetValidationError(
                f"locked reference model ID must be {REFERENCE_MODEL_ID}, got {model_id}"
            )
        if revision != REFERENCE_REVISION:
            raise AssetValidationError(
                f"locked reference revision must be {REFERENCE_REVISION}, got {revision}"
            )
        if tensor_count != REFERENCE_TENSOR_COUNT:
            raise AssetValidationError(
                "locked reference must contain "
                f"{REFERENCE_TENSOR_COUNT} tensors, got {tensor_count}"
            )
        if parameter_count != REFERENCE_PARAMETER_COUNT:
            raise AssetValidationError(
                "locked reference must contain "
                f"{REFERENCE_PARAMETER_COUNT} parameters, got {parameter_count}"
            )

    _validate_safetensors_header(weights_path, records)

    return AssetBundle(
        root=export_root,
        model_id=model_id,
        revision=revision,
        weights_path=weights_path,
        config_path=config_path,
        tensor_manifest_path=tensor_manifest_path,
        tokenizer_files=tokenizer_paths,
        tokenizer_special_tokens=dict(special_tokens),
        tensors=records,
        tensor_count=tensor_count,
        parameter_count=parameter_count,
        weights_sha256=weights_hash,
        config=config,
        export_manifest=export,
        tensor_manifest=tensor_manifest,
    )


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AssetValidationError(f"{description} contains non-finite number {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssetValidationError(f"{description} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetValidationError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssetValidationError(f"{description} must be a JSON object")
    return value


def _require_schema_version(value: dict[str, Any], description: str) -> None:
    version = _require_nonnegative_int(value, "schema_version", description)
    if version != SCHEMA_VERSION:
        raise AssetValidationError(
            f"unsupported {description} schema version {version}; expected {SCHEMA_VERSION}"
        )


def _require_object(value: dict[str, Any], key: str, description: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise AssetValidationError(f"{description}.{key} must be an object")
    return result


def _require_nonempty_string(value: dict[str, Any], key: str, description: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise AssetValidationError(f"{description}.{key} must be a non-empty string")
    return result


def _require_nonnegative_int(value: dict[str, Any], key: str, description: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise AssetValidationError(f"{description}.{key} must be a non-negative integer")
    return result


def _require_sha256(value: dict[str, Any], key: str, description: str) -> str:
    result = _require_nonempty_string(value, key, description).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise AssetValidationError(f"{description}.{key} must be a 64-character SHA-256 hex digest")
    return result


def _require_string_list(value: dict[str, Any], key: str, description: str) -> list[str]:
    result = value.get(key)
    if not isinstance(result, list) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise AssetValidationError(f"{description}.{key} must be a list of non-empty strings")
    if len(set(result)) != len(result):
        raise AssetValidationError(f"{description}.{key} contains duplicates")
    return result


def _contained_file(root: Path, relative: str, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AssetValidationError(f"{description} must be a non-empty relative path")
    if "\\" in relative or "\x00" in relative:
        raise AssetValidationError(f"{description} contains an invalid path separator or NUL")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AssetValidationError(f"{description} must not be absolute or traverse directories")
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AssetValidationError(f"{description} escapes the asset root") from exc
    if not candidate.is_file():
        raise AssetValidationError(f"{description} is not a file: {relative}")
    if candidate.suffix.lower() in _UNSAFE_SUFFIXES:
        raise AssetValidationError(f"{description} references unsafe pickle-like file {relative!r}")
    return candidate


def _validate_file_records(root: Path, export: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_records = export.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise AssetValidationError("export manifest.files must be a non-empty list")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for index, raw in enumerate(raw_records):
        description = f"export manifest.files[{index}]"
        if not isinstance(raw, dict):
            raise AssetValidationError(f"{description} must be an object")
        name = _require_nonempty_string(raw, "name", description)
        path = _contained_file(
            root,
            _require_nonempty_string(raw, "path", description),
            description,
        )
        role = _require_nonempty_string(raw, "role", description)
        expected_hash = _require_sha256(raw, "sha256", description)
        expected_size = _require_nonnegative_int(raw, "size_bytes", description)
        if name in names:
            raise AssetValidationError(f"duplicate file name {name!r}")
        if path in paths:
            raise AssetValidationError(f"duplicate file path {path.relative_to(root)!s}")
        names.add(name)
        paths.add(path)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise AssetValidationError(
                f"size mismatch for {path.name}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise AssetValidationError(
                f"SHA-256 mismatch for {path.name}: expected {expected_hash}, got {actual_hash}"
            )
        records.append({**raw, "_resolved_path": path, "_role": role})
    return tuple(records)


def _index_role_paths(records: tuple[dict[str, Any], ...]) -> dict[str, tuple[Path, ...]]:
    mutable: dict[str, list[Path]] = {}
    for record in records:
        mutable.setdefault(record["_role"], []).append(record["_resolved_path"])
    return {role: tuple(paths) for role, paths in mutable.items()}


def _single_role_path(role_paths: dict[str, tuple[Path, ...]], role: str) -> Path:
    paths = role_paths.get(role, ())
    if len(paths) != 1:
        raise AssetValidationError(
            f"export manifest must contain exactly one file with role {role!r}"
        )
    return paths[0]


def _cross_check_file_record(
    records: tuple[dict[str, Any], ...], path: Path, expected_hash: str, expected_size: int
) -> None:
    record = next((record for record in records if record["_resolved_path"] == path), None)
    if record is None:
        raise AssetValidationError(f"file missing from export inventory: {path.name}")
    if record["sha256"].lower() != expected_hash or record["size_bytes"] != expected_size:
        raise AssetValidationError("weights metadata disagrees between the two manifests")


def _parse_tensor_records(manifest: dict[str, Any]) -> tuple[TensorRecord, ...]:
    rows = manifest.get("tensors")
    if not isinstance(rows, list):
        raise AssetValidationError("tensor manifest.tensors must be a list")
    records: list[TensorRecord] = []
    reference_names: set[str] = set()
    mlx_keys: set[str] = set()
    for index, row in enumerate(rows):
        description = f"tensor manifest.tensors[{index}]"
        if not isinstance(row, dict):
            raise AssetValidationError(f"{description} must be an object")
        reference_name = _require_nonempty_string(row, "reference_name", description)
        mlx_key = _require_nonempty_string(row, "mlx_key", description)
        component = _require_nonempty_string(row, "component", description)
        transform = _require_nonempty_string(row, "transform", description)
        if component.strip().lower() in _UNCLASSIFIED_VALUES:
            raise AssetValidationError(f"{description}.component is unclassified")
        if mlx_key.strip().lower() in _UNCLASSIFIED_VALUES:
            raise AssetValidationError(f"{description}.mlx_key is unclassified")
        if component not in COMPONENTS:
            raise AssetValidationError(f"{description}.component is unsupported: {component!r}")
        if transform not in TRANSFORMS:
            raise AssetValidationError(
                f"{description}.transform is {transform!r}; locked checkpoint expects 'none'"
            )
        try:
            canonical = map_reference_key(reference_name)
        except WeightMappingError as exc:
            raise AssetValidationError(f"{description} is unclassified: {exc}") from exc
        if (
            canonical.component != component
            or canonical.mlx_key != mlx_key
            or canonical.transform != transform
        ):
            raise AssetValidationError(
                f"{description} classification disagrees with the canonical weight map"
            )
        if reference_name in reference_names:
            raise AssetValidationError(f"duplicate reference tensor key {reference_name!r}")
        if mlx_key in mlx_keys:
            raise AssetValidationError(f"duplicate MLX tensor key {mlx_key!r}")
        reference_names.add(reference_name)
        mlx_keys.add(mlx_key)

        raw_shape = row.get("shape")
        if not isinstance(raw_shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in raw_shape
        ):
            raise AssetValidationError(
                f"{description}.shape must be non-negative integer dimensions"
            )
        shape = tuple(raw_shape)
        numel = _require_nonnegative_int(row, "numel", description)
        if numel != math.prod(shape):
            raise AssetValidationError(
                f"{description}.numel is {numel}, but shape {shape} "
                f"contains {math.prod(shape)} values"
            )
        dtype = _normalize_dtype(_require_nonempty_string(row, "dtype", description), description)
        records.append(
            TensorRecord(
                reference_name=reference_name,
                mlx_key=mlx_key,
                component=component,
                transform=transform,
                shape=shape,
                dtype=dtype,
                numel=numel,
            )
        )
    if [record.reference_name for record in records] != sorted(reference_names):
        raise AssetValidationError("tensor manifest rows must be sorted by reference_name")
    return tuple(records)


def _normalize_dtype(value: str, description: str) -> str:
    normalized = _DTYPE_ALIASES.get(value, _DTYPE_ALIASES.get(value.lower()))
    if normalized is None:
        raise AssetValidationError(f"{description}.dtype is unsupported: {value!r}")
    return normalized


def _validate_safetensors_header(path: Path, records: tuple[TensorRecord, ...]) -> None:
    expected = {record.reference_name: record for record in records}
    try:
        with safe_open(path, framework="np") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != set(expected):
                missing = sorted(set(expected) - actual_keys)
                unexpected = sorted(actual_keys - set(expected))
                raise AssetValidationError(
                    "safe weights keys disagree with manifest; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for key, record in expected.items():
                tensor_slice = handle.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape())
                actual_dtype = _normalize_dtype(
                    tensor_slice.get_dtype(), f"safe weights tensor {key}"
                )
                if actual_shape != record.shape:
                    raise AssetValidationError(
                        f"shape mismatch for {key}: manifest {record.shape}, weights {actual_shape}"
                    )
                if actual_dtype != record.dtype:
                    raise AssetValidationError(
                        f"dtype mismatch for {key}: manifest {record.dtype}, weights {actual_dtype}"
                    )
    except AssetValidationError:
        raise
    except (OSError, SafetensorError) as exc:
        raise AssetValidationError(f"cannot inspect safe weights: {exc}") from exc


def _validate_loaded_arrays(
    arrays: dict[str, np.ndarray[Any, Any]], records: tuple[TensorRecord, ...]
) -> None:
    expected = {record.reference_name: record for record in records}
    if set(arrays) != set(expected):
        raise AssetValidationError("loaded safe weights keys disagree with validated inventory")
    for key, array in arrays.items():
        record = expected[key]
        shape_changed = tuple(array.shape) != record.shape
        dtype_changed = _normalize_dtype(str(array.dtype), key) != record.dtype
        if shape_changed or dtype_changed:
            raise AssetValidationError(f"loaded safe tensor metadata changed for {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AssetValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()
