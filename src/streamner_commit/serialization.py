"""Deterministic, failure-safe Parquet persistence for Phase 11 traces.

The trace writer deliberately keeps policy evaluation out of the model-running
boundary.  A completed directory is immutable: it can be reused only after its
fingerprint, schemas, row counts, and file digests have all been verified.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from streamner_commit.streaming.replay import replay_span_updates
from streamner_commit.types import (
    ColdFullResult,
    GoldEntity,
    PublicEntity,
    SnapshotStep,
    SpanBoundary,
    SpanScoreUpdate,
)

TRACE_FORMAT_VERSION = 1
PARQUET_FILENAMES = (
    "examples.parquet",
    "steps.parquet",
    "span_updates.parquet",
    "snapshots.parquet",
    "cold_full.parquet",
)
OPTIONAL_GOLD_FILENAME = "gold_entities.parquet"


class TraceSerializationError(ValueError):
    """A trace cannot be safely serialized or verified."""


class TraceIntegrityError(TraceSerializationError):
    """A persisted trace fails its immutable integrity contract."""


class IncompleteTraceRunError(TraceIntegrityError):
    """A path looks like a trace run but has no valid complete manifest."""


def _list_type(value_type: pa.DataType) -> pa.ListType:
    return pa.list_(pa.field("element", value_type, nullable=False))


PUBLIC_ENTITY_TYPE = pa.struct(
    [
        pa.field("start_char", pa.int64(), nullable=False),
        pa.field("end_char", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)

RAW_SPAN_STATE_TYPE = pa.struct(
    [
        pa.field("start_word", pa.int64(), nullable=False),
        pa.field("end_word", pa.int64(), nullable=False),
        pa.field("logits", _list_type(pa.float64()), nullable=False),
    ]
)

EXAMPLES_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("labels", _list_type(pa.string()), nullable=False),
        pa.field("final_state_sha256", pa.string(), nullable=False),
        pa.field("task_name", pa.string()),
        pa.field("uid", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("parent_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_uid", pa.string()),
        pa.field("sentence_index", pa.int64()),
        pa.field("language", pa.string()),
        pa.field("metadata_sha256", pa.string()),
    ]
)

STEPS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("step", pa.int64(), nullable=False),
        pa.field("chunk", pa.string(), nullable=False),
        pa.field("accumulated_text", pa.string(), nullable=False),
        pa.field("visible_char_count", pa.int64(), nullable=False),
        pa.field("visible_word_count", pa.int64(), nullable=False),
        pa.field("elapsed_ms", pa.float64(), nullable=False),
    ]
)

SPAN_UPDATES_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("step", pa.int64(), nullable=False),
        pa.field("chunk", pa.string(), nullable=False),
        pa.field("visible_char_count", pa.int64(), nullable=False),
        pa.field("visible_word_count", pa.int64(), nullable=False),
        pa.field("start_word", pa.int64(), nullable=False),
        pa.field("end_word", pa.int64(), nullable=False),
        pa.field("start_char", pa.int64(), nullable=False),
        pa.field("end_char", pa.int64(), nullable=False),
        pa.field("span_text", pa.string(), nullable=False),
        pa.field("logits", _list_type(pa.float64()), nullable=False),
        pa.field("probs", _list_type(pa.float64()), nullable=False),
        pa.field("top_label_index", pa.int64(), nullable=False),
        pa.field("top_label", pa.string(), nullable=False),
        pa.field("top_probability", pa.float64(), nullable=False),
        pa.field("second_probability", pa.float64(), nullable=False),
        pa.field("label_margin", pa.float64(), nullable=False),
        pa.field("previous_top_probability", pa.float64()),
        pa.field("top_probability_delta", pa.float64()),
        pa.field("update_kind", pa.string(), nullable=False),
        pa.field("tail_distance_words", pa.int64(), nullable=False),
    ]
)

SNAPSHOTS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("step", pa.int64(), nullable=False),
        pa.field("entity_index", pa.int64(), nullable=False),
        pa.field("start_char", pa.int64(), nullable=False),
        pa.field("end_char", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)

COLD_FULL_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("full_text", pa.string(), nullable=False),
        pa.field("public_entities", _list_type(PUBLIC_ENTITY_TYPE), nullable=False),
        pa.field("raw_final_span_state", _list_type(RAW_SPAN_STATE_TYPE), nullable=False),
    ]
)

GOLD_ENTITIES_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("entity_index", pa.int64(), nullable=False),
        pa.field("start_char", pa.int64(), nullable=False),
        pa.field("end_char", pa.int64(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]
)

TRACE_SCHEMAS: Mapping[str, pa.Schema] = MappingProxyType(
    {
        "examples.parquet": EXAMPLES_SCHEMA,
        "steps.parquet": STEPS_SCHEMA,
        "span_updates.parquet": SPAN_UPDATES_SCHEMA,
        "snapshots.parquet": SNAPSHOTS_SCHEMA,
        "cold_full.parquet": COLD_FULL_SCHEMA,
        "gold_entities.parquet": GOLD_ENTITIES_SCHEMA,
    }
)

_EXAMPLE_OPTIONAL_FIELDS = (
    "task_name",
    "uid",
    "source_row_index",
    "parent_id",
    "split",
    "source_dataset",
    "source_uid",
    "sentence_index",
    "language",
    "metadata_sha256",
)
_TREE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-reference",
        "__pycache__",
        "artifacts",
        "data",
        "results",
    }
)
_RUNTIME_VERSION_FIELDS = (
    "mlx_version",
    "mlx_lm_version",
    "transformers_version",
    "torch_version",
    "gliner_version",
)


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceSerializationError(f"{name} must be a nonblank string")
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name=name)


def _optional_nonnegative_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceSerializationError(f"{name} must be a nonnegative integer or null")
    return value


def _optional_sha256(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TraceSerializationError(f"{name} must be a lowercase SHA-256 digest or null")
    return value


def _optional_bool(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TraceSerializationError(f"{name} must be a boolean or null")
    return value


def _optional_probability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TraceSerializationError(f"{name} must be a probability or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TraceSerializationError(f"{name} must be a probability or null")
    return result


def _json_safe(value: object, *, path: str = "value") -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TraceSerializationError(f"{path} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TraceSerializationError(f"{path} keys must be strings")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TraceSerializationError(f"{path} is not canonical JSON data: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite JSON data with one deterministic, UTF-8 representation."""

    normalized = _json_safe(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 digest of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def warm_span_state_sha256(
    state: Mapping[SpanBoundary, Sequence[float]],
) -> str:
    """Hash a replayed warm logit map in the generator's canonical form."""

    rows: list[dict[str, object]] = []
    for boundary, values in sorted(state.items()):
        if not isinstance(boundary, SpanBoundary):
            raise TraceSerializationError("warm state keys must be SpanBoundary values")
        logits = tuple(float(value) for value in values)
        if not logits or not all(math.isfinite(value) for value in logits):
            raise TraceSerializationError("warm state vectors must be nonempty and finite")
        rows.append(
            {
                "start_word": boundary.start_word,
                "end_word": boundary.end_word,
                "logits": logits,
            }
        )
    return canonical_sha256(rows)


def source_tree_sha256(project_root: str | Path) -> str:
    """Hash an unborn repository's relevant files without hashing run outputs.

    Relative paths and contents are included.  Timestamps and filesystem
    enumeration order are not, and symlinks are represented by their target
    text rather than followed outside the repository.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise TraceSerializationError("project_root must be an existing directory")
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if not any(part in _TREE_EXCLUDED_PARTS for part in path.relative_to(root).parts)
            and (path.is_file() or path.is_symlink())
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            kind = b"L"
            content_digest = hashlib.sha256(os.readlink(path).encode("utf-8")).digest()
        else:
            kind = b"F"
            content_digest = bytes.fromhex(_sha256_file(path))
        digest.update(kind)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def project_version(project_root: str | Path) -> tuple[str | None, str | None]:
    """Return ``(git_commit, source_tree_fallback)`` for trace identity.

    A clean real HEAD is preferred.  Unborn repositories and committed working
    trees with source changes also include a deterministic source-tree digest;
    final evaluation rejects traces that are not exact to the commit.  Generated
    data, artifacts, and results are outside this source identity.
    """

    root = Path(project_root).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None, source_tree_sha256(root)
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise TraceSerializationError("git returned an invalid HEAD commit")
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                ".",
                ":(exclude)artifacts/**",
                ":(exclude)data/**",
                ":(exclude)results/**",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise TraceSerializationError("cannot verify Git source-tree state") from error
    return (commit, source_tree_sha256(root)) if status.stdout.strip() else (commit, None)


@dataclass(frozen=True, slots=True)
class TraceExample:
    """One persisted trace input and its fixed session label order."""

    example_id: str
    text: str
    labels: tuple[str, ...]
    final_state_sha256: str | None = None
    task_name: str | None = None
    uid: str | None = None
    source_row_index: int | None = None
    parent_id: str | None = None
    split: str | None = None
    source_dataset: str | None = None
    source_uid: str | None = None
    sentence_index: int | None = None
    language: str | None = None
    metadata_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.example_id, name="example_id")
        if not isinstance(self.text, str):
            raise TraceSerializationError("text must be a string")
        if isinstance(self.labels, str | bytes) or not isinstance(self.labels, Sequence):
            raise TraceSerializationError("labels must be an ordered sequence")
        labels = tuple(
            _identifier(label, name=f"labels[{index}]")
            for index, label in enumerate(self.labels)
        )
        if not labels:
            raise TraceSerializationError("labels must not be empty")
        if len(labels) != len(set(labels)):
            raise TraceSerializationError("labels must not contain duplicates")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "final_state_sha256",
            _optional_sha256(self.final_state_sha256, name="final_state_sha256"),
        )
        for name in (
            "task_name",
            "uid",
            "parent_id",
            "split",
            "source_dataset",
            "source_uid",
            "language",
        ):
            object.__setattr__(
                self,
                name,
                _optional_string(getattr(self, name), name=name),
            )
        for name in ("source_row_index", "sentence_index"):
            object.__setattr__(
                self,
                name,
                _optional_nonnegative_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "metadata_sha256",
            _optional_sha256(self.metadata_sha256, name="metadata_sha256"),
        )

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self, *, run_id: str | None = None) -> dict[str, object]:
        row: dict[str, object] = {
            "example_id": self.example_id,
            "text": self.text,
            "labels": list(self.labels),
            "final_state_sha256": self.final_state_sha256,
        }
        if run_id is not None:
            row["run_id"] = run_id
        for name in _EXAMPLE_OPTIONAL_FIELDS:
            row[name] = getattr(self, name)
        return row

    def fingerprint_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "uid": self.uid or self.example_id,
            "source_row_index": self.source_row_index,
            "task_name": self.task_name,
            "text_sha256": self.text_sha256,
            "labels": list(self.labels),
        }


def _value_mapping(value: object, *, name: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        converted = dataclasses.asdict(value)
        if isinstance(converted, dict):
            return converted
    raise TraceSerializationError(f"{name} must be a mapping or dataclass-like record")


def _trace_example(value: object) -> TraceExample:
    if isinstance(value, TraceExample):
        return value
    row = _value_mapping(value, name="example")
    row.pop("run_id", None)
    row.pop("gold_entities", None)
    metadata = row.pop("metadata", None)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TraceSerializationError("example metadata must be a mapping")
        for key, item in metadata.items():
            if key not in _EXAMPLE_OPTIONAL_FIELDS:
                raise TraceSerializationError(f"unsupported example metadata field: {key}")
            if key in row and row[key] != item:
                raise TraceSerializationError(f"conflicting example metadata field: {key}")
            row[key] = item
    allowed = {
        "example_id",
        "text",
        "labels",
        "final_state_sha256",
        *_EXAMPLE_OPTIONAL_FIELDS,
    }
    extra = sorted(set(row) - allowed)
    if extra:
        raise TraceSerializationError(f"unsupported example fields: {extra}")
    missing = sorted({"example_id", "text", "labels"} - set(row))
    if missing:
        raise TraceSerializationError(f"missing example fields: {missing}")
    return TraceExample(**row)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TraceFingerprint:
    """Canonical immutable identity of all model-running inputs."""

    run_id: str
    sha256: str
    payload: Mapping[str, object] = field(hash=False)

    def __post_init__(self) -> None:
        _identifier(self.run_id, name="run_id")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise TraceSerializationError("fingerprint sha256 must be lowercase hexadecimal")
        payload = _json_safe(self.payload, path="fingerprint")
        if canonical_sha256(payload) != self.sha256:
            raise TraceSerializationError("fingerprint digest does not match its payload")
        if self.run_id != f"trace-{self.sha256[:24]}":
            raise TraceSerializationError("run_id does not match the fingerprint digest")
        object.__setattr__(self, "payload", MappingProxyType(payload))

    def to_dict(self) -> dict[str, object]:
        return dict(_json_safe(self.payload, path="fingerprint"))


def build_trace_fingerprint(
    *,
    examples: Sequence[object],
    project_root: str | Path,
    model_sha: str,
    backend: str,
    dtype: str,
    chunk_strategy: str,
    chunk_words: int,
    model_config: Mapping[str, object],
    device: str | None = None,
    runtime_versions: Mapping[str, str | None] | None = None,
    model_id: str | None = None,
    checkpoint_config_sha256: str | None = None,
    public_threshold: float | None = None,
    flat_ner: bool | None = None,
    multi_label: bool | None = None,
    max_width: int | None = None,
    right_context_width: int | None = None,
    gliner_reference_tag: str | None = None,
    gliner_reference_commit: str | None = None,
    platform_name: str | None = None,
    machine_architecture: str | None = None,
    python_version: str | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
    dataset_subset: str | None = None,
    dataset_tasks: Sequence[str] = (),
    sample_manifest_sha256: str | None = None,
    random_seed: int = 0,
    extra_inputs: Mapping[str, object] | None = None,
) -> TraceFingerprint:
    """Build the deterministic run identity used for exact trace reuse."""

    normalized_examples = tuple(
        sorted((_trace_example(value) for value in examples), key=lambda value: value.example_id)
    )
    if not normalized_examples:
        raise TraceSerializationError("a trace fingerprint needs at least one example")
    if len({example.example_id for example in normalized_examples}) != len(normalized_examples):
        raise TraceSerializationError("fingerprint example IDs must be unique")
    if isinstance(chunk_words, bool) or not isinstance(chunk_words, int) or chunk_words <= 0:
        raise TraceSerializationError("chunk_words must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise TraceSerializationError("random_seed must be a nonnegative integer")
    normalized_model_config = _json_safe(model_config, path="model_config")
    assert isinstance(normalized_model_config, dict)
    normalized_extra_inputs = _json_safe(extra_inputs or {}, path="extra_inputs")
    assert isinstance(normalized_extra_inputs, dict)
    if device is None and isinstance(normalized_extra_inputs.get("device"), str):
        device = normalized_extra_inputs["device"]
    if public_threshold is None:
        raw_threshold = normalized_model_config.get("public_threshold")
        if isinstance(raw_threshold, int | float) and not isinstance(raw_threshold, bool):
            public_threshold = float(raw_threshold)
    if flat_ner is None and isinstance(normalized_model_config.get("flat_ner"), bool):
        flat_ner = normalized_model_config["flat_ner"]
    if multi_label is None and isinstance(normalized_model_config.get("multi_label"), bool):
        multi_label = normalized_model_config["multi_label"]
    if max_width is None:
        raw_max_width = normalized_model_config.get("max_width")
        if isinstance(raw_max_width, int) and not isinstance(raw_max_width, bool):
            max_width = raw_max_width
    if right_context_width is None:
        raw_right_context = normalized_model_config.get("right_context_width")
        if isinstance(raw_right_context, int) and not isinstance(raw_right_context, bool):
            right_context_width = raw_right_context

    normalized_max_width = _optional_nonnegative_int(max_width, name="max_width")
    if normalized_max_width == 0:
        raise TraceSerializationError("max_width must be positive or null")
    normalized_right_context = _optional_nonnegative_int(
        right_context_width,
        name="right_context_width",
    )
    if runtime_versions is None:
        runtime_versions = {}
    if not isinstance(runtime_versions, Mapping):
        raise TraceSerializationError("runtime_versions must be a mapping")
    extra_runtime_fields = sorted(set(runtime_versions) - set(_RUNTIME_VERSION_FIELDS))
    if extra_runtime_fields:
        raise TraceSerializationError(
            f"unsupported runtime version fields: {extra_runtime_fields}"
        )
    normalized_runtime_versions = {
        name: _optional_string(runtime_versions.get(name), name=name)
        for name in _RUNTIME_VERSION_FIELDS
    }
    commit, tree_digest = project_version(project_root)
    model_config_sha256 = canonical_sha256(normalized_model_config)
    run_config: dict[str, object] = {
        "platform": _identifier(platform_name or platform.system(), name="platform"),
        "machine_architecture": _identifier(
            machine_architecture or platform.machine(),
            name="machine_architecture",
        ),
        "python_version": _identifier(
            python_version or platform.python_version() or sys.version.split()[0],
            name="python_version",
        ),
        "backend": _identifier(backend, name="backend"),
        "device": _optional_string(device, name="device"),
        "dtype": _identifier(dtype, name="dtype"),
        **normalized_runtime_versions,
        "model_id": _optional_string(model_id, name="model_id"),
        "model_revision_sha": _identifier(model_sha, name="model_sha"),
        "checkpoint_config_sha256": _optional_sha256(
            checkpoint_config_sha256,
            name="checkpoint_config_sha256",
        )
        or model_config_sha256,
        "model_config_sha256": model_config_sha256,
        "gliner_reference_tag": _optional_string(
            gliner_reference_tag,
            name="gliner_reference_tag",
        ),
        "gliner_reference_commit": _optional_string(
            gliner_reference_commit,
            name="gliner_reference_commit",
        ),
        "public_threshold": _optional_probability(
            public_threshold,
            name="public_threshold",
        ),
        "flat_ner": _optional_bool(flat_ner, name="flat_ner"),
        "multi_label": _optional_bool(multi_label, name="multi_label"),
        "max_width": normalized_max_width,
        "right_context_width": normalized_right_context,
        "chunk_strategy": _identifier(chunk_strategy, name="chunk_strategy"),
        "chunk_words": chunk_words,
        "dataset_id": _optional_string(dataset_id, name="dataset_id"),
        "dataset_revision": _optional_string(dataset_revision, name="dataset_revision"),
        "dataset_subset": _optional_string(dataset_subset, name="dataset_subset"),
        "dataset_tasks": sorted(
            _identifier(task, name=f"dataset_tasks[{index}]")
            for index, task in enumerate(dataset_tasks)
        ),
        "sample_manifest_sha256": _optional_sha256(
            sample_manifest_sha256,
            name="sample_manifest_sha256",
        ),
        "random_seed": random_seed,
        "project_git_commit": commit,
        "source_tree_sha256": tree_digest,
    }
    payload: dict[str, object] = {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "model_sha": _identifier(model_sha, name="model_sha"),
        "backend": _identifier(backend, name="backend"),
        "dtype": _identifier(dtype, name="dtype"),
        "chunk_strategy": _identifier(chunk_strategy, name="chunk_strategy"),
        "chunk_words": chunk_words,
        "examples": [example.fingerprint_dict() for example in normalized_examples],
        "model_config": normalized_model_config,
        "run_config": run_config,
        "project_git_commit": commit,
        "source_tree_sha256": tree_digest,
        "dataset_id": _optional_string(dataset_id, name="dataset_id"),
        "dataset_revision": _optional_string(dataset_revision, name="dataset_revision"),
        "dataset_subset": _optional_string(dataset_subset, name="dataset_subset"),
        "dataset_tasks": sorted(
            _identifier(task, name=f"dataset_tasks[{index}]")
            for index, task in enumerate(dataset_tasks)
        ),
        "sample_manifest_sha256": _optional_sha256(
            sample_manifest_sha256,
            name="sample_manifest_sha256",
        ),
        "random_seed": random_seed,
        "extra_inputs": normalized_extra_inputs,
    }
    digest = canonical_sha256(payload)
    return TraceFingerprint(
        run_id=f"trace-{digest[:24]}",
        sha256=digest,
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class TraceRunData:
    """In-memory whole-run contract accepted by the persistence boundary."""

    examples: Sequence[object]
    steps: Sequence[object]
    span_updates: Sequence[object]
    cold_full: Sequence[object]
    gold_entities: Sequence[object] | None = None

    def __post_init__(self) -> None:
        for name in ("examples", "steps", "span_updates", "cold_full"):
            value = getattr(self, name)
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise TraceSerializationError(f"{name} must be a sequence")
            object.__setattr__(self, name, tuple(value))
        if self.gold_entities is not None:
            if isinstance(self.gold_entities, str | bytes) or not isinstance(
                self.gold_entities, Sequence
            ):
                raise TraceSerializationError("gold_entities must be a sequence or null")
            object.__setattr__(self, "gold_entities", tuple(self.gold_entities))

    @classmethod
    def from_generated(
        cls,
        traces: Sequence[object],
        *,
        include_gold: bool = True,
    ) -> TraceRunData:
        """Flatten ``GeneratedExampleTrace``-like values without nested JSON."""

        if isinstance(traces, str | bytes) or not isinstance(traces, Sequence):
            raise TraceSerializationError("generated traces must be a sequence")
        examples: list[dict[str, object]] = []
        steps: list[object] = []
        updates: list[object] = []
        cold_results: list[object] = []
        gold_entities: list[object] = []
        for index, trace in enumerate(traces):
            source = getattr(trace, "example", None)
            final_digest = getattr(trace, "final_state_sha256", None)
            snapshots = getattr(trace, "snapshots", None)
            span_updates = getattr(trace, "span_updates", None)
            cold_full = getattr(trace, "cold_full", None)
            if source is None or snapshots is None or span_updates is None or cold_full is None:
                raise TraceSerializationError(
                    f"generated traces[{index}] does not expose the trace contract"
                )
            source_row = _value_mapping(source, name=f"generated traces[{index}].example")
            source_row["final_state_sha256"] = final_digest
            examples.append(source_row)
            if not isinstance(snapshots, Sequence) or isinstance(snapshots, str | bytes):
                raise TraceSerializationError("generated snapshots must be a sequence")
            if not isinstance(span_updates, Sequence) or isinstance(span_updates, str | bytes):
                raise TraceSerializationError("generated span_updates must be a sequence")
            steps.extend(snapshots)
            updates.extend(span_updates)
            cold_results.append(cold_full)
            if include_gold:
                source_gold = getattr(source, "gold_entities", None)
                if not isinstance(source_gold, Sequence) or isinstance(source_gold, str | bytes):
                    raise TraceSerializationError("generated gold_entities must be a sequence")
                gold_entities.extend(source_gold)
        return cls(
            examples=examples,
            steps=steps,
            span_updates=updates,
            cold_full=cold_results,
            gold_entities=gold_entities if include_gold else None,
        )


@dataclass(frozen=True, slots=True)
class TraceRun:
    """A verified completed run plus reconstructed immutable records."""

    path: Path
    manifest: Mapping[str, object] = field(hash=False)
    fingerprint: TraceFingerprint
    data: TraceRunData


@dataclass(frozen=True, slots=True)
class _NormalizedData:
    data: TraceRunData
    tables: Mapping[str, pa.Table]


def _public_entity(value: object) -> PublicEntity:
    if isinstance(value, PublicEntity):
        return value
    return PublicEntity(**_value_mapping(value, name="public entity"))  # type: ignore[arg-type]


def _snapshot_step(value: object, *, run_id: str) -> SnapshotStep:
    if isinstance(value, SnapshotStep):
        snapshot = value
    else:
        row = _value_mapping(value, name="snapshot step")
        existing_run_id = row.setdefault("run_id", run_id)
        if existing_run_id != run_id:
            raise TraceSerializationError("snapshot run_id differs from the fingerprint")
        entities = row.get("public_entities")
        if not isinstance(entities, Sequence) or isinstance(entities, str | bytes):
            raise TraceSerializationError("snapshot public_entities must be a sequence")
        row["public_entities"] = tuple(_public_entity(item) for item in entities)
        snapshot = SnapshotStep(**row)  # type: ignore[arg-type]
    if snapshot.run_id != run_id:
        raise TraceSerializationError("snapshot run_id differs from the fingerprint")
    if not snapshot.chunk:
        raise TraceSerializationError("persisted append steps must have a nonempty chunk")
    return snapshot


def _span_update(value: object, *, run_id: str) -> SpanScoreUpdate:
    if isinstance(value, SpanScoreUpdate):
        update = value
    else:
        row = _value_mapping(value, name="span update")
        existing_run_id = row.setdefault("run_id", run_id)
        if existing_run_id != run_id:
            raise TraceSerializationError("span update run_id differs from the fingerprint")
        update = SpanScoreUpdate(**row)  # type: ignore[arg-type]
    if update.run_id != run_id:
        raise TraceSerializationError("span update run_id differs from the fingerprint")
    if update.update_kind == "full":
        raise TraceSerializationError("full events belong in cold_full, not warm span_updates")
    return update


def _cold_result(value: object) -> ColdFullResult:
    if isinstance(value, ColdFullResult):
        return value
    row = _value_mapping(value, name="cold full result")
    row.pop("run_id", None)
    entities = row.get("public_entities")
    if not isinstance(entities, Sequence) or isinstance(entities, str | bytes):
        raise TraceSerializationError("cold public_entities must be a sequence")
    row["public_entities"] = tuple(_public_entity(item) for item in entities)
    raw_state = row.get("raw_final_span_state")
    if isinstance(raw_state, Sequence) and not isinstance(raw_state, str | bytes):
        converted: dict[SpanBoundary, Sequence[float]] = {}
        for item in raw_state:
            state_row = _value_mapping(item, name="cold raw span state")
            boundary = SpanBoundary(
                start_word=state_row["start_word"],  # type: ignore[arg-type]
                end_word=state_row["end_word"],  # type: ignore[arg-type]
            )
            if boundary in converted:
                raise TraceSerializationError("duplicate cold raw span boundary")
            logits = state_row.get("logits")
            if not isinstance(logits, Sequence) or isinstance(logits, str | bytes):
                raise TraceSerializationError("cold raw span logits must be a sequence")
            converted[boundary] = logits  # type: ignore[assignment]
        row["raw_final_span_state"] = converted
    return ColdFullResult(**row)  # type: ignore[arg-type]


def _gold_entity(value: object) -> GoldEntity:
    if isinstance(value, GoldEntity):
        return value
    row = _value_mapping(value, name="gold entity")
    row.pop("run_id", None)
    row.pop("entity_index", None)
    return GoldEntity(**row)  # type: ignore[arg-type]


def _entity_row(entity: PublicEntity) -> dict[str, object]:
    return entity.to_dict()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _normalise_data(fingerprint: TraceFingerprint, raw: TraceRunData) -> _NormalizedData:
    run_id = fingerprint.run_id
    examples = tuple(
        sorted((_trace_example(value) for value in raw.examples), key=lambda row: row.example_id)
    )
    if not examples:
        raise TraceSerializationError("trace data must contain at least one example")
    example_by_id = {example.example_id: example for example in examples}
    if len(example_by_id) != len(examples):
        raise TraceSerializationError("trace data example IDs must be unique")
    if any(example.final_state_sha256 is None for example in examples):
        raise TraceSerializationError(
            "every persisted example needs its generated final_state_sha256"
        )
    expected_examples = fingerprint.payload.get("examples")
    actual_examples = [example.fingerprint_dict() for example in examples]
    if actual_examples != expected_examples:
        raise TraceSerializationError("trace examples differ from fingerprint inputs")

    task_labels: dict[str, tuple[str, ...]] = {}
    for example in examples:
        if example.task_name is None:
            continue
        prior = task_labels.setdefault(example.task_name, example.labels)
        if prior != example.labels:
            raise TraceSerializationError("examples in one task must use one fixed label order")

    steps = tuple(
        sorted(
            (_snapshot_step(value, run_id=run_id) for value in raw.steps),
            key=lambda row: (row.example_id, row.step),
        )
    )
    step_by_key: dict[tuple[str, int], SnapshotStep] = {}
    steps_by_example: dict[str, list[SnapshotStep]] = {}
    for step in steps:
        if step.example_id not in example_by_id:
            raise TraceSerializationError("snapshot refers to an unknown example")
        key = step.example_id, step.step
        if key in step_by_key:
            raise TraceSerializationError("duplicate snapshot step")
        step_by_key[key] = step
        labels = example_by_id[step.example_id].labels
        if any(entity.label not in labels for entity in step.public_entities):
            raise TraceSerializationError("snapshot entity label is outside the session labels")
        steps_by_example.setdefault(step.example_id, []).append(step)

    for example in examples:
        example_steps = steps_by_example.get(example.example_id, [])
        if [step.step for step in example_steps] != list(range(1, len(example_steps) + 1)):
            raise TraceSerializationError("snapshot steps must be contiguous and one-based")
        if example_steps:
            if example_steps[-1].accumulated_text != example.text:
                raise TraceSerializationError("final warm text differs from source text")
            previous = ""
            for step in example_steps:
                if step.accumulated_text != previous + step.chunk:
                    raise TraceSerializationError("snapshot chunks do not exactly reconstruct text")
                previous = step.accumulated_text
        elif example.text:
            raise TraceSerializationError("a nonempty example must have at least one append step")

    updates = tuple(
        sorted(
            (_span_update(value, run_id=run_id) for value in raw.span_updates),
            key=lambda row: (row.example_id, row.step, row.start_word, row.end_word),
        )
    )
    update_keys: set[tuple[str, int, int, int]] = set()
    updates_by_example: dict[str, list[SpanScoreUpdate]] = {}
    for update in updates:
        matching_step = step_by_key.get((update.example_id, update.step))
        if matching_step is None:
            raise TraceSerializationError("span update has no corresponding append step")
        update_key = update.example_id, update.step, update.start_word, update.end_word
        if update_key in update_keys:
            raise TraceSerializationError("duplicate span update boundary within a step")
        update_keys.add(update_key)
        if (
            update.chunk != matching_step.chunk
            or update.visible_char_count != matching_step.visible_char_count
            or update.visible_word_count != matching_step.visible_word_count
            or matching_step.accumulated_text[update.start_char : update.end_char]
            != update.span_text
        ):
            raise TraceSerializationError("span update metadata differs from its append step")
        labels = example_by_id[update.example_id].labels
        if len(update.logits) != len(labels) or update.top_label != labels[update.top_label_index]:
            raise TraceSerializationError(
                "span update vector or top label differs from label order"
            )
        updates_by_example.setdefault(update.example_id, []).append(update)

    for example in examples:
        example_updates = tuple(updates_by_example.get(example.example_id, ()))
        replayed = replay_span_updates(example_updates)
        replay_digest = warm_span_state_sha256(replayed)
        if replay_digest != example.final_state_sha256:
            raise TraceSerializationError(
                f"replayed warm state differs from final_state_sha256 for {example.example_id}"
            )

    cold_results = tuple(
        sorted((_cold_result(value) for value in raw.cold_full), key=lambda row: row.example_id)
    )
    if len(cold_results) != len(examples):
        raise TraceSerializationError("cold_full must contain exactly one row per example")
    if len({result.example_id for result in cold_results}) != len(cold_results):
        raise TraceSerializationError("cold_full example IDs must be unique")
    for result in cold_results:
        matching_example = example_by_id.get(result.example_id)
        if matching_example is None or result.full_text != matching_example.text:
            raise TraceSerializationError("cold full text differs from its source example")
        if any(entity.label not in matching_example.labels for entity in result.public_entities):
            raise TraceSerializationError("cold entity label is outside the session labels")
        if any(
            len(logits) != len(matching_example.labels)
            for logits in result.raw_final_span_state.values()
        ):
            raise TraceSerializationError("cold raw vector width differs from label order")

    gold_entities: tuple[GoldEntity, ...] | None
    if raw.gold_entities is None:
        gold_entities = None
    else:
        gold_entities = tuple(
            sorted(
                (_gold_entity(value) for value in raw.gold_entities),
                key=lambda row: (row.example_id, row.start_char, row.end_char, row.label, row.text),
            )
        )
        for entity in gold_entities:
            matching_example = example_by_id.get(entity.example_id)
            if matching_example is None:
                raise TraceSerializationError("gold entity refers to an unknown example")
            if (
                matching_example.text[entity.start_char : entity.end_char]
                != entity.text
            ):
                raise TraceSerializationError("gold entity offsets do not match source text")

    example_rows = [example.to_dict(run_id=run_id) for example in examples]
    step_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    for step in steps:
        row = step.to_dict()
        entities = row.pop("public_entities")
        step_rows.append(row)
        assert isinstance(entities, list)
        for entity_index, entity in enumerate(entities):
            assert isinstance(entity, dict)
            snapshot_rows.append(
                {
                    "run_id": run_id,
                    "example_id": step.example_id,
                    "step": step.step,
                    "entity_index": entity_index,
                    **entity,
                }
            )
    update_rows = [update.to_dict() for update in updates]
    cold_rows: list[dict[str, object]] = [
        {
            "run_id": run_id,
            "example_id": result.example_id,
            "full_text": result.full_text,
            "public_entities": [_entity_row(entity) for entity in result.public_entities],
            "raw_final_span_state": [
                {
                    "start_word": boundary.start_word,
                    "end_word": boundary.end_word,
                    "logits": list(logits),
                }
                for boundary, logits in sorted(result.raw_final_span_state.items())
            ],
        }
        for result in cold_results
    ]
    rows_by_file: dict[str, list[dict[str, object]]] = {
        "examples.parquet": example_rows,
        "steps.parquet": step_rows,
        "span_updates.parquet": update_rows,
        "snapshots.parquet": snapshot_rows,
        "cold_full.parquet": cold_rows,
    }
    if gold_entities is not None:
        gold_rows: list[dict[str, object]] = []
        per_example_index: dict[str, int] = {}
        for entity in gold_entities:
            entity_index = per_example_index.get(entity.example_id, 0)
            per_example_index[entity.example_id] = entity_index + 1
            gold_rows.append(
                {
                    "run_id": run_id,
                    "entity_index": entity_index,
                    **entity.to_dict(),
                }
            )
        rows_by_file[OPTIONAL_GOLD_FILENAME] = gold_rows

    tables = {
        name: pa.Table.from_pylist(rows, schema=TRACE_SCHEMAS[name])
        for name, rows in rows_by_file.items()
    }
    typed_data = TraceRunData(
        examples=examples,
        steps=steps,
        span_updates=updates,
        cold_full=cold_results,
        gold_entities=gold_entities,
    )
    return _NormalizedData(data=typed_data, tables=MappingProxyType(tables))


def _write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(
        table,
        path,
        compression="zstd",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _created_utc(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if not isinstance(value, str):
        raise TraceSerializationError("created_utc must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise TraceSerializationError("created_utc must be an ISO-8601 string") from error
    if parsed.tzinfo is None:
        raise TraceSerializationError("created_utc must include a timezone")
    return value


def _manifest(
    fingerprint: TraceFingerprint,
    tables: Mapping[str, pa.Table],
    staging: Path,
    *,
    created_utc: str | None,
) -> dict[str, object]:
    files: dict[str, object] = {}
    for name in sorted(tables):
        path = staging / name
        files[name] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_count": tables[name].num_rows,
            "schema_sha256": _schema_sha256(TRACE_SCHEMAS[name]),
        }
    examples = fingerprint.payload["examples"]
    assert isinstance(examples, list)
    labels_by_example = {
        str(example["example_id"]): list(example["labels"])
        for example in examples
        if isinstance(example, dict)
    }
    task_labels: dict[str, object] = {}
    for example in examples:
        if not isinstance(example, dict) or example.get("task_name") is None:
            continue
        task = str(example["task_name"])
        labels = list(example["labels"])
        if task in task_labels and task_labels[task] != labels:
            raise TraceSerializationError("one task has conflicting label orders")
        task_labels[task] = labels
    return {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "run_id": fingerprint.run_id,
        "fingerprint_sha256": fingerprint.sha256,
        "fingerprint": fingerprint.to_dict(),
        "run_config": fingerprint.payload["run_config"],
        "created_utc": _created_utc(created_utc),
        "labels_by_example": labels_by_example,
        "task_labels": task_labels,
        "files": files,
        "complete": True,
    }


def _remove_stale_staging(output_root: Path, run_id: str) -> None:
    for path in output_root.glob(f".{run_id}.staging-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def write_trace_run(
    output_root: str | Path,
    fingerprint: TraceFingerprint,
    data: TraceRunData,
    *,
    created_utc: str | None = None,
) -> TraceRun:
    """Atomically persist or exactly reuse one immutable whole-run trace."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / fingerprint.run_id
    if destination.exists():
        return read_trace_run(destination, expected_fingerprint=fingerprint)

    normalized = _normalise_data(fingerprint, data)
    _remove_stale_staging(root, fingerprint.run_id)
    staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint.run_id}.staging-", dir=root))
    try:
        for name in sorted(normalized.tables):
            path = staging / name
            _write_parquet(normalized.tables[name], path)
            _fsync_file(path)
        manifest = _manifest(
            fingerprint,
            normalized.tables,
            staging,
            created_utc=created_utc,
        )
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        _fsync_file(manifest_path)
        _fsync_directory(staging)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            shutil.rmtree(staging)
            return read_trace_run(destination, expected_fingerprint=fingerprint)
        _fsync_directory(root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return read_trace_run(destination, expected_fingerprint=fingerprint)


def _read_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise IncompleteTraceRunError(f"trace run has no manifest: {path}")
    try:
        raw = manifest_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IncompleteTraceRunError(f"trace manifest is unreadable: {path}") from error
    if not isinstance(value, dict) or value.get("complete") is not True:
        raise IncompleteTraceRunError(f"trace manifest is not complete: {path}")
    if raw != canonical_json_bytes(value) + b"\n":
        raise TraceIntegrityError("trace manifest is not canonically encoded")
    return value


def _fingerprint_from_manifest(manifest: Mapping[str, object]) -> TraceFingerprint:
    payload = manifest.get("fingerprint")
    run_id = manifest.get("run_id")
    digest = manifest.get("fingerprint_sha256")
    if (
        not isinstance(payload, Mapping)
        or not isinstance(run_id, str)
        or not isinstance(digest, str)
    ):
        raise TraceIntegrityError("manifest fingerprint fields are invalid")
    return TraceFingerprint(run_id=run_id, sha256=digest, payload=payload)


def _verify_manifest_provenance(
    manifest: Mapping[str, object],
    fingerprint: TraceFingerprint,
) -> None:
    if manifest.get("run_config") != fingerprint.payload.get("run_config"):
        raise TraceIntegrityError("manifest run_config differs from its fingerprint")
    examples = fingerprint.payload.get("examples")
    if not isinstance(examples, list):
        raise TraceIntegrityError("fingerprint examples are invalid")
    labels_by_example: dict[str, object] = {}
    task_labels: dict[str, object] = {}
    for example in examples:
        if not isinstance(example, dict):
            raise TraceIntegrityError("fingerprint example identity is invalid")
        example_id = example.get("example_id")
        labels = example.get("labels")
        task_name = example.get("task_name")
        if not isinstance(example_id, str) or not isinstance(labels, list):
            raise TraceIntegrityError("fingerprint example labels are invalid")
        labels_by_example[example_id] = labels
        if task_name is not None:
            if not isinstance(task_name, str):
                raise TraceIntegrityError("fingerprint task name is invalid")
            prior = task_labels.setdefault(task_name, labels)
            if prior != labels:
                raise TraceIntegrityError("fingerprint task labels conflict")
    if manifest.get("labels_by_example") != labels_by_example:
        raise TraceIntegrityError("manifest labels_by_example differs from fingerprint")
    if manifest.get("task_labels") != task_labels:
        raise TraceIntegrityError("manifest task_labels differs from fingerprint")


def _verified_tables(path: Path, manifest: Mapping[str, object]) -> dict[str, pa.Table]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TraceIntegrityError("manifest files must be an object")
    names = set(files)
    required = set(PARQUET_FILENAMES)
    if not required.issubset(names) or names - (required | {OPTIONAL_GOLD_FILENAME}):
        raise TraceIntegrityError("manifest Parquet file set is invalid")
    tables: dict[str, pa.Table] = {}
    for name in sorted(names):
        if not isinstance(name, str) or name not in TRACE_SCHEMAS:
            raise TraceIntegrityError("manifest contains an unsupported trace file")
        metadata = files[name]
        if not isinstance(metadata, Mapping):
            raise TraceIntegrityError(f"manifest metadata for {name} is invalid")
        file_path = path / name
        if not file_path.is_file():
            raise TraceIntegrityError(f"trace file is missing: {name}")
        if metadata.get("sha256") != _sha256_file(file_path):
            raise TraceIntegrityError(f"trace checksum differs: {name}")
        if metadata.get("size_bytes") != file_path.stat().st_size:
            raise TraceIntegrityError(f"trace size differs: {name}")
        expected_schema = TRACE_SCHEMAS[name]
        if metadata.get("schema_sha256") != _schema_sha256(expected_schema):
            raise TraceIntegrityError(f"manifest schema digest differs: {name}")
        actual_schema = pq.read_schema(file_path)
        if not actual_schema.equals(expected_schema, check_metadata=True):
            raise TraceIntegrityError(f"Parquet schema differs: {name}")
        table = pq.read_table(file_path, schema=expected_schema)
        if metadata.get("row_count") != table.num_rows:
            raise TraceIntegrityError(f"trace row count differs: {name}")
        tables[name] = table
    return tables


def _data_from_tables(run_id: str, tables: Mapping[str, pa.Table]) -> TraceRunData:
    examples = tuple(_trace_example(row) for row in tables["examples.parquet"].to_pylist())
    snapshot_entities: dict[tuple[str, int], list[tuple[int, PublicEntity]]] = {}
    for row in tables["snapshots.parquet"].to_pylist():
        entity_index = row.pop("entity_index")
        row_run_id = row.pop("run_id")
        example_id = row.pop("example_id")
        step = row.pop("step")
        if row_run_id != run_id or not isinstance(entity_index, int):
            raise TraceIntegrityError("snapshot identity is invalid")
        entity = _public_entity(row)
        snapshot_entities.setdefault((example_id, step), []).append((entity_index, entity))

    steps: list[SnapshotStep] = []
    for row in tables["steps.parquet"].to_pylist():
        key = row["example_id"], row["step"]
        indexed = sorted(snapshot_entities.pop(key, []), key=lambda item: item[0])
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise TraceIntegrityError("snapshot entity indices are not contiguous")
        row["public_entities"] = tuple(entity for _, entity in indexed)
        steps.append(_snapshot_step(row, run_id=run_id))
    if snapshot_entities:
        raise TraceIntegrityError("snapshot entities refer to missing steps")

    updates = tuple(
        _span_update(row, run_id=run_id)
        for row in tables["span_updates.parquet"].to_pylist()
    )
    cold_results = tuple(
        _cold_result(row)
        for row in tables["cold_full.parquet"].to_pylist()
    )
    gold: tuple[GoldEntity, ...] | None = None
    if OPTIONAL_GOLD_FILENAME in tables:
        gold = tuple(
            _gold_entity(row)
            for row in tables[OPTIONAL_GOLD_FILENAME].to_pylist()
        )
    return TraceRunData(
        examples=examples,
        steps=tuple(steps),
        span_updates=updates,
        cold_full=cold_results,
        gold_entities=gold,
    )


def read_trace_run(
    path: str | Path,
    *,
    expected_fingerprint: TraceFingerprint | None = None,
) -> TraceRun:
    """Verify every persisted byte/schema and reconstruct replayable records."""

    run_path = Path(path)
    if not run_path.is_dir():
        raise IncompleteTraceRunError(f"trace run directory does not exist: {run_path}")
    manifest = _read_manifest(run_path)
    if manifest.get("trace_format_version") != TRACE_FORMAT_VERSION:
        raise TraceIntegrityError("unsupported trace format version")
    fingerprint = _fingerprint_from_manifest(manifest)
    _verify_manifest_provenance(manifest, fingerprint)
    if run_path.name != fingerprint.run_id:
        raise TraceIntegrityError("trace directory name differs from run_id")
    if expected_fingerprint is not None and (
        fingerprint.run_id != expected_fingerprint.run_id
        or fingerprint.sha256 != expected_fingerprint.sha256
        or fingerprint.to_dict() != expected_fingerprint.to_dict()
    ):
        raise TraceIntegrityError("existing trace fingerprint differs from requested run")
    tables = _verified_tables(run_path, manifest)
    data = _data_from_tables(fingerprint.run_id, tables)
    normalized = _normalise_data(fingerprint, data)
    for name, table in normalized.tables.items():
        persisted = tables.get(name)
        if persisted is None or not table.equals(persisted, check_metadata=True):
            raise TraceIntegrityError(f"trace records are not canonically reconstructable: {name}")
    return TraceRun(
        path=run_path,
        manifest=MappingProxyType(manifest),
        fingerprint=fingerprint,
        data=normalized.data,
    )


__all__ = [
    "COLD_FULL_SCHEMA",
    "EXAMPLES_SCHEMA",
    "GOLD_ENTITIES_SCHEMA",
    "IncompleteTraceRunError",
    "PARQUET_FILENAMES",
    "SNAPSHOTS_SCHEMA",
    "SPAN_UPDATES_SCHEMA",
    "STEPS_SCHEMA",
    "TRACE_FORMAT_VERSION",
    "TRACE_SCHEMAS",
    "TraceExample",
    "TraceFingerprint",
    "TraceIntegrityError",
    "TraceRun",
    "TraceRunData",
    "TraceSerializationError",
    "build_trace_fingerprint",
    "canonical_json_bytes",
    "canonical_sha256",
    "project_version",
    "read_trace_run",
    "source_tree_sha256",
    "write_trace_run",
]
