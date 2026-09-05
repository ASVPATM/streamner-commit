"""Content-safe PIIMB manifest loading and in-memory trace input reconstruction.

This module is deliberately limited to the boundary between the sanitized,
checked-in benchmark manifests and the licensed PIIMB JSONL object.  It never
writes source text or annotations.  Trace generation and persistence consume
the validated in-memory records returned here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from streamner_commit.datasets.piimb import (
    DEFAULT_SPLIT_SALT,
    DEV_PERCENT,
    PIIMB_DATASET_ID,
    PIIMB_LICENSE,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SPLIT,
    PIIMB_SUBSET,
    PRIMARY_TASKS,
    SAMPLE_ALGORITHM,
    SPLIT_ALGORITHM,
    TEST_PERCENT,
    PIIMBExample,
    PIIMBPreset,
    parent_split,
    parse_piimb_row,
    resolve_preset,
)
from streamner_commit.types import GoldEntity

SplitName = Literal["dev", "test"]
_SPLITS: tuple[SplitName, ...] = ("dev", "test")

_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "sampling",
        "tasks",
        "task_labels",
        "source_diagnostics",
        "counts",
        "examples",
        "checksums",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "uid",
        "source_row_index",
        "task_name",
        "source_dataset",
        "source_uid",
        "parent_id",
        "sentence_index",
        "language",
        "metadata_sha256",
    }
)
_METADATA_FIELDS = tuple(sorted(_RECORD_FIELDS - {"metadata_sha256"}))
_SOURCE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "row_count",
        "unique_uid_count",
        "duplicate_uid_count",
        "duplicate_uid_occurrences",
        "conflicting_uid_count",
        "exact_duplicate_occurrences",
    }
)
_PINNED_PRIMARY_SOURCE_DIAGNOSTICS = {
    "row_count": 121_409,
    "unique_uid_count": 120_107,
    "duplicate_uid_count": 1_302,
    "duplicate_uid_occurrences": 1_302,
    "conflicting_uid_count": 1_227,
    "exact_duplicate_occurrences": 75,
}
_CHECKSUM_FIELDS = frozenset(
    {
        "task_labels_sha256",
        "dev_row_ids_sha256",
        "test_row_ids_sha256",
        "metadata_sha256",
        "manifest_sha256",
    }
)


class PIIMBTraceSourceError(ValueError):
    """A PIIMB trace manifest or its locked source object is inconsistent."""


@dataclass(frozen=True, slots=True)
class PIIMBTraceSourceLock:
    """Expected byte identity and row count for the pinned PIIMB JSONL object."""

    size_bytes: int
    sha256: str
    rows: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("source size_bytes must be a positive integer")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("source sha256 must be 64 lowercase hexadecimal characters")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 1:
            raise ValueError("source rows must be a positive integer")

    def dataset_manifest(self) -> dict[str, object]:
        """Return the exact dataset lock embedded in selection manifests."""

        return {
            "id": PIIMB_DATASET_ID,
            "subset": PIIMB_SUBSET,
            "revision": PIIMB_REVISION,
            "split": PIIMB_SPLIT,
            "license": PIIMB_LICENSE,
            "source_file": {
                "path": PIIMB_SOURCE_FILE,
                "size_bytes": self.size_bytes,
                "sha256": self.sha256,
                "rows": self.rows,
            },
        }


PINNED_PIIMB_TRACE_SOURCE = PIIMBTraceSourceLock(
    size_bytes=PIIMB_SOURCE_SIZE_BYTES,
    sha256=PIIMB_SOURCE_SHA256,
    rows=PIIMB_SOURCE_ROW_COUNT,
)


@dataclass(frozen=True, slots=True)
class PIIMBTraceManifestRecord:
    """One content-free manifest identity, including its deterministic split."""

    split: SplitName
    uid: str
    source_row_index: int
    task_name: str
    source_dataset: str
    source_uid: str
    parent_id: str
    sentence_index: int
    language: str
    metadata_sha256: str

    def source_metadata(self) -> dict[str, object]:
        """Return the fields covered by ``metadata_sha256``."""

        return {field: getattr(self, field) for field in _METADATA_FIELDS}

    @property
    def row_identity(self) -> tuple[int, str]:
        """Return the source identity; UIDs alone are not unique in PIIMB."""

        return self.source_row_index, self.uid


@dataclass(frozen=True, slots=True)
class PIIMBTraceManifest:
    """A fully validated smoke, research-small, or research-full manifest."""

    preset: PIIMBPreset
    task_labels: Mapping[str, tuple[str, ...]]
    records: tuple[PIIMBTraceManifestRecord, ...]
    manifest_sha256: str
    task_labels_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_labels", MappingProxyType(dict(self.task_labels)))
        object.__setattr__(self, "records", tuple(self.records))

    def records_for_split(self, split: SplitName) -> tuple[PIIMBTraceManifestRecord, ...]:
        """Return records in their checksum-covered manifest order."""

        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        return tuple(record for record in self.records if record.split == split)


@dataclass(frozen=True, slots=True)
class PIIMBTraceExample:
    """One selected source row ready for trace generation, held in memory only."""

    split: SplitName
    example_id: str
    text: str
    labels: tuple[str, ...]
    gold_entities: tuple[GoldEntity, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a nonblank string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a nonblank string")
        labels = tuple(self.labels)
        if not labels or labels != tuple(sorted(set(labels))):
            raise ValueError("labels must be nonempty, unique, and sorted")
        gold = tuple(self.gold_entities)
        for index, entity in enumerate(gold):
            if not isinstance(entity, GoldEntity):
                raise TypeError(f"gold_entities[{index}] must be a GoldEntity")
            if entity.example_id != self.example_id:
                raise ValueError(f"gold_entities[{index}] belongs to another example")
            if self.text[entity.start_char : entity.end_char] != entity.text:
                raise ValueError(f"gold_entities[{index}] does not preserve its source slice")
            if entity.label not in labels:
                raise ValueError(f"gold_entities[{index}] label is outside the task vocabulary")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "gold_entities", gold)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_negative(self) -> bool:
        """Whether the source sentence has no gold entity annotations."""

        return not self.gold_entities


@dataclass(frozen=True, slots=True)
class ReconstructedPIIMBTrace:
    """A validated manifest paired with split-ordered in-memory source records."""

    manifest: PIIMBTraceManifest
    examples: tuple[PIIMBTraceExample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, PIIMBTraceManifest):
            raise TypeError("manifest must be a PIIMBTraceManifest")
        examples = tuple(self.examples)
        if len(examples) != len(self.manifest.records):
            raise ValueError("reconstructed example count differs from the manifest")
        object.__setattr__(self, "examples", examples)


def _mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PIIMBTraceSourceError(f"{name} must be an object")
    if fields is not None and set(value) != fields:
        raise PIIMBTraceSourceError(
            f"{name} fields differ: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise PIIMBTraceSourceError(f"{name} must be a sequence")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PIIMBTraceSourceError(f"{name} must be a nonblank string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PIIMBTraceSourceError(f"{name} must be a nonnegative integer")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise PIIMBTraceSourceError(f"{name} must be 64 lowercase hexadecimal characters")
    return result


def _checksum(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_raw_fields(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = {str(key) for key in value} & {"text", "entities"}
        if forbidden:
            raise PIIMBTraceSourceError(
                f"sanitized manifest contains forbidden raw fields: {sorted(forbidden)}"
            )
        for nested in value.values():
            _reject_raw_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for nested in value:
            _reject_raw_fields(nested)


def _validate_task_labels(value: object) -> Mapping[str, tuple[str, ...]]:
    mapping = _mapping(value, name="task_labels")
    if set(mapping) != set(PRIMARY_TASKS):
        raise PIIMBTraceSourceError("task_labels must contain exactly the primary tasks")
    labels_by_task: dict[str, tuple[str, ...]] = {}
    for task in PRIMARY_TASKS:
        labels = tuple(
            _string(label, name=f"task_labels.{task}[{index}]")
            for index, label in enumerate(_sequence(mapping[task], name=f"task_labels.{task}"))
        )
        if not labels or labels != tuple(sorted(set(labels))):
            raise PIIMBTraceSourceError(
                f"task_labels.{task} must be nonempty, unique, and sorted exactly"
            )
        labels_by_task[task] = labels
    return MappingProxyType(labels_by_task)


def _parse_record(value: object, *, split: SplitName, index: int) -> PIIMBTraceManifestRecord:
    row = _mapping(value, name=f"examples.{split}[{index}]", fields=_RECORD_FIELDS)
    metadata = {field: row[field] for field in _METADATA_FIELDS}
    for field in (
        "uid",
        "task_name",
        "source_dataset",
        "source_uid",
        "parent_id",
        "language",
    ):
        _string(metadata[field], name=f"examples.{split}[{index}].{field}")
    source_row_index = _nonnegative_int(
        metadata["source_row_index"], name=f"examples.{split}[{index}].source_row_index"
    )
    sentence_index = _nonnegative_int(
        metadata["sentence_index"], name=f"examples.{split}[{index}].sentence_index"
    )
    metadata_sha256 = _sha256(
        row["metadata_sha256"], name=f"examples.{split}[{index}].metadata_sha256"
    )
    if metadata_sha256 != _checksum(metadata):
        raise PIIMBTraceSourceError(f"examples.{split}[{index}] metadata checksum differs")
    task_name = str(metadata["task_name"])
    if task_name not in PRIMARY_TASKS:
        raise PIIMBTraceSourceError(f"examples.{split}[{index}] has an unconfigured task")
    return PIIMBTraceManifestRecord(
        split=split,
        uid=str(metadata["uid"]),
        source_row_index=source_row_index,
        task_name=task_name,
        source_dataset=str(metadata["source_dataset"]),
        source_uid=str(metadata["source_uid"]),
        parent_id=str(metadata["parent_id"]),
        sentence_index=sentence_index,
        language=str(metadata["language"]),
        metadata_sha256=metadata_sha256,
    )


def _validate_diagnostics(value: object, *, require_pinned: bool) -> None:
    diagnostics = _mapping(
        value,
        name="source_diagnostics",
        fields=_SOURCE_DIAGNOSTIC_FIELDS,
    )
    for field, raw_value in diagnostics.items():
        _nonnegative_int(raw_value, name=f"source_diagnostics.{field}")
    if diagnostics["unique_uid_count"] + diagnostics["duplicate_uid_occurrences"] != diagnostics[
        "row_count"
    ]:
        raise PIIMBTraceSourceError("source_diagnostics UID counts are inconsistent")
    if diagnostics["duplicate_uid_count"] > diagnostics["duplicate_uid_occurrences"]:
        raise PIIMBTraceSourceError("source_diagnostics duplicate counts are inconsistent")
    if diagnostics["conflicting_uid_count"] > diagnostics["duplicate_uid_count"]:
        raise PIIMBTraceSourceError("source_diagnostics conflicting counts are inconsistent")
    if diagnostics["exact_duplicate_occurrences"] > diagnostics["duplicate_uid_occurrences"]:
        raise PIIMBTraceSourceError("source_diagnostics exact-duplicate counts are inconsistent")
    if require_pinned and dict(diagnostics) != _PINNED_PRIMARY_SOURCE_DIAGNOSTICS:
        raise PIIMBTraceSourceError("source_diagnostics differ from the pinned primary-task scan")


def _sampling_rank(record: PIIMBTraceManifestRecord, *, salt: str) -> tuple[str, str]:
    payload = (
        f"{salt}:sample:{record.task_name}:{record.parent_id}:"
        f"{record.uid}:{record.source_row_index}"
    )
    return hashlib.sha256(payload.encode()).hexdigest(), record.uid


def validate_piimb_trace_manifest(
    document: Mapping[str, Any],
    *,
    source_lock: PIIMBTraceSourceLock = PINNED_PIIMB_TRACE_SOURCE,
) -> PIIMBTraceManifest:
    """Validate a content-free manifest for any of the three locked presets."""

    if not isinstance(source_lock, PIIMBTraceSourceLock):
        raise TypeError("source_lock must be a PIIMBTraceSourceLock")
    root = _mapping(document, name="PIIMB trace manifest", fields=_MANIFEST_FIELDS)
    _reject_raw_fields(root)
    if root["schema_version"] != 1:
        raise PIIMBTraceSourceError("PIIMB trace manifest schema_version must be 1")
    if root["dataset"] != source_lock.dataset_manifest():
        raise PIIMBTraceSourceError("manifest dataset identity differs from the source lock")
    if root["tasks"] != list(PRIMARY_TASKS):
        raise PIIMBTraceSourceError("manifest task order differs from the primary tasks")

    sampling = _mapping(root["sampling"], name="sampling")
    preset_name = _string(sampling.get("preset"), name="sampling.preset")
    try:
        preset = resolve_preset(preset_name)
    except (TypeError, ValueError) as exc:
        raise PIIMBTraceSourceError(str(exc)) from exc
    expected_sampling = {
        "preset": preset.name,
        "split_algorithm": SPLIT_ALGORITHM,
        "sample_algorithm": SAMPLE_ALGORITHM,
        "split_salt": DEFAULT_SPLIT_SALT,
        "dev_percent": DEV_PERCENT,
        "test_percent": TEST_PERCENT,
        "requested_per_task": {
            "dev": preset.dev_per_task,
            "test": preset.test_per_task,
        },
    }
    if dict(sampling) != expected_sampling:
        raise PIIMBTraceSourceError("manifest sampling policy differs from its locked preset")

    task_labels = _validate_task_labels(root["task_labels"])
    _validate_diagnostics(
        root["source_diagnostics"],
        require_pinned=source_lock == PINNED_PIIMB_TRACE_SOURCE,
    )
    examples = _mapping(
        root["examples"], name="examples", fields=frozenset({"dev", "test"})
    )
    by_split: dict[SplitName, list[PIIMBTraceManifestRecord]] = {"dev": [], "test": []}
    records: list[PIIMBTraceManifestRecord] = []
    for split in _SPLITS:
        raw_records = _sequence(examples[split], name=f"examples.{split}")
        parsed = [
            _parse_record(value, split=split, index=index)
            for index, value in enumerate(raw_records)
        ]
        by_split[split].extend(parsed)
        records.extend(parsed)

    expected_counts = {
        "total": len(PRIMARY_TASKS) * (preset.dev_per_task + preset.test_per_task),
        "dev": len(PRIMARY_TASKS) * preset.dev_per_task,
        "test": len(PRIMARY_TASKS) * preset.test_per_task,
        "by_task": {
            task: {"dev": preset.dev_per_task, "test": preset.test_per_task}
            for task in PRIMARY_TASKS
        },
    }
    if root["counts"] != expected_counts:
        raise PIIMBTraceSourceError("manifest counts differ from its exact preset quotas")
    if len(by_split["dev"]) != expected_counts["dev"] or len(by_split["test"]) != expected_counts[
        "test"
    ]:
        raise PIIMBTraceSourceError("manifest example lists differ from their exact quotas")

    for task in PRIMARY_TASKS:
        for split, quota in (
            ("dev", preset.dev_per_task),
            ("test", preset.test_per_task),
        ):
            task_records = [record for record in by_split[split] if record.task_name == task]
            if len(task_records) != quota:
                raise PIIMBTraceSourceError(f"task {task!r} does not satisfy its {split} quota")
            if task_records != sorted(
                task_records,
                key=lambda record: _sampling_rank(record, salt=DEFAULT_SPLIT_SALT),
            ):
                raise PIIMBTraceSourceError(
                    f"task {task!r} {split} records differ from deterministic sample order"
                )
            if any(
                parent_split(task, record.parent_id, salt=DEFAULT_SPLIT_SALT) != split
                for record in task_records
            ):
                raise PIIMBTraceSourceError(
                    f"task {task!r} contains a record outside its stable {split} partition"
                )
        dev_parents = {
            record.parent_id for record in by_split["dev"] if record.task_name == task
        }
        test_parents = {
            record.parent_id for record in by_split["test"] if record.task_name == task
        }
        if dev_parents & test_parents:
            raise PIIMBTraceSourceError(f"task {task!r} leaks a parent across dev and test")

    expected_task_sequence: list[str] = []
    for task in PRIMARY_TASKS:
        expected_task_sequence.extend([task] * preset.dev_per_task)
    if [record.task_name for record in by_split["dev"]] != expected_task_sequence:
        raise PIIMBTraceSourceError("development records differ from deterministic task order")
    expected_task_sequence = []
    for task in PRIMARY_TASKS:
        expected_task_sequence.extend([task] * preset.test_per_task)
    if [record.task_name for record in by_split["test"]] != expected_task_sequence:
        raise PIIMBTraceSourceError("test records differ from deterministic task order")

    row_identities = [record.row_identity for record in records]
    source_indices = [record.source_row_index for record in records]
    if len(row_identities) != len(set(row_identities)):
        raise PIIMBTraceSourceError("selected (source_row_index, uid) identities must be unique")
    if len(source_indices) != len(set(source_indices)):
        raise PIIMBTraceSourceError("selected source_row_index values must be unique")
    if any(index >= source_lock.rows for index in source_indices):
        raise PIIMBTraceSourceError("a selected source_row_index lies beyond the source lock")

    checksums = _mapping(root["checksums"], name="checksums", fields=_CHECKSUM_FIELDS)
    validated_checksums = {
        field: _sha256(value, name=f"checksums.{field}") for field, value in checksums.items()
    }
    expected_partials = {
        "task_labels_sha256": _checksum(root["task_labels"]),
        "dev_row_ids_sha256": _checksum(
            [[record.uid, record.source_row_index] for record in by_split["dev"]]
        ),
        "test_row_ids_sha256": _checksum(
            [[record.uid, record.source_row_index] for record in by_split["test"]]
        ),
        "metadata_sha256": _checksum(root["examples"]),
    }
    for field, expected in expected_partials.items():
        if validated_checksums[field] != expected:
            raise PIIMBTraceSourceError(f"checksums.{field} differs from manifest contents")
    manifest_body = dict(root)
    manifest_body["checksums"] = expected_partials
    if validated_checksums["manifest_sha256"] != _checksum(manifest_body):
        raise PIIMBTraceSourceError("checksums.manifest_sha256 differs from manifest contents")

    return PIIMBTraceManifest(
        preset=preset,
        task_labels=task_labels,
        records=tuple(records),
        manifest_sha256=validated_checksums["manifest_sha256"],
        task_labels_sha256=validated_checksums["task_labels_sha256"],
    )


def load_piimb_trace_manifest(
    path: str | Path,
    *,
    source_lock: PIIMBTraceSourceLock = PINNED_PIIMB_TRACE_SOURCE,
) -> PIIMBTraceManifest:
    """Load and validate one sanitized PIIMB selection manifest."""

    manifest_path = Path(path)
    try:
        document: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PIIMBTraceSourceError(f"cannot read PIIMB trace manifest: {exc}") from exc
    if not isinstance(document, Mapping):
        raise PIIMBTraceSourceError("PIIMB trace manifest must contain a JSON object")
    return validate_piimb_trace_manifest(document, source_lock=source_lock)


def verify_piimb_trace_source(
    path: str | Path,
    *,
    source_lock: PIIMBTraceSourceLock = PINNED_PIIMB_TRACE_SOURCE,
) -> Path:
    """Verify the exact byte length, SHA-256, and physical row count of a JSONL object."""

    if not isinstance(source_lock, PIIMBTraceSourceLock):
        raise TypeError("source_lock must be a PIIMBTraceSourceLock")
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise PIIMBTraceSourceError(f"PIIMB source is not a regular file: {source_path}")
    if source_path.stat().st_size != source_lock.size_bytes:
        raise PIIMBTraceSourceError(
            f"PIIMB source size is {source_path.stat().st_size}, "
            f"expected {source_lock.size_bytes}"
        )

    digest = hashlib.sha256()
    observed_size = 0
    newline_count = 0
    last_byte = b""
    try:
        with source_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                observed_size += len(block)
                newline_count += block.count(b"\n")
                last_byte = block[-1:]
    except OSError as exc:
        raise PIIMBTraceSourceError(f"cannot read PIIMB source: {exc}") from exc
    if observed_size != source_lock.size_bytes:
        raise PIIMBTraceSourceError(
            f"PIIMB source changed while reading: {observed_size} bytes, "
            f"expected {source_lock.size_bytes}"
        )
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != source_lock.sha256:
        raise PIIMBTraceSourceError(
            f"PIIMB source SHA-256 is {observed_sha256}, expected {source_lock.sha256}"
        )
    observed_rows = newline_count + int(bool(observed_size) and last_byte != b"\n")
    if observed_rows != source_lock.rows:
        raise PIIMBTraceSourceError(
            f"PIIMB source has {observed_rows} physical rows, expected {source_lock.rows}"
        )
    return source_path


def _verify_source_metadata(
    record: PIIMBTraceManifestRecord,
    example: PIIMBExample,
) -> None:
    if (example.source_row_index, example.uid) != record.row_identity:
        raise PIIMBTraceSourceError(
            f"source identity at row {record.source_row_index} differs from "
            f"({record.source_row_index}, {record.uid!r})"
        )
    actual_metadata = {field: getattr(example, field) for field in _METADATA_FIELDS}
    if _checksum(actual_metadata) != record.metadata_sha256:
        raise PIIMBTraceSourceError(
            f"source row {record.source_row_index} metadata checksum differs from the manifest"
        )
    for field, expected in record.source_metadata().items():
        if actual_metadata[field] != expected:
            raise PIIMBTraceSourceError(
                f"source row {record.source_row_index} {field} differs from the manifest"
            )


def _trace_example(
    *,
    selection_index: int,
    record: PIIMBTraceManifestRecord,
    example: PIIMBExample,
    labels: tuple[str, ...],
    manifest: PIIMBTraceManifest,
) -> PIIMBTraceExample:
    if not example.text.strip():
        raise PIIMBTraceSourceError(f"selected source row {record.source_row_index} has blank text")
    example_id = (
        f"piimb:{record.split}:{record.task_name}:{record.source_row_index}:{record.uid}"
    )
    gold_entities: list[GoldEntity] = []
    for entity_index, entity in enumerate(example.entities):
        if not 0 <= entity.start < entity.end <= len(example.text):
            raise PIIMBTraceSourceError(
                f"source row {record.source_row_index} entity {entity_index} has invalid offsets"
            )
        entity_text = example.text[entity.start : entity.end]
        if not entity_text:
            raise PIIMBTraceSourceError(
                f"source row {record.source_row_index} entity {entity_index} selects empty text"
            )
        if entity.label not in labels:
            raise PIIMBTraceSourceError(
                f"source row {record.source_row_index} entity {entity_index} label is not in "
                "the full task vocabulary"
            )
        gold_entities.append(
            GoldEntity(
                example_id=example_id,
                start_char=entity.start,
                end_char=entity.end,
                label=entity.label,
                text=entity_text,
            )
        )
    metadata: dict[str, object] = {
        "selection_index": selection_index,
        "benchmark_split": record.split,
        **record.source_metadata(),
        "metadata_sha256": record.metadata_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "preset": manifest.preset.name,
    }
    return PIIMBTraceExample(
        split=record.split,
        example_id=example_id,
        text=example.text,
        labels=labels,
        gold_entities=tuple(gold_entities),
        metadata=metadata,
    )


def reconstruct_piimb_trace(
    manifest: PIIMBTraceManifest,
    source_path: str | Path,
    *,
    source_lock: PIIMBTraceSourceLock = PINNED_PIIMB_TRACE_SOURCE,
) -> ReconstructedPIIMBTrace:
    """Reconstruct selected examples by exact row identity and rescan task vocabularies."""

    if not isinstance(manifest, PIIMBTraceManifest):
        raise TypeError("manifest must be a PIIMBTraceManifest")
    path = verify_piimb_trace_source(source_path, source_lock=source_lock)
    wanted_by_index = {record.source_row_index: record for record in manifest.records}
    recovered: dict[tuple[int, str], PIIMBExample] = {}
    observed_labels: dict[str, set[str]] = {task: set() for task in PRIMARY_TASKS}
    parsed_rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for source_row_index, line in enumerate(handle):
                parsed_rows = source_row_index + 1
                try:
                    raw: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PIIMBTraceSourceError(
                        f"PIIMB source row {source_row_index} is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise PIIMBTraceSourceError(
                        f"PIIMB source row {source_row_index} is not an object"
                    )
                try:
                    example = parse_piimb_row(raw, source_row_index=source_row_index)
                except (TypeError, ValueError) as exc:
                    raise PIIMBTraceSourceError(
                        f"PIIMB source row {source_row_index} violates schema: {exc}"
                    ) from exc
                if example.task_name in observed_labels:
                    observed_labels[example.task_name].update(
                        entity.label for entity in example.entities
                    )
                selected = wanted_by_index.get(source_row_index)
                if selected is not None:
                    _verify_source_metadata(selected, example)
                    recovered[selected.row_identity] = example
    except (OSError, UnicodeError) as exc:
        raise PIIMBTraceSourceError(f"cannot parse PIIMB source: {exc}") from exc

    if parsed_rows != source_lock.rows:
        raise PIIMBTraceSourceError(
            f"PIIMB source parsed as {parsed_rows} rows, expected {source_lock.rows}"
        )
    missing = [
        record.row_identity
        for record in manifest.records
        if record.row_identity not in recovered
    ]
    if missing:
        raise PIIMBTraceSourceError(f"PIIMB source is missing selected row identities: {missing}")
    for task in PRIMARY_TASKS:
        observed = tuple(sorted(observed_labels[task]))
        if observed != manifest.task_labels[task]:
            raise PIIMBTraceSourceError(
                f"task {task!r} full source vocabulary differs from the manifest"
            )

    trace_examples = tuple(
        _trace_example(
            selection_index=selection_index,
            record=record,
            example=recovered[record.row_identity],
            labels=manifest.task_labels[record.task_name],
            manifest=manifest,
        )
        for selection_index, record in enumerate(manifest.records)
    )
    return ReconstructedPIIMBTrace(manifest=manifest, examples=trace_examples)


def load_piimb_trace(
    manifest_path: str | Path,
    source_path: str | Path,
    *,
    source_lock: PIIMBTraceSourceLock = PINNED_PIIMB_TRACE_SOURCE,
) -> ReconstructedPIIMBTrace:
    """Load a sanitized manifest and reconstruct its source rows without writing content."""

    manifest = load_piimb_trace_manifest(manifest_path, source_lock=source_lock)
    return reconstruct_piimb_trace(manifest, source_path, source_lock=source_lock)


__all__ = [
    "PINNED_PIIMB_TRACE_SOURCE",
    "PIIMBTraceExample",
    "PIIMBTraceManifest",
    "PIIMBTraceManifestRecord",
    "PIIMBTraceSourceError",
    "PIIMBTraceSourceLock",
    "ReconstructedPIIMBTrace",
    "load_piimb_trace",
    "load_piimb_trace_manifest",
    "reconstruct_piimb_trace",
    "validate_piimb_trace_manifest",
    "verify_piimb_trace_source",
]
