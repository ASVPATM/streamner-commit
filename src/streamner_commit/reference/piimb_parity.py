"""Pinned PIIMB real-data smoke oracle built on the existing reference observer.

The checked-in input manifest contains identifiers only. Raw source text is read from
the exact locked JSONL object, held in memory only long enough to capture one cold
append, and may be written only to an ignored or out-of-repository oracle path.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
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
    parse_piimb_row,
)
from streamner_commit.reference.observer import PINNED_GLINER_VERSION
from streamner_commit.reference.streaming_parity import (
    capture_streaming_case,
    deterministic_json_bytes,
)

PIIMB_PARITY_SCHEMA_VERSION = 1
PIIMB_PARITY_THRESHOLD = 0.5
SMOKE_DEV_PER_TASK = 5
SMOKE_TEST_PER_TASK = 20
SMOKE_CASE_COUNT = len(PRIMARY_TASKS) * (SMOKE_DEV_PER_TASK + SMOKE_TEST_PER_TASK)

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
_PINNED_PRIMARY_TASK_DIAGNOSTICS = {
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
_SAFE_REPOSITORY_OUTPUTS = (Path("artifacts/reference"), Path("results/raw"))

SplitName = Literal["dev", "test"]
_SPLITS: tuple[SplitName, ...] = ("dev", "test")


class PIIMBParityError(ValueError):
    """A manifest, pinned source object, or reference capture is inconsistent."""


@dataclass(frozen=True, slots=True)
class PIIMBSourceLock:
    """Content identity for the single pinned PIIMB sentence object."""

    size_bytes: int
    sha256: str
    rows: int

    def __post_init__(self) -> None:
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("source size_bytes must be an integer")
        if self.size_bytes < 1:
            raise ValueError("source size_bytes must be positive")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("source sha256 must be 64 lowercase hexadecimal characters")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int):
            raise TypeError("source rows must be an integer")
        if self.rows < 1:
            raise ValueError("source rows must be positive")

    def manifest_dataset(self) -> dict[str, object]:
        """Return the exact dataset object expected in a sanitized manifest."""

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


PINNED_SOURCE_LOCK = PIIMBSourceLock(
    size_bytes=PIIMB_SOURCE_SIZE_BYTES,
    sha256=PIIMB_SOURCE_SHA256,
    rows=PIIMB_SOURCE_ROW_COUNT,
)


@dataclass(frozen=True, slots=True)
class PIIMBSmokeRecord:
    """One sanitized source identity selected by the smoke manifest."""

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
        return {field: getattr(self, field) for field in _METADATA_FIELDS}

    def to_selection_dict(self, *, selection_index: int) -> dict[str, object]:
        return {
            "selection_index": selection_index,
            "benchmark_split": self.split,
            **self.source_metadata(),
            "metadata_sha256": self.metadata_sha256,
        }


@dataclass(frozen=True, slots=True)
class PIIMBSmokeManifest:
    """Validated content-free manifest fields needed by the reference export."""

    task_labels: Mapping[str, tuple[str, ...]]
    records: tuple[PIIMBSmokeRecord, ...]
    manifest_sha256: str
    task_labels_sha256: str


@dataclass(frozen=True, slots=True)
class ReconstructedPIIMBSmoke:
    """Selected identities paired with raw in-memory source examples."""

    manifest: PIIMBSmokeManifest
    rows: tuple[tuple[PIIMBSmokeRecord, PIIMBExample], ...]


def _mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PIIMBParityError(f"{name} must be an object")
    if fields is not None and set(value) != fields:
        raise PIIMBParityError(
            f"{name} fields differ: missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise PIIMBParityError(f"{name} must be a sequence")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PIIMBParityError(f"{name} must be a nonblank string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PIIMBParityError(f"{name} must be a nonnegative integer")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise PIIMBParityError(f"{name} must be 64 lowercase hexadecimal characters")
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


def _reject_raw_manifest_fields(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = {str(key) for key in value} & {"text", "entities"}
        if forbidden:
            raise PIIMBParityError(
                f"sanitized manifest contains forbidden raw fields: {sorted(forbidden)}"
            )
        for nested in value.values():
            _reject_raw_manifest_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for nested in value:
            _reject_raw_manifest_fields(nested)


def _validate_task_labels(value: object) -> Mapping[str, tuple[str, ...]]:
    mapping = _mapping(value, name="task_labels")
    if set(mapping) != set(PRIMARY_TASKS):
        raise PIIMBParityError("task_labels must contain exactly the primary PIIMB tasks")
    result: dict[str, tuple[str, ...]] = {}
    for task in PRIMARY_TASKS:
        labels = tuple(
            _string(label, name=f"task_labels.{task}[{index}]")
            for index, label in enumerate(_sequence(mapping[task], name=f"task_labels.{task}"))
        )
        if not labels or labels != tuple(sorted(set(labels))):
            raise PIIMBParityError(
                f"task_labels.{task} must be nonempty, unique, and sorted exactly"
            )
        result[task] = labels
    return MappingProxyType(result)


def _parse_record(value: object, *, split: SplitName, index: int) -> PIIMBSmokeRecord:
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
        metadata["source_row_index"],
        name=f"examples.{split}[{index}].source_row_index",
    )
    sentence_index = _nonnegative_int(
        metadata["sentence_index"],
        name=f"examples.{split}[{index}].sentence_index",
    )
    metadata_sha256 = _sha256(
        row["metadata_sha256"],
        name=f"examples.{split}[{index}].metadata_sha256",
    )
    if metadata_sha256 != _checksum(metadata):
        raise PIIMBParityError(f"examples.{split}[{index}] metadata checksum differs")
    task = str(metadata["task_name"])
    if task not in PRIMARY_TASKS:
        raise PIIMBParityError(f"examples.{split}[{index}] has an unconfigured task")
    return PIIMBSmokeRecord(
        split=split,
        uid=str(metadata["uid"]),
        source_row_index=source_row_index,
        task_name=task,
        source_dataset=str(metadata["source_dataset"]),
        source_uid=str(metadata["source_uid"]),
        parent_id=str(metadata["parent_id"]),
        sentence_index=sentence_index,
        language=str(metadata["language"]),
        metadata_sha256=metadata_sha256,
    )


def validate_piimb_smoke_manifest(
    document: Mapping[str, Any],
    *,
    source_lock: PIIMBSourceLock = PINNED_SOURCE_LOCK,
) -> PIIMBSmokeManifest:
    """Validate one generated smoke manifest without resolving any raw text."""

    root = _mapping(document, name="PIIMB smoke manifest", fields=_MANIFEST_FIELDS)
    _reject_raw_manifest_fields(root)
    if root["schema_version"] != 1:
        raise PIIMBParityError("PIIMB smoke manifest schema_version must be 1")
    if root["dataset"] != source_lock.manifest_dataset():
        raise PIIMBParityError("PIIMB smoke manifest dataset lock differs from the source lock")
    if root["tasks"] != list(PRIMARY_TASKS):
        raise PIIMBParityError("PIIMB smoke manifest tasks differ from primary task order")

    expected_sampling = {
        "preset": "smoke",
        "split_algorithm": SPLIT_ALGORITHM,
        "sample_algorithm": SAMPLE_ALGORITHM,
        "split_salt": DEFAULT_SPLIT_SALT,
        "dev_percent": DEV_PERCENT,
        "test_percent": TEST_PERCENT,
        "requested_per_task": {"dev": SMOKE_DEV_PER_TASK, "test": SMOKE_TEST_PER_TASK},
    }
    if root["sampling"] != expected_sampling:
        raise PIIMBParityError("PIIMB smoke manifest sampling policy differs from the lock")

    task_labels = _validate_task_labels(root["task_labels"])
    diagnostics = _mapping(
        root["source_diagnostics"],
        name="source_diagnostics",
        fields=_SOURCE_DIAGNOSTIC_FIELDS,
    )
    for field, value in diagnostics.items():
        _nonnegative_int(value, name=f"source_diagnostics.{field}")
    if (
        diagnostics["unique_uid_count"] + diagnostics["duplicate_uid_occurrences"]
        != diagnostics["row_count"]
    ):
        raise PIIMBParityError("source_diagnostics UID counts are inconsistent")
    if diagnostics["duplicate_uid_count"] > diagnostics["duplicate_uid_occurrences"]:
        raise PIIMBParityError("source_diagnostics duplicate counts are inconsistent")
    if diagnostics["conflicting_uid_count"] > diagnostics["duplicate_uid_count"]:
        raise PIIMBParityError("source_diagnostics conflicting UID count is inconsistent")
    if diagnostics["exact_duplicate_occurrences"] > diagnostics["duplicate_uid_occurrences"]:
        raise PIIMBParityError("source_diagnostics exact-duplicate count is inconsistent")
    if source_lock == PINNED_SOURCE_LOCK and diagnostics != _PINNED_PRIMARY_TASK_DIAGNOSTICS:
        raise PIIMBParityError("source_diagnostics differ from the pinned primary-task scan")

    examples = _mapping(
        root["examples"],
        name="examples",
        fields=frozenset({"dev", "test"}),
    )
    records: list[PIIMBSmokeRecord] = []
    by_split: dict[SplitName, list[PIIMBSmokeRecord]] = {"dev": [], "test": []}
    for split in _SPLITS:
        raw_rows = _sequence(examples[split], name=f"examples.{split}")
        parsed = [
            _parse_record(row, split=split, index=index) for index, row in enumerate(raw_rows)
        ]
        by_split[split].extend(parsed)
        records.extend(parsed)

    split_counts = {
        "dev": len(by_split["dev"]),
        "test": len(by_split["test"]),
    }
    expected_counts = {
        "total": SMOKE_CASE_COUNT,
        "dev": len(PRIMARY_TASKS) * SMOKE_DEV_PER_TASK,
        "test": len(PRIMARY_TASKS) * SMOKE_TEST_PER_TASK,
        "by_task": {
            task: {"dev": SMOKE_DEV_PER_TASK, "test": SMOKE_TEST_PER_TASK} for task in PRIMARY_TASKS
        },
    }
    if root["counts"] != expected_counts or split_counts != {
        "dev": expected_counts["dev"],
        "test": expected_counts["test"],
    }:
        raise PIIMBParityError("PIIMB smoke manifest counts differ from exact quotas")
    for task in PRIMARY_TASKS:
        for split, quota in (
            ("dev", SMOKE_DEV_PER_TASK),
            ("test", SMOKE_TEST_PER_TASK),
        ):
            if sum(record.task_name == task for record in by_split[split]) != quota:
                raise PIIMBParityError(f"task {task!r} does not satisfy its {split} quota")
        if {row.parent_id for row in by_split["dev"] if row.task_name == task} & {
            row.parent_id for row in by_split["test"] if row.task_name == task
        }:
            raise PIIMBParityError(f"task {task!r} leaks a parent across dev and test")

    row_ids = [(record.uid, record.source_row_index) for record in records]
    source_indices = [record.source_row_index for record in records]
    if len(row_ids) != len(set(row_ids)) or len(source_indices) != len(set(source_indices)):
        raise PIIMBParityError("selected source identities must be unique")
    if any(index >= source_lock.rows for index in source_indices):
        raise PIIMBParityError("selected source_row_index lies beyond the pinned source")

    checksums = _mapping(root["checksums"], name="checksums", fields=_CHECKSUM_FIELDS)
    validated_checksums = {
        field: _sha256(value, name=f"checksums.{field}") for field, value in checksums.items()
    }
    expected_partials = {
        "task_labels_sha256": _checksum(root["task_labels"]),
        "dev_row_ids_sha256": _checksum(
            [[row.uid, row.source_row_index] for row in by_split["dev"]]
        ),
        "test_row_ids_sha256": _checksum(
            [[row.uid, row.source_row_index] for row in by_split["test"]]
        ),
        "metadata_sha256": _checksum(root["examples"]),
    }
    for field, expected in expected_partials.items():
        if validated_checksums[field] != expected:
            raise PIIMBParityError(f"checksums.{field} differs from manifest contents")
    manifest_body = dict(root)
    manifest_body["checksums"] = expected_partials
    if validated_checksums["manifest_sha256"] != _checksum(manifest_body):
        raise PIIMBParityError("checksums.manifest_sha256 differs from manifest contents")

    return PIIMBSmokeManifest(
        task_labels=task_labels,
        records=tuple(records),
        manifest_sha256=validated_checksums["manifest_sha256"],
        task_labels_sha256=validated_checksums["task_labels_sha256"],
    )


def load_piimb_smoke_manifest(
    path: str | Path,
    *,
    source_lock: PIIMBSourceLock = PINNED_SOURCE_LOCK,
) -> PIIMBSmokeManifest:
    """Read and validate one sanitized smoke manifest from disk."""

    manifest_path = Path(path)
    try:
        value: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PIIMBParityError(f"cannot read PIIMB smoke manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PIIMBParityError("PIIMB smoke manifest must contain a JSON object")
    return validate_piimb_smoke_manifest(value, source_lock=source_lock)


def verify_piimb_source_file(
    path: str | Path,
    *,
    source_lock: PIIMBSourceLock = PINNED_SOURCE_LOCK,
) -> Path:
    """Verify the exact byte length and SHA-256 of the local sentence JSONL."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise PIIMBParityError(f"PIIMB source is not a regular file: {source_path}")
    observed_size = source_path.stat().st_size
    if observed_size != source_lock.size_bytes:
        raise PIIMBParityError(
            f"PIIMB source size is {observed_size}, expected {source_lock.size_bytes}"
        )
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    observed_sha = digest.hexdigest()
    if observed_sha != source_lock.sha256:
        raise PIIMBParityError(
            f"PIIMB source SHA-256 is {observed_sha}, expected {source_lock.sha256}"
        )
    return source_path


def _verify_record(record: PIIMBSmokeRecord, example: PIIMBExample) -> None:
    for field in _METADATA_FIELDS:
        if getattr(example, field) != getattr(record, field):
            raise PIIMBParityError(
                f"source row {record.source_row_index} {field} differs from the manifest"
            )


def reconstruct_piimb_smoke(
    manifest: PIIMBSmokeManifest,
    source_path: str | Path,
    *,
    source_lock: PIIMBSourceLock = PINNED_SOURCE_LOCK,
) -> ReconstructedPIIMBSmoke:
    """Recover all 100 selected rows by source index and UID, validating full labels."""

    if not isinstance(manifest, PIIMBSmokeManifest):
        raise TypeError("manifest must be a PIIMBSmokeManifest")
    path = verify_piimb_source_file(source_path, source_lock=source_lock)
    wanted = {record.source_row_index: record for record in manifest.records}
    recovered: dict[int, PIIMBExample] = {}
    observed_labels: dict[str, set[str]] = {task: set() for task in PRIMARY_TASKS}
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for source_row_index, line in enumerate(handle):
            row_count = source_row_index + 1
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PIIMBParityError(
                    f"PIIMB source row {source_row_index} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise PIIMBParityError(f"PIIMB source row {source_row_index} is not an object")
            try:
                example = parse_piimb_row(raw, source_row_index=source_row_index)
            except (TypeError, ValueError) as exc:
                raise PIIMBParityError(
                    f"PIIMB source row {source_row_index} violates schema: {exc}"
                ) from exc
            if example.task_name in observed_labels:
                observed_labels[example.task_name].update(
                    entity.label for entity in example.entities
                )
            record = wanted.get(source_row_index)
            if record is not None:
                _verify_record(record, example)
                if not example.text.strip():
                    raise PIIMBParityError(f"selected source row {source_row_index} has blank text")
                recovered[source_row_index] = example

    if row_count != source_lock.rows:
        raise PIIMBParityError(f"PIIMB source has {row_count} rows, expected {source_lock.rows}")
    missing = sorted(set(wanted) - set(recovered))
    if missing:
        raise PIIMBParityError(f"PIIMB source is missing selected row indices: {missing}")
    for task in PRIMARY_TASKS:
        expected = manifest.task_labels[task]
        actual = tuple(sorted(observed_labels[task]))
        if actual != expected:
            raise PIIMBParityError(
                f"task {task!r} full source vocabulary differs from the manifest"
            )
    return ReconstructedPIIMBSmoke(
        manifest=manifest,
        rows=tuple((record, recovered[record.source_row_index]) for record in manifest.records),
    )


def _model_identity(model_id: object, model_revision: object) -> tuple[str, str]:
    identifier = _string(model_id, name="model_id")
    revision = _string(model_revision, name="model_revision")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise PIIMBParityError("model_revision must be 40 lowercase hexadecimal characters")
    return identifier, revision


def capture_piimb_reference_smoke(
    model: object,
    *,
    manifest: PIIMBSmokeManifest,
    source_path: str | Path,
    model_id: str,
    model_revision: str,
    verify_reference: bool = True,
    source_lock: PIIMBSourceLock = PINNED_SOURCE_LOCK,
) -> dict[str, Any]:
    """Capture one cold, single-append reference trace for every smoke selection row."""

    if not isinstance(verify_reference, bool):
        raise TypeError("verify_reference must be a boolean")
    identifier, revision = _model_identity(model_id, model_revision)
    reconstructed = reconstruct_piimb_smoke(
        manifest,
        source_path,
        source_lock=source_lock,
    )
    run_id = f"piimb-smoke-{revision[:12]}-{manifest.manifest_sha256[:12]}"
    cases: list[dict[str, Any]] = []
    total_updates = 0
    total_spans = 0
    total_public_score_checks = 0
    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()

    for selection_index, (record, example) in enumerate(reconstructed.rows):
        case_id = f"piimb:{record.split}:{record.task_name}:{record.source_row_index}:{record.uid}"
        trace = capture_streaming_case(
            model,
            case_id=case_id,
            text=example.text,
            labels=manifest.task_labels[record.task_name],
            chunk_units=None,
            run_id=run_id,
            threshold=PIIMB_PARITY_THRESHOLD,
            verify_reference=verify_reference,
            single_chunk=True,
        )
        if (
            trace.get("chunk_mode") != "single"
            or trace.get("chunk_units") is not None
            or trace.get("step_count") != 1
            or trace.get("chunks") != [example.text]
        ):
            raise PIIMBParityError("reference case did not preserve single-append capture")
        final_state = _mapping(trace.get("final_state"), name="reference final_state")
        steps = _sequence(trace.get("steps"), name="reference steps")
        total_updates += _nonnegative_int(
            trace.get("span_update_count"), name="reference span_update_count"
        )
        total_spans += _nonnegative_int(final_state.get("span_count"), name="final span_count")
        total_public_score_checks += sum(
            _nonnegative_int(
                _mapping(step, name="reference step").get("validated_public_score_count"),
                name="validated_public_score_count",
            )
            for step in steps
        )
        split_counts[record.split] += 1
        task_counts[record.task_name] += 1
        cases.append(
            {
                "selection": record.to_selection_dict(selection_index=selection_index),
                "trace": trace,
            }
        )

    if len(cases) != SMOKE_CASE_COUNT:
        raise PIIMBParityError(f"captured {len(cases)} smoke cases, expected {SMOKE_CASE_COUNT}")
    return {
        "schema_version": PIIMB_PARITY_SCHEMA_VERSION,
        "kind": "piimb_reference_parity_smoke",
        "backend": "gliner-reference",
        "model_id": identifier,
        "model_revision_sha": revision,
        "gliner_version": PINNED_GLINER_VERSION,
        "device": "cpu",
        "dtype": "float32",
        "dataset": source_lock.manifest_dataset(),
        "threshold": PIIMB_PARITY_THRESHOLD,
        "capture": {"chunk_mode": "single", "appends_per_case": 1},
        "selection": {
            "preset": "smoke",
            "manifest_sha256": manifest.manifest_sha256,
            "task_labels_sha256": manifest.task_labels_sha256,
            "case_count": len(cases),
            "split_counts": {split: split_counts[split] for split in ("dev", "test")},
            "task_counts": {task: task_counts[task] for task in PRIMARY_TASKS},
        },
        "task_labels": {task: list(manifest.task_labels[task]) for task in PRIMARY_TASKS},
        "totals": {
            "case_count": len(cases),
            "step_count": len(cases),
            "span_update_count": total_updates,
            "final_span_count": total_spans,
            "validated_public_score_count": total_public_score_checks,
        },
        "cases": cases,
    }


def _safe_output_path(path: Path, *, repository_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = repository_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved
    if not any(relative.is_relative_to(prefix) for prefix in _SAFE_REPOSITORY_OUTPUTS):
        raise PIIMBParityError(
            "PIIMB oracle contains licensed text and must be written outside the repository "
            "or under artifacts/reference or results/raw"
        )
    return resolved


def write_piimb_reference_smoke(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write deterministic oracle bytes only to an ignored or external location."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    root = Path.cwd() if repository_root is None else Path(repository_root)
    path = _safe_output_path(Path(output_path), repository_root=root)
    data = deterministic_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


__all__ = [
    "PINNED_SOURCE_LOCK",
    "PIIMB_PARITY_SCHEMA_VERSION",
    "PIIMB_PARITY_THRESHOLD",
    "SMOKE_CASE_COUNT",
    "PIIMBParityError",
    "PIIMBSourceLock",
    "PIIMBSmokeManifest",
    "PIIMBSmokeRecord",
    "ReconstructedPIIMBSmoke",
    "capture_piimb_reference_smoke",
    "load_piimb_smoke_manifest",
    "reconstruct_piimb_smoke",
    "validate_piimb_smoke_manifest",
    "verify_piimb_source_file",
    "write_piimb_reference_smoke",
]
