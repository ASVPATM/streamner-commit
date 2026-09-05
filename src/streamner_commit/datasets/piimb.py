"""Pinned, leakage-safe PIIMB sentence loading and deterministic sampling.

Raw PIIMB text is intentionally kept in memory only.  The manifest builders in
this module serialize identifiers and non-content metadata, never text or gold
entity annotations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

PIIMB_DATASET_ID = "piimb/pii-masking-benchmark"
PIIMB_SUBSET = "sentences"
PIIMB_REVISION = "4a13e9ffe6fd0d275efbde8afd4d8d8f1ffc2133"
PIIMB_SPLIT = "test"
PIIMB_LICENSE = "CC-BY-NC-4.0"
PIIMB_SOURCE_FILE = "data/test_sentences.jsonl"
PIIMB_SOURCE_SIZE_BYTES = 60_412_185
PIIMB_SOURCE_SHA256 = "5ff46f3a80316318794f94596fa374060d70f2f32a85909f958cbabc70bae41f"
PIIMB_SOURCE_ROW_COUNT = 150_022

PRIMARY_TASKS = (
    "ai4privacy-en",
    "gretel",
    "nemotron-pii",
    "privy",
)

DEFAULT_SPLIT_SALT = "streamner-commit-piimb-v1"
DEV_PERCENT = 20
TEST_PERCENT = 80
SPLIT_ALGORITHM = "sha256-task-parent-mod100-v1"
SAMPLE_ALGORITHM = "sha256-task-parent-uid-row-v1"

_ROW_FIELDS = frozenset(
    {
        "uid",
        "task_name",
        "source_dataset",
        "source_uid",
        "parent_id",
        "sentence_index",
        "text",
        "entities",
        "language",
    }
)
_ENTITY_FIELDS = frozenset({"start", "end", "label"})

SplitName = Literal["dev", "test"]
LoadDataset = Callable[..., Iterable[Mapping[str, object]]]


class PIIMBError(ValueError):
    """Pinned PIIMB data or sampling configuration is invalid."""


class PIIMBSchemaError(PIIMBError):
    """A source row does not match the pinned ``sentences`` schema."""


class PIIMBSelectionError(PIIMBError):
    """A deterministic benchmark selection cannot satisfy its quota."""


def _require_string(value: object, *, name: str, allow_blank: bool = False) -> str:
    if not isinstance(value, str):
        raise PIIMBSchemaError(f"{name} must be a string")
    if not allow_blank and not value.strip():
        raise PIIMBSchemaError(f"{name} must be nonblank")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PIIMBSchemaError(f"{name} must be an integer")
    if value < 0:
        raise PIIMBSchemaError(f"{name} must be nonnegative")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    result = _require_nonnegative_int(value, name=name)
    if result == 0:
        raise PIIMBError(f"{name} must be positive")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _checksum(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tasks, str | bytes) or not isinstance(tasks, Sequence):
        raise TypeError("tasks must be an ordered sequence of strings")
    values = tuple(
        _require_string(task, name=f"tasks[{index}]") for index, task in enumerate(tasks)
    )
    if not values:
        raise PIIMBError("at least one task is required")
    if len(set(values)) != len(values):
        raise PIIMBError("tasks must not contain duplicates")
    return tuple(sorted(values))


def _mapping_fields(
    value: object,
    *,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PIIMBSchemaError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PIIMBSchemaError(
            f"{name} fields differ from pinned schema: missing={missing}, extra={extra}"
        )
    return value


@dataclass(frozen=True, slots=True)
class PIIMBEntity:
    """One unmodified PIIMB gold annotation in half-open character offsets."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        start = _require_nonnegative_int(self.start, name="entity.start")
        end = _require_nonnegative_int(self.end, name="entity.end")
        if end <= start:
            raise PIIMBSchemaError("entity.end must be greater than entity.start")
        _require_string(self.label, name="entity.label")


@dataclass(frozen=True, slots=True)
class PIIMBExample:
    """One validated source sentence; this content is never put in a manifest."""

    uid: str
    source_row_index: int
    task_name: str
    source_dataset: str
    source_uid: str
    parent_id: str
    sentence_index: int
    text: str
    entities: tuple[PIIMBEntity, ...]
    language: str

    def __post_init__(self) -> None:
        for name in (
            "uid",
            "task_name",
            "source_dataset",
            "source_uid",
            "parent_id",
            "language",
        ):
            _require_string(getattr(self, name), name=name)
        _require_nonnegative_int(self.source_row_index, name="source_row_index")
        _require_nonnegative_int(self.sentence_index, name="sentence_index")
        if not isinstance(self.text, str) or not self.text:
            raise PIIMBSchemaError("text must be a nonempty string")
        if not isinstance(self.entities, tuple):
            raise PIIMBSchemaError("entities must be an immutable tuple")
        for index, entity in enumerate(self.entities):
            if not isinstance(entity, PIIMBEntity):
                raise PIIMBSchemaError(f"entities[{index}] must be a PIIMBEntity")
            if entity.end > len(self.text):
                raise PIIMBSchemaError(
                    f"entities[{index}] ends at {entity.end}, beyond text length {len(self.text)}"
                )
            if not self.text[entity.start : entity.end]:
                raise PIIMBSchemaError(f"entities[{index}] selects empty text")

    @property
    def is_negative(self) -> bool:
        """Whether this sentence has no gold PII annotations."""

        return not self.entities


@dataclass(frozen=True, slots=True)
class PIIMBPreset:
    """Exact per-task development and test quotas."""

    name: str
    dev_per_task: int
    test_per_task: int

    def __post_init__(self) -> None:
        _require_string(self.name, name="preset.name")
        _require_positive_int(self.dev_per_task, name="preset.dev_per_task")
        _require_positive_int(self.test_per_task, name="preset.test_per_task")


PRESETS: Mapping[str, PIIMBPreset] = MappingProxyType(
    {
        "smoke": PIIMBPreset("smoke", 5, 20),
        "research-small": PIIMBPreset("research-small", 100, 300),
        "research-full": PIIMBPreset("research-full", 250, 1000),
    }
)


def resolve_preset(value: str | PIIMBPreset) -> PIIMBPreset:
    """Resolve a built-in preset, accepting underscore config spellings."""

    if isinstance(value, PIIMBPreset):
        return value
    if not isinstance(value, str):
        raise TypeError("preset must be a name or PIIMBPreset")
    canonical = value.replace("_", "-")
    try:
        return PRESETS[canonical]
    except KeyError as exc:
        raise PIIMBSelectionError(
            f"unknown PIIMB preset {value!r}; expected one of {sorted(PRESETS)}"
        ) from exc


def parse_piimb_row(
    row: Mapping[str, object],
    *,
    source_row_index: int = 0,
) -> PIIMBExample:
    """Strictly parse one row from the pinned PIIMB ``sentences`` subset."""

    values = _mapping_fields(row, expected=_ROW_FIELDS, name="PIIMB row")
    row_index = _require_nonnegative_int(source_row_index, name="source_row_index")
    raw_entities = values["entities"]
    if isinstance(raw_entities, str | bytes) or not isinstance(raw_entities, Sequence):
        raise PIIMBSchemaError("entities must be a list")
    entities: list[PIIMBEntity] = []
    for index, raw_entity in enumerate(raw_entities):
        entity = _mapping_fields(
            raw_entity,
            expected=_ENTITY_FIELDS,
            name=f"entities[{index}]",
        )
        entities.append(
            PIIMBEntity(
                start=_require_nonnegative_int(entity["start"], name=f"entities[{index}].start"),
                end=_require_nonnegative_int(entity["end"], name=f"entities[{index}].end"),
                label=_require_string(entity["label"], name=f"entities[{index}].label"),
            )
        )
    return PIIMBExample(
        uid=_require_string(values["uid"], name="uid"),
        source_row_index=row_index,
        task_name=_require_string(values["task_name"], name="task_name"),
        source_dataset=_require_string(values["source_dataset"], name="source_dataset"),
        source_uid=_require_string(values["source_uid"], name="source_uid"),
        parent_id=_require_string(values["parent_id"], name="parent_id"),
        sentence_index=_require_nonnegative_int(values["sentence_index"], name="sentence_index"),
        text=_require_string(values["text"], name="text", allow_blank=True),
        entities=tuple(entities),
        language=_require_string(values["language"], name="language"),
    )


def load_piimb(
    *,
    tasks: Sequence[str] = PRIMARY_TASKS,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    download_mode: str | None = None,
    load_dataset_fn: LoadDataset | None = None,
) -> tuple[PIIMBExample, ...]:
    """Load the exact pinned revision lazily and materialize reusable examples."""

    ordered_tasks = _normalized_tasks(tasks)
    if not isinstance(streaming, bool):
        raise TypeError("streaming must be a boolean")
    if download_mode is not None and not isinstance(download_mode, str):
        raise TypeError("download_mode must be a string or None")
    validate_source_count = load_dataset_fn is None
    if load_dataset_fn is None:
        from datasets import (  # type: ignore[import-untyped]
            load_dataset as huggingface_load_dataset,
        )

        load_dataset_fn = huggingface_load_dataset
    kwargs: dict[str, object] = {
        "path": PIIMB_DATASET_ID,
        "name": PIIMB_SUBSET,
        "split": PIIMB_SPLIT,
        "revision": PIIMB_REVISION,
        "streaming": streaming,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if download_mode is not None:
        kwargs["download_mode"] = download_mode
    loaded = load_dataset_fn(**kwargs)
    requested = set(ordered_tasks)
    examples: list[PIIMBExample] = []
    observed_tasks: set[str] = set()
    loaded_row_count = 0
    for source_row_index, raw_row in enumerate(loaded):
        loaded_row_count = source_row_index + 1
        example = parse_piimb_row(raw_row, source_row_index=source_row_index)
        if example.task_name not in requested:
            continue
        observed_tasks.add(example.task_name)
        examples.append(example)
    if validate_source_count and loaded_row_count != PIIMB_SOURCE_ROW_COUNT:
        raise PIIMBSchemaError(
            f"pinned PIIMB split has {loaded_row_count} rows, expected {PIIMB_SOURCE_ROW_COUNT}"
        )
    missing = sorted(requested - observed_tasks)
    if missing:
        raise PIIMBSchemaError(f"pinned PIIMB data is missing requested tasks: {missing}")
    examples.sort(key=lambda item: (item.task_name, item.uid, item.source_row_index))
    return tuple(examples)


def task_label_vocabulary(
    rows: Iterable[PIIMBExample],
    *,
    tasks: Sequence[str] = PRIMARY_TASKS,
) -> Mapping[str, tuple[str, ...]]:
    """Return sorted exact label strings from all rows of each requested task."""

    ordered_tasks = _normalized_tasks(tasks)
    labels: dict[str, set[str]] = {task: set() for task in ordered_tasks}
    for row in rows:
        if not isinstance(row, PIIMBExample):
            raise TypeError("rows must contain PIIMBExample values")
        if row.task_name in labels:
            labels[row.task_name].update(entity.label for entity in row.entities)
    missing = [task for task in ordered_tasks if not labels[task]]
    if missing:
        raise PIIMBSchemaError(f"requested tasks have no label vocabulary: {missing}")
    return MappingProxyType({task: tuple(sorted(labels[task])) for task in ordered_tasks})


def parent_split(
    task_name: str,
    parent_id: str,
    *,
    salt: str = DEFAULT_SPLIT_SALT,
) -> SplitName:
    """Assign one task-qualified parent to the stable 20/80 development/test split."""

    task = _require_string(task_name, name="task_name")
    parent = _require_string(parent_id, name="parent_id")
    split_salt = _require_string(salt, name="salt")
    digest = hashlib.sha256(f"{split_salt}:{task}:{parent}".encode()).hexdigest()
    return "dev" if int(digest, 16) % 100 < DEV_PERCENT else "test"


def _sample_rank(row: PIIMBExample, *, salt: str) -> str:
    payload = f"{salt}:sample:{row.task_name}:{row.parent_id}:{row.uid}:{row.source_row_index}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_signature(row: PIIMBExample) -> tuple[object, ...]:
    return (
        row.uid,
        row.task_name,
        row.source_dataset,
        row.source_uid,
        row.parent_id,
        row.sentence_index,
        row.text,
        row.entities,
        row.language,
    )


@dataclass(frozen=True, slots=True)
class PIIMBSourceDiagnostics:
    """Aggregate source-identity quirks without exposing source content."""

    row_count: int
    unique_uid_count: int
    duplicate_uid_count: int
    duplicate_uid_occurrences: int
    conflicting_uid_count: int
    exact_duplicate_occurrences: int

    @classmethod
    def from_rows(cls, rows: Sequence[PIIMBExample]) -> PIIMBSourceDiagnostics:
        by_uid: dict[str, list[PIIMBExample]] = {}
        for row in rows:
            by_uid.setdefault(row.uid, []).append(row)
        duplicate_groups = [group for group in by_uid.values() if len(group) > 1]
        exact_duplicate_occurrences = 0
        conflicting_uid_count = 0
        for group in duplicate_groups:
            signature_counts: dict[tuple[object, ...], int] = {}
            for row in group:
                signature = _source_signature(row)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
            exact_duplicate_occurrences += sum(count - 1 for count in signature_counts.values())
            conflicting_uid_count += len(signature_counts) > 1
        return cls(
            row_count=len(rows),
            unique_uid_count=len(by_uid),
            duplicate_uid_count=len(duplicate_groups),
            duplicate_uid_occurrences=len(rows) - len(by_uid),
            conflicting_uid_count=conflicting_uid_count,
            exact_duplicate_occurrences=exact_duplicate_occurrences,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "row_count": self.row_count,
            "unique_uid_count": self.unique_uid_count,
            "duplicate_uid_count": self.duplicate_uid_count,
            "duplicate_uid_occurrences": self.duplicate_uid_occurrences,
            "conflicting_uid_count": self.conflicting_uid_count,
            "exact_duplicate_occurrences": self.exact_duplicate_occurrences,
        }


@dataclass(frozen=True, slots=True)
class PIIMBManifestRecord:
    """Sanitized example identity safe to write to the repository."""

    uid: str
    source_row_index: int
    task_name: str
    source_dataset: str
    source_uid: str
    parent_id: str
    sentence_index: int
    language: str

    @classmethod
    def from_example(cls, example: PIIMBExample) -> PIIMBManifestRecord:
        return cls(
            uid=example.uid,
            source_row_index=example.source_row_index,
            task_name=example.task_name,
            source_dataset=example.source_dataset,
            source_uid=example.source_uid,
            parent_id=example.parent_id,
            sentence_index=example.sentence_index,
            language=example.language,
        )

    def to_dict(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "uid": self.uid,
            "source_row_index": self.source_row_index,
            "task_name": self.task_name,
            "source_dataset": self.source_dataset,
            "source_uid": self.source_uid,
            "parent_id": self.parent_id,
            "sentence_index": self.sentence_index,
            "language": self.language,
        }
        return {**metadata, "metadata_sha256": _checksum(metadata)}


@dataclass(frozen=True, slots=True)
class PIIMBManifest:
    """Content-free deterministic manifest for one selected PIIMB benchmark."""

    preset: PIIMBPreset
    salt: str
    tasks: tuple[str, ...]
    task_labels: Mapping[str, tuple[str, ...]]
    source_diagnostics: PIIMBSourceDiagnostics
    dev: tuple[PIIMBManifestRecord, ...]
    test: tuple[PIIMBManifestRecord, ...]

    def to_dict(self) -> dict[str, object]:
        task_labels = {task: list(self.task_labels[task]) for task in self.tasks}
        dev = [row.to_dict() for row in self.dev]
        test = [row.to_dict() for row in self.test]
        counts = {
            "total": len(dev) + len(test),
            "dev": len(dev),
            "test": len(test),
            "by_task": {
                task: {
                    "dev": sum(row.task_name == task for row in self.dev),
                    "test": sum(row.task_name == task for row in self.test),
                }
                for task in self.tasks
            },
        }
        payload: dict[str, object] = {
            "schema_version": 1,
            "dataset": {
                "id": PIIMB_DATASET_ID,
                "subset": PIIMB_SUBSET,
                "revision": PIIMB_REVISION,
                "split": PIIMB_SPLIT,
                "license": PIIMB_LICENSE,
                "source_file": {
                    "path": PIIMB_SOURCE_FILE,
                    "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
                    "sha256": PIIMB_SOURCE_SHA256,
                    "rows": PIIMB_SOURCE_ROW_COUNT,
                },
            },
            "sampling": {
                "preset": self.preset.name,
                "split_algorithm": SPLIT_ALGORITHM,
                "sample_algorithm": SAMPLE_ALGORITHM,
                "split_salt": self.salt,
                "dev_percent": DEV_PERCENT,
                "test_percent": TEST_PERCENT,
                "requested_per_task": {
                    "dev": self.preset.dev_per_task,
                    "test": self.preset.test_per_task,
                },
            },
            "tasks": list(self.tasks),
            "task_labels": task_labels,
            "source_diagnostics": self.source_diagnostics.to_dict(),
            "counts": counts,
            "examples": {"dev": dev, "test": test},
        }
        partial_checksums = {
            "task_labels_sha256": _checksum(task_labels),
            "dev_row_ids_sha256": _checksum([[row.uid, row.source_row_index] for row in self.dev]),
            "test_row_ids_sha256": _checksum(
                [[row.uid, row.source_row_index] for row in self.test]
            ),
            "metadata_sha256": _checksum(payload["examples"]),
        }
        payload["checksums"] = {
            **partial_checksums,
            "manifest_sha256": _checksum({**payload, "checksums": partial_checksums}),
        }
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize with stable key order and a final newline."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                separators=(",", ":") if indent is None else None,
                allow_nan=False,
            )
            + "\n"
        )


@dataclass(frozen=True, slots=True)
class PIIMBSelection:
    """In-memory raw examples selected for development and held-out testing."""

    preset: PIIMBPreset
    salt: str
    tasks: tuple[str, ...]
    task_labels: Mapping[str, tuple[str, ...]]
    source_diagnostics: PIIMBSourceDiagnostics
    dev: tuple[PIIMBExample, ...]
    test: tuple[PIIMBExample, ...]

    def __post_init__(self) -> None:
        _require_string(self.salt, name="salt")
        if self.tasks != _normalized_tasks(self.tasks):
            raise PIIMBSelectionError("selection tasks must be unique and sorted")
        if set(self.task_labels) != set(self.tasks):
            raise PIIMBSelectionError("selection task labels must match its tasks")
        row_ids = [(row.uid, row.source_row_index) for row in (*self.dev, *self.test)]
        if len(row_ids) != len(set(row_ids)):
            raise PIIMBSelectionError("selected (uid, source_row_index) identities must be unique")
        for task in self.tasks:
            task_dev = tuple(row for row in self.dev if row.task_name == task)
            task_test = tuple(row for row in self.test if row.task_name == task)
            if len(task_dev) != self.preset.dev_per_task:
                raise PIIMBSelectionError(f"task {task!r} does not satisfy the development quota")
            if len(task_test) != self.preset.test_per_task:
                raise PIIMBSelectionError(f"task {task!r} does not satisfy the test quota")
            dev_parents = {row.parent_id for row in task_dev}
            test_parents = {row.parent_id for row in task_test}
            if dev_parents & test_parents:
                raise PIIMBSelectionError(f"task {task!r} has parent leakage across dev/test")

    def manifest(self) -> PIIMBManifest:
        """Drop all source content and gold annotations into safe identity records."""

        return PIIMBManifest(
            preset=self.preset,
            salt=self.salt,
            tasks=self.tasks,
            task_labels=self.task_labels,
            source_diagnostics=self.source_diagnostics,
            dev=tuple(PIIMBManifestRecord.from_example(row) for row in self.dev),
            test=tuple(PIIMBManifestRecord.from_example(row) for row in self.test),
        )


def build_piimb_selection(
    rows: Iterable[PIIMBExample],
    *,
    preset: str | PIIMBPreset = "smoke",
    tasks: Sequence[str] = PRIMARY_TASKS,
    salt: str = DEFAULT_SPLIT_SALT,
) -> PIIMBSelection:
    """Build an exact, task-balanced selection without inspecting entity presence."""

    resolved = resolve_preset(preset)
    ordered_tasks = _normalized_tasks(tasks)
    split_salt = _require_string(salt, name="salt")
    materialized = tuple(rows)
    if any(not isinstance(row, PIIMBExample) for row in materialized):
        raise TypeError("rows must contain PIIMBExample values")
    row_ids = [(row.uid, row.source_row_index) for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise PIIMBSchemaError("PIIMB rows must have unique (uid, source_row_index) identities")
    source_diagnostics = PIIMBSourceDiagnostics.from_rows(materialized)
    vocabulary = task_label_vocabulary(materialized, tasks=ordered_tasks)
    requested = set(ordered_tasks)
    selected: dict[SplitName, list[PIIMBExample]] = {"dev": [], "test": []}
    for task in ordered_tasks:
        task_rows = [row for row in materialized if row.task_name == task]
        quotas: tuple[tuple[SplitName, int], ...] = (
            ("dev", resolved.dev_per_task),
            ("test", resolved.test_per_task),
        )
        for split, quota in quotas:
            candidates = [
                row
                for row in task_rows
                if parent_split(task, row.parent_id, salt=split_salt) == split
            ]
            candidates.sort(
                key=lambda row: (
                    _sample_rank(row, salt=split_salt),
                    row.uid,
                )
            )
            if len(candidates) < quota:
                raise PIIMBSelectionError(
                    f"task {task!r} has {len(candidates)} {split} rows, fewer than quota {quota}"
                )
            selected[split].extend(candidates[:quota])
    if any(row.task_name not in requested for row in (*selected["dev"], *selected["test"])):
        raise AssertionError("selection included an unrequested task")
    return PIIMBSelection(
        preset=resolved,
        salt=split_salt,
        tasks=ordered_tasks,
        task_labels=vocabulary,
        source_diagnostics=source_diagnostics,
        dev=tuple(selected["dev"]),
        test=tuple(selected["test"]),
    )


def load_piimb_selection(
    *,
    preset: str | PIIMBPreset = "smoke",
    tasks: Sequence[str] = PRIMARY_TASKS,
    salt: str = DEFAULT_SPLIT_SALT,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    download_mode: str | None = None,
    load_dataset_fn: LoadDataset | None = None,
) -> PIIMBSelection:
    """Load the pinned dataset once, then build a deterministic preset selection."""

    rows = load_piimb(
        tasks=tasks,
        cache_dir=cache_dir,
        streaming=streaming,
        download_mode=download_mode,
        load_dataset_fn=load_dataset_fn,
    )
    return build_piimb_selection(rows, preset=preset, tasks=tasks, salt=salt)


def build_task_labels_manifest(
    rows: Iterable[PIIMBExample],
    *,
    tasks: Sequence[str] = PRIMARY_TASKS,
) -> dict[str, object]:
    """Build the standalone, content-free task vocabulary artifact."""

    ordered_tasks = _normalized_tasks(tasks)
    materialized = tuple(rows)
    vocabulary = task_label_vocabulary(materialized, tasks=ordered_tasks)
    task_labels = {task: list(vocabulary[task]) for task in ordered_tasks}
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "id": PIIMB_DATASET_ID,
            "subset": PIIMB_SUBSET,
            "revision": PIIMB_REVISION,
            "split": PIIMB_SPLIT,
            "source_file": {
                "path": PIIMB_SOURCE_FILE,
                "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
                "sha256": PIIMB_SOURCE_SHA256,
                "rows": PIIMB_SOURCE_ROW_COUNT,
            },
        },
        "tasks": list(ordered_tasks),
        "task_labels": task_labels,
        "source_diagnostics": PIIMBSourceDiagnostics.from_rows(materialized).to_dict(),
    }
    partial = {"task_labels_sha256": _checksum(task_labels)}
    payload["checksums"] = {
        **partial,
        "manifest_sha256": _checksum({**payload, "checksums": partial}),
    }
    return payload


def task_labels_manifest_json(
    rows: Iterable[PIIMBExample],
    *,
    tasks: Sequence[str] = PRIMARY_TASKS,
    indent: int | None = 2,
) -> str:
    """Serialize the standalone task vocabulary artifact deterministically."""

    return (
        json.dumps(
            build_task_labels_manifest(rows, tasks=tasks),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            allow_nan=False,
        )
        + "\n"
    )


__all__ = [
    "DEFAULT_SPLIT_SALT",
    "DEV_PERCENT",
    "PIIMB_DATASET_ID",
    "PIIMB_LICENSE",
    "PIIMB_REVISION",
    "PIIMB_SOURCE_FILE",
    "PIIMB_SOURCE_ROW_COUNT",
    "PIIMB_SOURCE_SHA256",
    "PIIMB_SOURCE_SIZE_BYTES",
    "PIIMB_SPLIT",
    "PIIMB_SUBSET",
    "PRESETS",
    "PRIMARY_TASKS",
    "SAMPLE_ALGORITHM",
    "SPLIT_ALGORITHM",
    "TEST_PERCENT",
    "PIIMBEntity",
    "PIIMBError",
    "PIIMBExample",
    "PIIMBManifest",
    "PIIMBManifestRecord",
    "PIIMBPreset",
    "PIIMBSchemaError",
    "PIIMBSelection",
    "PIIMBSelectionError",
    "PIIMBSourceDiagnostics",
    "build_piimb_selection",
    "build_task_labels_manifest",
    "load_piimb",
    "load_piimb_selection",
    "parent_split",
    "parse_piimb_row",
    "resolve_preset",
    "task_label_vocabulary",
    "task_labels_manifest_json",
]
