from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.datasets.piimb import (
    DEFAULT_SPLIT_SALT,
    PIIMB_DATASET_ID,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SPLIT,
    PIIMB_SUBSET,
    PRESETS,
    PRIMARY_TASKS,
    PIIMBEntity,
    PIIMBExample,
    PIIMBPreset,
    PIIMBSchemaError,
    PIIMBSelectionError,
    PIIMBSourceDiagnostics,
    build_piimb_selection,
    build_task_labels_manifest,
    load_piimb,
    parent_split,
    parse_piimb_row,
    resolve_preset,
    task_label_vocabulary,
    task_labels_manifest_json,
)


def _raw_row(
    uid: str = "uid-1",
    *,
    task: str = "gretel",
    parent: str = "parent-1",
    label: str | None = "PERSON_NAME",
) -> dict[str, object]:
    return {
        "uid": uid,
        "task_name": task,
        "source_dataset": "synthetic-test-source",
        "source_uid": f"source-{uid}",
        "parent_id": parent,
        "sentence_index": 0,
        "text": "Call Ada.",
        "entities": [] if label is None else [{"start": 5, "end": 8, "label": label}],
        "language": "en",
    }


def _example(
    uid: str,
    task: str,
    parent: str,
    *,
    label: str | None,
) -> PIIMBExample:
    return parse_piimb_row(_raw_row(uid, task=task, parent=parent, label=label))


def _balanced_rows(
    tasks: tuple[str, ...] = ("task-a", "task-b"),
    *,
    salt: str = "unit-salt",
) -> tuple[PIIMBExample, ...]:
    rows: list[PIIMBExample] = []
    for task in tasks:
        parents: dict[str, list[str]] = {"dev": [], "test": []}
        index = 0
        while len(parents["dev"]) < 2 or len(parents["test"]) < 3:
            parent = f"{task}-parent-{index}"
            split = parent_split(task, parent, salt=salt)
            target = 2 if split == "dev" else 3
            if len(parents[split]) < target:
                parents[split].append(parent)
            index += 1
        ordered = [("dev", parent) for parent in parents["dev"]] + [
            ("test", parent) for parent in parents["test"]
        ]
        for position, (_split, parent) in enumerate(ordered):
            label = ("Z_LABEL", "A label", None, "Z_LABEL", None)[position]
            rows.append(_example(f"{task}-{position}", task, parent, label=label))
    return tuple(rows)


def test_pinned_identity_and_preset_sizes_are_exact() -> None:
    assert PIIMB_DATASET_ID == "piimb/pii-masking-benchmark"
    assert PIIMB_SUBSET == "sentences"
    assert PIIMB_REVISION == "4a13e9ffe6fd0d275efbde8afd4d8d8f1ffc2133"
    assert PIIMB_SPLIT == "test"
    assert PIIMB_SOURCE_FILE == "data/test_sentences.jsonl"
    assert PIIMB_SOURCE_SIZE_BYTES == 60_412_185
    assert PIIMB_SOURCE_ROW_COUNT == 150_022
    assert PIIMB_SOURCE_SHA256 == "5ff46f3a80316318794f94596fa374060d70f2f32a85909f958cbabc70bae41f"
    assert PRIMARY_TASKS == ("ai4privacy-en", "gretel", "nemotron-pii", "privy")
    assert DEFAULT_SPLIT_SALT == "streamner-commit-piimb-v1"
    assert (PRESETS["smoke"].dev_per_task, PRESETS["smoke"].test_per_task) == (5, 20)
    assert (PRESETS["research-small"].dev_per_task, PRESETS["research-small"].test_per_task) == (
        100,
        300,
    )
    assert (PRESETS["research-full"].dev_per_task, PRESETS["research-full"].test_per_task) == (
        250,
        1000,
    )
    assert resolve_preset("research_small") is PRESETS["research-small"]


def test_strict_row_parser_preserves_exact_labels_and_validates_offsets() -> None:
    row = parse_piimb_row(_raw_row(label="Given Name"))
    assert row.entities == (PIIMBEntity(start=5, end=8, label="Given Name"),)
    assert row.text[row.entities[0].start : row.entities[0].end] == "Ada"

    missing = _raw_row()
    del missing["language"]
    with pytest.raises(PIIMBSchemaError, match="missing=.*language"):
        parse_piimb_row(missing)
    extra = {**_raw_row(), "normalized_label": "person"}
    with pytest.raises(PIIMBSchemaError, match="extra=.*normalized_label"):
        parse_piimb_row(extra)
    invalid_offset = _raw_row()
    invalid_offset["entities"] = [{"start": 5, "end": 99, "label": "PERSON_NAME"}]
    with pytest.raises(PIIMBSchemaError, match="beyond text length"):
        parse_piimb_row(invalid_offset)
    invalid_type = _raw_row()
    invalid_type["entities"] = [{"start": True, "end": 8, "label": "PERSON_NAME"}]
    with pytest.raises(PIIMBSchemaError, match="must be an integer"):
        parse_piimb_row(invalid_type)


def test_loader_is_lazy_pinned_and_materializes_streamed_rows(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(**kwargs: object) -> list[dict[str, object]]:
        calls.append(dict(kwargs))
        return [_raw_row(), _raw_row("other", task="mapa", parent="other-parent")]

    rows = load_piimb(
        tasks=("gretel",),
        cache_dir=tmp_path,
        streaming=True,
        download_mode="reuse_dataset_if_exists",
        load_dataset_fn=fake_load_dataset,
    )

    assert isinstance(rows, tuple) and [(row.uid, row.source_row_index) for row in rows] == [
        ("uid-1", 0)
    ]
    assert calls == [
        {
            "path": PIIMB_DATASET_ID,
            "name": PIIMB_SUBSET,
            "split": PIIMB_SPLIT,
            "revision": PIIMB_REVISION,
            "streaming": True,
            "cache_dir": str(tmp_path),
            "download_mode": "reuse_dataset_if_exists",
        }
    ]


def test_duplicate_uids_are_preserved_and_diagnosed_by_global_row_index() -> None:
    first = parse_piimb_row(_raw_row("duplicate"), source_row_index=7)
    exact = parse_piimb_row(_raw_row("duplicate"), source_row_index=8)
    conflict = parse_piimb_row(
        _raw_row("duplicate", label="DIFFERENT_LABEL"),
        source_row_index=9,
    )
    diagnostics = PIIMBSourceDiagnostics.from_rows((first, exact, conflict))

    assert diagnostics.to_dict() == {
        "row_count": 3,
        "unique_uid_count": 1,
        "duplicate_uid_count": 1,
        "duplicate_uid_occurrences": 2,
        "conflicting_uid_count": 1,
        "exact_duplicate_occurrences": 1,
    }


def test_task_vocab_and_parent_hash_are_exact_and_order_independent() -> None:
    rows = (
        _example("2", "gretel", "p2", label="Z_LABEL"),
        _example("1", "gretel", "p1", label="A label"),
        _example("3", "gretel", "p3", label="Z_LABEL"),
    )
    assert task_label_vocabulary(rows, tasks=("gretel",)) == {"gretel": ("A label", "Z_LABEL")}
    expected_hash = hashlib.sha256(b"salt:gretel:document-7").hexdigest()
    expected = "dev" if int(expected_hash, 16) % 100 < 20 else "test"
    assert parent_split("gretel", "document-7", salt="salt") == expected
    assert parent_split("gretel", "document-7", salt="salt") == expected


def test_selection_is_reproducible_balanced_parent_disjoint_and_keeps_negatives() -> None:
    rows = _balanced_rows()
    preset = PIIMBPreset("unit", dev_per_task=2, test_per_task=3)
    first = build_piimb_selection(rows, preset=preset, tasks=("task-b", "task-a"), salt="unit-salt")
    second = build_piimb_selection(
        reversed(rows),
        preset=preset,
        tasks=("task-a", "task-b"),
        salt="unit-salt",
    )

    assert first.tasks == ("task-a", "task-b")
    assert [row.uid for row in first.dev] == [row.uid for row in second.dev]
    assert [row.uid for row in first.test] == [row.uid for row in second.test]
    for task in first.tasks:
        dev = [row for row in first.dev if row.task_name == task]
        test = [row for row in first.test if row.task_name == task]
        assert (len(dev), len(test)) == (2, 3)
        assert {row.parent_id for row in dev}.isdisjoint(row.parent_id for row in test)
        assert all(parent_split(task, row.parent_id, salt="unit-salt") == "dev" for row in dev)
        assert all(parent_split(task, row.parent_id, salt="unit-salt") == "test" for row in test)
    assert any(row.is_negative for row in (*first.dev, *first.test))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_manifests_are_deterministic_content_free_and_checksummed() -> None:
    rows = _balanced_rows()
    selection = build_piimb_selection(
        rows,
        preset=PIIMBPreset("unit", 2, 3),
        tasks=("task-a", "task-b"),
        salt="unit-salt",
    )
    manifest = selection.manifest()
    first = manifest.to_json(indent=None)
    second = selection.manifest().to_json(indent=None)
    payload = json.loads(first)

    assert first == second and first.endswith("\n")
    assert not {"text", "entities"} & _all_keys(payload)
    assert payload["counts"]["by_task"] == {
        "task-a": {"dev": 2, "test": 3},
        "task-b": {"dev": 2, "test": 3},
    }
    assert payload["dataset"]["source_file"] == {
        "path": PIIMB_SOURCE_FILE,
        "rows": PIIMB_SOURCE_ROW_COUNT,
        "sha256": PIIMB_SOURCE_SHA256,
        "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
    }
    assert all("source_row_index" in row for row in payload["examples"]["test"])
    assert len(payload["checksums"]["manifest_sha256"]) == 64
    assert all(len(row["metadata_sha256"]) == 64 for row in payload["examples"]["dev"])

    labels = build_task_labels_manifest(reversed(rows), tasks=("task-b", "task-a"))
    labels_json = task_labels_manifest_json(rows, tasks=("task-a", "task-b"), indent=None)
    assert json.loads(labels_json) == labels
    assert not {"text", "entities"} & _all_keys(labels)
    checksums = labels["checksums"]
    assert isinstance(checksums, dict)
    assert len(checksums["manifest_sha256"]) == 64


def test_selection_fails_clearly_when_a_partition_cannot_meet_quota() -> None:
    rows = _balanced_rows(tasks=("task-a",))
    with pytest.raises(PIIMBSelectionError, match="fewer than quota"):
        build_piimb_selection(
            rows,
            preset=PIIMBPreset("too-large", 3, 3),
            tasks=("task-a",),
            salt="unit-salt",
        )


def test_no_dataset_content_is_embedded_in_module_constants() -> None:
    # The adapter contains identity/configuration only; source rows enter exclusively
    # through the lazy loader or explicit test fixtures.
    import streamner_commit.datasets.piimb as module

    public_values: list[Any] = [
        value for name, value in vars(module).items() if name.isupper() and not name.startswith("_")
    ]
    assert all("Call Ada" not in repr(value) for value in public_values)
