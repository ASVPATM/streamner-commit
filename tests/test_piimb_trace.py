from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.datasets.piimb import (
    DEFAULT_SPLIT_SALT,
    PRIMARY_TASKS,
    build_piimb_selection,
    parent_split,
    parse_piimb_row,
)
from streamner_commit.datasets.piimb_trace import (
    PINNED_PIIMB_TRACE_SOURCE,
    PIIMBTraceSourceError,
    PIIMBTraceSourceLock,
    load_piimb_trace,
    load_piimb_trace_manifest,
    reconstruct_piimb_trace,
    validate_piimb_trace_manifest,
    verify_piimb_trace_source,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "experiments" / "manifests"
CHECKED_MANIFESTS = {
    "piimb_smoke.json": (
        "smoke",
        100,
        "d8477e6c76e2e828e9f7a3609c1aca1a91d50c127c2545663adce56a59d4c962",
    ),
    "piimb_research_small.json": (
        "research-small",
        1_600,
        "3aebf0ba390a4b15ebd20bec91b5d5de2c9ac50f048ff3f5c3e9a7bc695606bf",
    ),
    "piimb_research_full.json": (
        "research-full",
        5_000,
        "5d6222aff64ee866f17b6737b7bea145eee102ada4a635c263d8b63c437b8465",
    ),
}


def _checksum(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _refresh_checksums(document: dict[str, Any]) -> None:
    examples = document["examples"]
    partial = {
        "task_labels_sha256": _checksum(document["task_labels"]),
        "dev_row_ids_sha256": _checksum(
            [[row["uid"], row["source_row_index"]] for row in examples["dev"]]
        ),
        "test_row_ids_sha256": _checksum(
            [[row["uid"], row["source_row_index"]] for row in examples["test"]]
        ),
        "metadata_sha256": _checksum(examples),
    }
    body = copy.deepcopy(document)
    body["checksums"] = partial
    document["checksums"] = {**partial, "manifest_sha256": _checksum(body)}


def _checked_document(name: str = "piimb_smoke.json") -> dict[str, Any]:
    value = json.loads((MANIFEST_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _source_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _synthetic_source_and_manifest(
    tmp_path: Path,
) -> tuple[Path, PIIMBTraceSourceLock, dict[str, Any], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    examples = []
    for task in PRIMARY_TASKS:
        split_counts = {"dev": 0, "test": 0}
        candidate_index = 0
        while split_counts["dev"] < 8 or split_counts["test"] < 25:
            parent_id = f"{task}:synthetic-parent-{candidate_index}"
            split = parent_split(task, parent_id)
            target = 8 if split == "dev" else 25
            candidate_index += 1
            if split_counts[split] >= target:
                continue
            source_row_index = len(rows)
            uid = f"{task}:synthetic-{split}-{split_counts[split]}"
            if task == "nemotron-pii" and split_counts[split] == 0:
                uid = "synthetic-duplicate-uid"
            text = f"Row {source_row_index} belongs to {task}."
            label = f"{task}-label-{split_counts[split] % 2}"
            raw: dict[str, object] = {
                "uid": uid,
                "task_name": task,
                "source_dataset": f"synthetic/{task}",
                "source_uid": f"source-{source_row_index}",
                "parent_id": parent_id,
                "sentence_index": split_counts[split],
                "text": text,
                "entities": [{"start": 0, "end": 3, "label": label}],
                "language": "en",
            }
            rows.append(raw)
            examples.append(parse_piimb_row(raw, source_row_index=source_row_index))
            split_counts[split] += 1

    # Duplicate UIDs remain distinct because their global source row indices differ.
    gretel_first = next(row for row in rows if row["task_name"] == "gretel")
    gretel_first["uid"] = "synthetic-duplicate-uid"
    gretel_index = rows.index(gretel_first)
    examples[gretel_index] = parse_piimb_row(gretel_first, source_row_index=gretel_index)

    selection = build_piimb_selection(examples, preset="smoke")
    source_data = _source_bytes(rows)
    source_path = tmp_path / "test_sentences.jsonl"
    source_path.write_bytes(source_data)
    source_lock = PIIMBTraceSourceLock(
        size_bytes=len(source_data),
        sha256=hashlib.sha256(source_data).hexdigest(),
        rows=len(rows),
    )
    document = selection.manifest().to_dict()
    document["dataset"] = source_lock.dataset_manifest()
    _refresh_checksums(document)
    return source_path, source_lock, document, rows


def _local_pinned_source() -> Path:
    configured = os.environ.get("STREAMNER_PIIMB_SOURCE_JSONL")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/tmp/streamner-piimb-cache/test_sentences.jsonl"),
        REPOSITORY_ROOT / "data" / "test_sentences.jsonl",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    pytest.skip("the pinned PIIMB JSONL is not present locally")


@pytest.mark.parametrize("name", tuple(CHECKED_MANIFESTS))
def test_all_checked_manifests_validate_exact_preset_and_split_order(name: str) -> None:
    preset, expected_count, expected_sha = CHECKED_MANIFESTS[name]
    manifest = load_piimb_trace_manifest(MANIFEST_ROOT / name)

    assert manifest.preset.name == preset
    assert len(manifest.records) == expected_count
    assert manifest.manifest_sha256 == expected_sha
    assert manifest.task_labels_sha256 == (
        "37d4cedc6ebf5e1eb85cd55ae1c390b85c958c839ecd01ea7d8f39883a167082"
    )
    split_sequence = [record.split for record in manifest.records]
    assert split_sequence == sorted(split_sequence, key=("dev", "test").index)
    assert len({record.row_identity for record in manifest.records}) == expected_count


@pytest.mark.parametrize("name", tuple(CHECKED_MANIFESTS))
def test_real_checked_manifest_reconstructs_exact_gold_when_source_is_local(name: str) -> None:
    preset, expected_count, _expected_sha = CHECKED_MANIFESTS[name]
    reconstructed = load_piimb_trace(MANIFEST_ROOT / name, _local_pinned_source())

    assert reconstructed.manifest.preset.name == preset
    assert len(reconstructed.examples) == expected_count
    for selection_index, example in enumerate(reconstructed.examples):
        assert example.metadata["selection_index"] == selection_index
        assert example.metadata["benchmark_split"] == example.split
        assert example.labels == reconstructed.manifest.task_labels[
            str(example.metadata["task_name"])
        ]
        for entity in example.gold_entities:
            assert entity.example_id == example.example_id
            assert example.text[entity.start_char : entity.end_char] == entity.text
            assert entity.label in example.labels

    if preset == "research-full":
        duplicate_uid = "nemotron-pii:1b7796fb760b46088167f020d5c4422e_s16"
        duplicates = [
            example
            for example in reconstructed.examples
            if example.metadata["uid"] == duplicate_uid
        ]
        assert [example.metadata["source_row_index"] for example in duplicates] == [90573, 99743]
        assert duplicates[0].example_id != duplicates[1].example_id


def test_synthetic_reconstruction_preserves_row_identity_gold_and_full_vocab(
    tmp_path: Path,
) -> None:
    source_path, source_lock, document, _rows = _synthetic_source_and_manifest(tmp_path)
    manifest = validate_piimb_trace_manifest(document, source_lock=source_lock)
    reconstructed = reconstruct_piimb_trace(manifest, source_path, source_lock=source_lock)

    assert len(reconstructed.examples) == 100
    assert all(len(example.gold_entities) == 1 for example in reconstructed.examples)
    assert all(example.gold_entities[0].text == "Row" for example in reconstructed.examples)
    assert all(len(labels) == 2 for labels in manifest.task_labels.values())
    selected_identities = {
        (example.metadata["source_row_index"], example.metadata["uid"])
        for example in reconstructed.examples
    }
    assert len(selected_identities) == 100


def test_manifest_rejects_raw_fields_and_checksum_corruption() -> None:
    raw = _checked_document()
    raw["examples"]["dev"][0]["text"] = "licensed source must never be present"
    with pytest.raises(PIIMBTraceSourceError, match="forbidden raw fields"):
        validate_piimb_trace_manifest(raw)

    checksum = _checked_document()
    checksum["checksums"]["manifest_sha256"] = "0" * 64
    with pytest.raises(PIIMBTraceSourceError, match="manifest_sha256"):
        validate_piimb_trace_manifest(checksum)


def test_manifest_rejects_metadata_corruption_and_parent_partition_leak() -> None:
    metadata = _checked_document()
    metadata["examples"]["dev"][0]["source_uid"] = "changed"
    with pytest.raises(PIIMBTraceSourceError, match="metadata checksum"):
        validate_piimb_trace_manifest(metadata)

    partition = _checked_document()
    dev_parent = partition["examples"]["dev"][0]["parent_id"]
    test_record = partition["examples"]["test"][0]
    test_record["parent_id"] = dev_parent
    record_metadata = {
        key: value for key, value in test_record.items() if key != "metadata_sha256"
    }
    test_record["metadata_sha256"] = _checksum(record_metadata)
    partition["examples"]["test"].sort(
        key=lambda row: (
            PRIMARY_TASKS.index(row["task_name"]),
            hashlib.sha256(
                (
                    f"{DEFAULT_SPLIT_SALT}:sample:{row['task_name']}:"
                    f"{row['parent_id']}:{row['uid']}:{row['source_row_index']}"
                ).encode()
            ).hexdigest(),
            row["uid"],
        )
    )
    _refresh_checksums(partition)
    with pytest.raises(PIIMBTraceSourceError, match="stable test partition"):
        validate_piimb_trace_manifest(partition)


def test_source_verification_checks_size_sha_and_physical_row_count(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    data = b'{"one":1}\n{"two":2}\n'
    path.write_bytes(data)
    exact = PIIMBTraceSourceLock(
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), rows=2
    )
    assert verify_piimb_trace_source(path, source_lock=exact) == path.resolve()

    wrong_rows = PIIMBTraceSourceLock(
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), rows=3
    )
    with pytest.raises(PIIMBTraceSourceError, match="physical rows"):
        verify_piimb_trace_source(path, source_lock=wrong_rows)
    with pytest.raises(PIIMBTraceSourceError, match="source size"):
        verify_piimb_trace_source(path, source_lock=PINNED_PIIMB_TRACE_SOURCE)


def test_reconstruction_rejects_wrong_uid_at_the_locked_source_index(tmp_path: Path) -> None:
    _source_path, _source_lock, document, rows = _synthetic_source_and_manifest(tmp_path)
    selected_index = int(document["examples"]["dev"][0]["source_row_index"])
    rows[selected_index]["uid"] = "source-uid-was-corrupted"
    corrupted_data = _source_bytes(rows)
    corrupted_path = tmp_path / "corrupted.jsonl"
    corrupted_path.write_bytes(corrupted_data)
    corrupted_lock = PIIMBTraceSourceLock(
        size_bytes=len(corrupted_data),
        sha256=hashlib.sha256(corrupted_data).hexdigest(),
        rows=len(rows),
    )
    document["dataset"] = corrupted_lock.dataset_manifest()
    _refresh_checksums(document)
    manifest = validate_piimb_trace_manifest(document, source_lock=corrupted_lock)

    with pytest.raises(PIIMBTraceSourceError, match="source identity"):
        reconstruct_piimb_trace(manifest, corrupted_path, source_lock=corrupted_lock)


def test_reconstruction_rejects_manifest_vocab_not_rescanned_from_full_source(
    tmp_path: Path,
) -> None:
    source_path, source_lock, document, _rows = _synthetic_source_and_manifest(tmp_path)
    task = PRIMARY_TASKS[0]
    document["task_labels"][task] = sorted([*document["task_labels"][task], "bogus-label"])
    _refresh_checksums(document)
    manifest = validate_piimb_trace_manifest(document, source_lock=source_lock)

    with pytest.raises(PIIMBTraceSourceError, match="full source vocabulary"):
        reconstruct_piimb_trace(manifest, source_path, source_lock=source_lock)
