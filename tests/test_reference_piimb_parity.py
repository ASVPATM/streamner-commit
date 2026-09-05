from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.datasets.piimb import (
    DEFAULT_SPLIT_SALT,
    DEV_PERCENT,
    PRIMARY_TASKS,
    SAMPLE_ALGORITHM,
    SPLIT_ALGORITHM,
    TEST_PERCENT,
)
from streamner_commit.reference import piimb_parity

REVISION = "e871777dc4b3b688747a0433fff8d94a36fcc7b0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def _manifest_record(row: dict[str, Any], source_row_index: int) -> dict[str, Any]:
    metadata = {
        "uid": row["uid"],
        "source_row_index": source_row_index,
        "task_name": row["task_name"],
        "source_dataset": row["source_dataset"],
        "source_uid": row["source_uid"],
        "parent_id": row["parent_id"],
        "sentence_index": row["sentence_index"],
        "language": row["language"],
    }
    return {**metadata, "metadata_sha256": _checksum(metadata)}


def _fake_source_and_manifest(
    tmp_path: Path,
) -> tuple[Path, piimb_parity.PIIMBSourceLock, dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    task_labels: dict[str, list[str]] = {}

    for task in PRIMARY_TASKS:
        selected_label = f"{task}-selected"
        full_only_label = f"{task}-vocabulary-only"
        task_labels[task] = sorted([selected_label, full_only_label])
        for local_index in range(26):
            source_row_index = len(source_rows)
            split = "dev" if local_index < 5 else "test"
            selected = local_index < 25
            uid = f"{task}-uid-{local_index}"
            if task == PRIMARY_TASKS[0] and local_index in (0, 1):
                uid = "intentionally-duplicated-uid"
            text = f"Row {source_row_index} belongs to {task}."
            label = selected_label if selected else full_only_label
            row = {
                "uid": uid,
                "task_name": task,
                "source_dataset": f"source-{task}",
                "source_uid": f"source-uid-{source_row_index}",
                "parent_id": f"{task}-{split}-parent-{local_index}",
                "sentence_index": local_index,
                "text": text,
                "entities": [{"start": 0, "end": 3, "label": label}],
                "language": "en",
            }
            source_rows.append(row)
            if selected:
                record = _manifest_record(row, source_row_index)
                (dev if split == "dev" else test).append(record)

    source_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in source_rows
    )
    source_path = tmp_path / "test_sentences.jsonl"
    source_path.write_bytes(source_bytes)
    source_lock = piimb_parity.PIIMBSourceLock(
        size_bytes=len(source_bytes),
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        rows=len(source_rows),
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "dataset": source_lock.manifest_dataset(),
        "sampling": {
            "preset": "smoke",
            "split_algorithm": SPLIT_ALGORITHM,
            "sample_algorithm": SAMPLE_ALGORITHM,
            "split_salt": DEFAULT_SPLIT_SALT,
            "dev_percent": DEV_PERCENT,
            "test_percent": TEST_PERCENT,
            "requested_per_task": {"dev": 5, "test": 20},
        },
        "tasks": list(PRIMARY_TASKS),
        "task_labels": task_labels,
        "source_diagnostics": {
            "row_count": len(source_rows),
            "unique_uid_count": len(source_rows) - 1,
            "duplicate_uid_count": 1,
            "duplicate_uid_occurrences": 1,
            "conflicting_uid_count": 1,
            "exact_duplicate_occurrences": 0,
        },
        "counts": {
            "total": 100,
            "dev": 20,
            "test": 80,
            "by_task": {task: {"dev": 5, "test": 20} for task in PRIMARY_TASKS},
        },
        "examples": {"dev": dev, "test": test},
        "checksums": {},
    }
    _refresh_checksums(document)
    return source_path, source_lock, document


def test_manifest_reconstruction_uses_row_identity_and_full_task_vocabulary(
    tmp_path: Path,
) -> None:
    source_path, source_lock, document = _fake_source_and_manifest(tmp_path)

    manifest = piimb_parity.validate_piimb_smoke_manifest(
        document,
        source_lock=source_lock,
    )
    reconstructed = piimb_parity.reconstruct_piimb_smoke(
        manifest,
        source_path,
        source_lock=source_lock,
    )

    assert len(reconstructed.rows) == 100
    duplicate_rows = [
        example
        for record, example in reconstructed.rows
        if record.uid == "intentionally-duplicated-uid"
    ]
    assert len(duplicate_rows) == 2
    assert duplicate_rows[0].source_row_index != duplicate_rows[1].source_row_index
    assert duplicate_rows[0].text != duplicate_rows[1].text
    for task in PRIMARY_TASKS:
        assert manifest.task_labels[task] == (
            f"{task}-selected",
            f"{task}-vocabulary-only",
        )
    encoded_manifest = json.dumps(document, sort_keys=True)
    assert '"text"' not in encoded_manifest
    assert '"entities"' not in encoded_manifest


def test_checked_in_smoke_manifest_matches_the_pinned_primary_task_scan() -> None:
    manifest = piimb_parity.load_piimb_smoke_manifest(
        REPOSITORY_ROOT / "experiments/manifests/piimb_smoke.json"
    )

    assert len(manifest.records) == 100
    assert {record.task_name for record in manifest.records} == set(PRIMARY_TASKS)
    assert manifest.manifest_sha256 == (
        "d8477e6c76e2e828e9f7a3609c1aca1a91d50c127c2545663adce56a59d4c962"
    )


def test_capture_is_single_append_variable_label_and_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path, source_lock, document = _fake_source_and_manifest(tmp_path)
    manifest = piimb_parity.validate_piimb_smoke_manifest(
        document,
        source_lock=source_lock,
    )
    model = object()
    calls: list[dict[str, Any]] = []

    def fake_capture(captured_model: object, **kwargs: Any) -> dict[str, Any]:
        assert captured_model is model
        assert kwargs["chunk_units"] is None
        assert kwargs["single_chunk"] is True
        assert kwargs["threshold"] == 0.5
        calls.append(kwargs)
        return {
            "id": kwargs["case_id"],
            "full_text": kwargs["text"],
            "labels": list(kwargs["labels"]),
            "chunk_units": None,
            "chunks": [kwargs["text"]],
            "step_count": 1,
            "span_update_count": 1,
            "steps": [{"validated_public_score_count": 1}],
            "final_state": {"span_count": 1},
            "chunk_mode": "single",
        }

    monkeypatch.setattr(piimb_parity, "capture_streaming_case", fake_capture)

    def run_capture() -> dict[str, Any]:
        return piimb_parity.capture_piimb_reference_smoke(
            model,
            manifest=manifest,
            source_path=source_path,
            model_id="example/pinned-model",
            model_revision=REVISION,
            verify_reference=False,
            source_lock=source_lock,
        )

    first = run_capture()
    second = run_capture()

    assert first == second
    assert len(calls) == 200
    assert first["selection"]["split_counts"] == {"dev": 20, "test": 80}
    assert first["selection"]["task_counts"] == {task: 25 for task in PRIMARY_TASKS}
    assert first["totals"] == {
        "case_count": 100,
        "step_count": 100,
        "span_update_count": 100,
        "final_span_count": 100,
        "validated_public_score_count": 100,
    }
    for case in first["cases"]:
        selection = case["selection"]
        trace = case["trace"]
        assert trace["labels"] == first["task_labels"][selection["task_name"]]
        assert trace["chunk_mode"] == "single"
        assert set(selection).isdisjoint({"text", "entities", "annotations"})

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_report = piimb_parity.write_piimb_reference_smoke(first, first_path)
    second_report = piimb_parity.write_piimb_reference_smoke(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_report["sha256"] == second_report["sha256"]


def test_source_and_manifest_mismatches_fail_closed(tmp_path: Path) -> None:
    source_path, source_lock, document = _fake_source_and_manifest(tmp_path)
    piimb_parity.validate_piimb_smoke_manifest(
        document,
        source_lock=source_lock,
    )

    corrupted_source = tmp_path / "corrupted.jsonl"
    corrupted = bytearray(source_path.read_bytes())
    corrupted[1] = ord("X") if corrupted[1] != ord("X") else ord("Y")
    corrupted_source.write_bytes(corrupted)
    with pytest.raises(piimb_parity.PIIMBParityError, match="SHA-256"):
        piimb_parity.verify_piimb_source_file(
            corrupted_source,
            source_lock=source_lock,
        )

    wrong_uid = copy.deepcopy(document)
    wrong_uid["examples"]["dev"][0]["uid"] = "not-the-source-uid"
    metadata = {
        key: value
        for key, value in wrong_uid["examples"]["dev"][0].items()
        if key != "metadata_sha256"
    }
    wrong_uid["examples"]["dev"][0]["metadata_sha256"] = _checksum(metadata)
    _refresh_checksums(wrong_uid)
    wrong_manifest = piimb_parity.validate_piimb_smoke_manifest(
        wrong_uid,
        source_lock=source_lock,
    )
    with pytest.raises(piimb_parity.PIIMBParityError, match="uid differs"):
        piimb_parity.reconstruct_piimb_smoke(
            wrong_manifest,
            source_path,
            source_lock=source_lock,
        )

    wrong_vocabulary = copy.deepcopy(document)
    wrong_vocabulary["task_labels"][PRIMARY_TASKS[0]].pop()
    _refresh_checksums(wrong_vocabulary)
    vocabulary_manifest = piimb_parity.validate_piimb_smoke_manifest(
        wrong_vocabulary,
        source_lock=source_lock,
    )
    with pytest.raises(piimb_parity.PIIMBParityError, match="vocabulary differs"):
        piimb_parity.reconstruct_piimb_smoke(
            vocabulary_manifest,
            source_path,
            source_lock=source_lock,
        )

    bad_checksum = copy.deepcopy(document)
    bad_checksum["checksums"]["metadata_sha256"] = "0" * 64
    with pytest.raises(piimb_parity.PIIMBParityError, match="metadata_sha256"):
        piimb_parity.validate_piimb_smoke_manifest(
            bad_checksum,
            source_lock=source_lock,
        )


def test_writer_refuses_nonignored_repository_destination(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    payload = {"schema_version": 1}

    with pytest.raises(piimb_parity.PIIMBParityError, match="licensed text"):
        piimb_parity.write_piimb_reference_smoke(
            payload,
            repository / "checked-in.json",
            repository_root=repository,
        )

    allowed = repository / "artifacts" / "reference" / "oracle.json"
    piimb_parity.write_piimb_reference_smoke(
        payload,
        allowed,
        repository_root=repository,
    )
    assert allowed.is_file()
