from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_REVISION,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PRIMARY_TASKS,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "experiments" / "manifests"
EXPECTED_FILES = {
    "piimb_task_labels.json": "6db538ee1dcac67f5f45dd03b26728d461d36584238116b3853622a18d43ff58",
    "piimb_smoke.json": "cab636c18cd298500cb3b5d77c1885a04ee16471de61fefdbb5001076e12359d",
    "piimb_research_small.json": "478ba9e2b10a505f111ac181f33c70e7c570776e23d8a47dc6f778ec169a1553",
    "piimb_research_full.json": "916842aa702916d4aabff9b207992044bedbca884fca9f4151153b93a6c1e8e2",
}
EXPECTED_COUNTS = {
    "piimb_smoke.json": (5, 20),
    "piimb_research_small.json": (100, 300),
    "piimb_research_full.json": (250, 1_000),
}


def _load(name: str) -> dict[str, Any]:
    path = MANIFEST_ROOT / name
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == EXPECTED_FILES[name]
    value = json.loads(data)
    assert isinstance(value, dict)
    return value


def _assert_no_raw_keys(value: object) -> None:
    if isinstance(value, dict):
        assert not ({"text", "entities"} & set(value))
        for nested in value.values():
            _assert_no_raw_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_raw_keys(nested)


def test_checked_in_piimb_manifests_are_locked_reproducible_and_content_free() -> None:
    labels = _load("piimb_task_labels.json")
    assert labels["dataset"] == {
        "id": PIIMB_DATASET_ID,
        "revision": PIIMB_REVISION,
        "source_file": {
            "path": "data/test_sentences.jsonl",
            "rows": PIIMB_SOURCE_ROW_COUNT,
            "sha256": PIIMB_SOURCE_SHA256,
            "size_bytes": 60_412_185,
        },
        "split": "test",
        "subset": "sentences",
    }
    assert labels["tasks"] == sorted(PRIMARY_TASKS)
    assert {task: len(values) for task, values in labels["task_labels"].items()} == {
        "ai4privacy-en": 19,
        "gretel": 42,
        "nemotron-pii": 55,
        "privy": 24,
    }
    assert labels["checksums"]["task_labels_sha256"] == (
        "37d4cedc6ebf5e1eb85cd55ae1c390b85c958c839ecd01ea7d8f39883a167082"
    )

    for name, (dev_per_task, test_per_task) in EXPECTED_COUNTS.items():
        manifest = _load(name)
        _assert_no_raw_keys(manifest)
        assert manifest["dataset"]["revision"] == PIIMB_REVISION
        assert manifest["dataset"]["source_file"]["sha256"] == PIIMB_SOURCE_SHA256
        assert manifest["task_labels"] == labels["task_labels"]
        assert manifest["source_diagnostics"] == {
            "conflicting_uid_count": 1_227,
            "duplicate_uid_count": 1_302,
            "duplicate_uid_occurrences": 1_302,
            "exact_duplicate_occurrences": 75,
            "row_count": 121_409,
            "unique_uid_count": 120_107,
        }
        rows = manifest["examples"]
        assert len(rows["dev"]) == dev_per_task * len(PRIMARY_TASKS)
        assert len(rows["test"]) == test_per_task * len(PRIMARY_TASKS)
        identities = [(row["uid"], row["source_row_index"]) for row in rows["dev"] + rows["test"]]
        assert len(identities) == len(set(identities))
        for task in PRIMARY_TASKS:
            dev = [row for row in rows["dev"] if row["task_name"] == task]
            test = [row for row in rows["test"] if row["task_name"] == task]
            assert len(dev) == dev_per_task
            assert len(test) == test_per_task
            assert {row["parent_id"] for row in dev}.isdisjoint(row["parent_id"] for row in test)
