from __future__ import annotations

import json
import os
import runpy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SPLIT,
    PIIMB_SUBSET,
    PRIMARY_TASKS,
    PIIMBSelectionError,
    parent_split,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "piimb.json"
SCRIPT = runpy.run_path(str(ROOT / "scripts" / "prepare_piimb.py"))
PRESET_FILES: dict[str, str] = SCRIPT["PRESET_FILES"]
_atomic_write_set = SCRIPT["_atomic_write_set"]
_normalize_presets = SCRIPT["_normalize_presets"]
prepare_piimb_manifests = SCRIPT["prepare_piimb_manifests"]


class FakeDatasetLoader:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Iterable[Mapping[str, object]]:
        self.calls.append(dict(kwargs))
        return (dict(row) for row in self.rows)


def _fake_rows(*, dev_per_task: int, test_per_task: int) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for task in PRIMARY_TASKS:
        remaining = {"dev": dev_per_task, "test": test_per_task}
        accepted = 0
        candidate = 0
        while any(remaining.values()):
            parent_id = f"{task}-parent-{candidate:05d}"
            split = parent_split(task, parent_id)
            candidate += 1
            if remaining[split] == 0:
                continue
            uid = f"{task}-uid-{accepted:05d}"
            text = f"RAW-CONTENT-{task}-{accepted:05d}-DO-NOT-SERIALIZE"
            entities: list[dict[str, object]] = []
            if accepted < 2:
                entities.append(
                    {
                        "start": 0,
                        "end": 3,
                        "label": f"{task}-label-{accepted}",
                    }
                )
            rows.append(
                {
                    "uid": uid,
                    "task_name": task,
                    "source_dataset": "unit-fixture",
                    "source_uid": f"source-{uid}",
                    "parent_id": parent_id,
                    "sentence_index": 0,
                    "text": text,
                    "entities": entities,
                    "language": "en",
                }
            )
            remaining[split] -= 1
            accepted += 1
    duplicate_indices = [
        index for index, row in enumerate(rows) if row["task_name"] == "nemotron-pii"
    ]
    rows[duplicate_indices[1]]["uid"] = rows[duplicate_indices[0]]["uid"]
    return tuple(rows)


def _read(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_prepare_loads_once_and_writes_only_deterministic_sanitized_manifests(
    tmp_path: Path,
) -> None:
    rows = _fake_rows(dev_per_task=105, test_per_task=305)
    loader = FakeDatasetLoader(rows)
    output = tmp_path / "first"

    written = prepare_piimb_manifests(
        config_path=CONFIG,
        output_dir=output,
        presets=("research-small", "smoke", "smoke"),
        cache_dir=tmp_path / "cache",
        streaming=True,
        load_dataset_fn=loader,
    )

    assert list(written) == [
        "piimb_task_labels.json",
        "piimb_smoke.json",
        "piimb_research_small.json",
    ]
    assert len(loader.calls) == 1
    assert loader.calls[0] == {
        "path": PIIMB_DATASET_ID,
        "name": PIIMB_SUBSET,
        "split": PIIMB_SPLIT,
        "revision": PIIMB_REVISION,
        "streaming": True,
        "cache_dir": str(tmp_path / "cache"),
    }
    assert all(path.is_file() for path in written.values())
    assert not (output / PRESET_FILES["research_full"]).exists()

    labels = _read(output / "piimb_task_labels.json")
    assert labels["tasks"] == list(PRIMARY_TASKS)
    expected_source = {
        "path": PIIMB_SOURCE_FILE,
        "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
        "sha256": PIIMB_SOURCE_SHA256,
        "rows": PIIMB_SOURCE_ROW_COUNT,
    }
    assert labels["dataset"]["source_file"] == expected_source
    assert labels["task_labels"] == {
        task: [f"{task}-label-0", f"{task}-label-1"] for task in PRIMARY_TASKS
    }
    manifests = {
        "smoke": _read(output / PRESET_FILES["smoke"]),
        "research_small": _read(output / PRESET_FILES["research_small"]),
    }
    assert manifests["smoke"]["counts"] | {} == {
        "total": 100,
        "dev": 20,
        "test": 80,
        "by_task": {task: {"dev": 5, "test": 20} for task in PRIMARY_TASKS},
    }
    assert manifests["research_small"]["counts"] | {} == {
        "total": 1600,
        "dev": 400,
        "test": 1200,
        "by_task": {task: {"dev": 100, "test": 300} for task in PRIMARY_TASKS},
    }
    source_by_row_id = {
        (str(row["uid"]), source_row_index): row for source_row_index, row in enumerate(rows)
    }
    for manifest in manifests.values():
        assert manifest["dataset"]["source_file"] == expected_source
        assert manifest["task_labels"] == labels["task_labels"]
        assert manifest["source_diagnostics"]["duplicate_uid_occurrences"] == 1
        assert manifest["source_diagnostics"]["conflicting_uid_count"] == 1
        dev = manifest["examples"]["dev"]
        test = manifest["examples"]["test"]
        assert {row["parent_id"] for row in dev}.isdisjoint(row["parent_id"] for row in test)
        for task in PRIMARY_TASKS:
            selected = [row for row in (*dev, *test) if row["task_name"] == task]
            assert any(
                not source_by_row_id[(row["uid"], row["source_row_index"])]["entities"]
                for row in selected
            )
        assert all(len(row["metadata_sha256"]) == 64 for row in (*dev, *test))

    first_bytes = {name: path.read_bytes() for name, path in written.items()}
    serialized = b"".join(first_bytes.values())
    assert b"RAW-CONTENT-" not in serialized
    assert b'"text"' not in serialized
    assert b'"entities"' not in serialized

    second_loader = FakeDatasetLoader(rows)
    second = prepare_piimb_manifests(
        config_path=CONFIG,
        output_dir=tmp_path / "second",
        presets=("smoke", "research_small"),
        load_dataset_fn=second_loader,
    )
    assert len(second_loader.calls) == 1
    assert {name: path.read_bytes() for name, path in second.items()} == first_bytes


def test_all_presets_are_default_and_quota_failure_publishes_nothing(tmp_path: Path) -> None:
    assert _normalize_presets(None) == ("smoke", "research_small", "research_full")
    assert _normalize_presets(("research-full", "smoke")) == ("smoke", "research_full")
    loader = FakeDatasetLoader(_fake_rows(dev_per_task=6, test_per_task=21))
    output = tmp_path / "manifests"

    with pytest.raises(PIIMBSelectionError, match="fewer than quota"):
        prepare_piimb_manifests(
            config_path=CONFIG,
            output_dir=output,
            presets=("smoke", "research_small"),
            load_dataset_fn=loader,
        )

    assert len(loader.calls) == 1
    assert not list(output.glob("*.json"))


def test_atomic_write_set_restores_existing_files_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "manifests"
    output.mkdir()
    first = output / "a.json"
    second = output / "b.json"
    first.write_bytes(b"old-a")
    second.write_bytes(b"old-b")
    real_replace = os.replace
    failed = False

    def fail_second_staged_replace(source: str | Path, target: str | Path) -> None:
        nonlocal failed
        source_path = Path(source)
        if not failed and source_path.parent.name == "staged" and source_path.name == "b.json":
            failed = True
            raise OSError("injected commit failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_second_staged_replace)
    with pytest.raises(OSError, match="injected commit failure"):
        _atomic_write_set(output, {"a.json": b"new-a", "b.json": b"new-b"})

    assert first.read_bytes() == b"old-a"
    assert second.read_bytes() == b"old-b"
    assert not list(output.glob(".prepare-piimb-*"))
