"""Build deterministic, text-free PIIMB task and sampling manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.datasets.piimb import (
    DEFAULT_SPLIT_SALT,
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
    SPLIT_ALGORITHM,
    PIIMBPreset,
    build_piimb_selection,
    load_piimb,
    task_labels_manifest_json,
)

DEFAULT_CONFIG = Path("configs/piimb.json")
DEFAULT_OUTPUT_DIR = Path("experiments/manifests")
PRESET_FILES = {
    "smoke": "piimb_smoke.json",
    "research_small": "piimb_research_small.json",
    "research_full": "piimb_research_full.json",
}
_PRESET_COUNTS = {
    "smoke": {"development_sentences_per_task": 5, "test_sentences_per_task": 20},
    "research_small": {
        "development_sentences_per_task": 100,
        "test_sentences_per_task": 300,
    },
    "research_full": {
        "development_sentences_per_task": 250,
        "test_sentences_per_task": 1000,
    },
}
_EXAMPLE_FIELDS = {
    "uid",
    "task_name",
    "source_dataset",
    "source_uid",
    "parent_id",
    "sentence_index",
    "language",
    "source_row_index",
    "metadata_sha256",
}


class PreparePIIMBError(ValueError):
    """The locked configuration or generated manifest violated the CLI contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Preset to build; repeat for multiple presets (default: all).",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args(argv)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PreparePIIMBError(f"{name} must contain a JSON object")
    return dict(value)


def _load_locked_config(
    path: Path,
) -> tuple[tuple[str, ...], str, dict[str, PIIMBPreset]]:
    config = _json_object(path, name="PIIMB configuration")
    expected_dataset = {
        "id": PIIMB_DATASET_ID,
        "subset": PIIMB_SUBSET,
        "split": PIIMB_SPLIT,
        "revision": PIIMB_REVISION,
        "license": PIIMB_LICENSE,
        "source_path": PIIMB_SOURCE_FILE,
        "source_size_bytes": PIIMB_SOURCE_SIZE_BYTES,
        "source_sha256": PIIMB_SOURCE_SHA256,
        "source_rows": PIIMB_SOURCE_ROW_COUNT,
    }
    expected_partition = {
        "algorithm": SPLIT_ALGORITHM,
        "salt": DEFAULT_SPLIT_SALT,
        "development_percent": 20,
        "test_percent": 80,
    }
    if config.get("schema_version") != 1:
        raise PreparePIIMBError("PIIMB configuration schema_version must be 1")
    if config.get("dataset") != expected_dataset:
        raise PreparePIIMBError("PIIMB dataset lock differs from the adapter constants")
    if config.get("primary_tasks") != list(PRIMARY_TASKS):
        raise PreparePIIMBError("PIIMB primary task order differs from the adapter constants")
    if config.get("partition") != expected_partition:
        raise PreparePIIMBError("PIIMB partition configuration differs from the locked policy")
    if config.get("presets") != _PRESET_COUNTS:
        raise PreparePIIMBError("PIIMB preset sizes differ from the locked policy")
    presets = {
        name: PIIMBPreset(
            name=name.replace("_", "-"),
            dev_per_task=counts["development_sentences_per_task"],
            test_per_task=counts["test_sentences_per_task"],
        )
        for name, counts in _PRESET_COUNTS.items()
    }
    return PRIMARY_TASKS, DEFAULT_SPLIT_SALT, presets


def _normalize_presets(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple(PRESET_FILES)
    requested: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise PreparePIIMBError("preset names must be nonblank strings")
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in PRESET_FILES:
            raise PreparePIIMBError(f"unknown PIIMB preset: {value!r}")
        requested.add(normalized)
    if not requested:
        raise PreparePIIMBError("at least one PIIMB preset is required")
    return tuple(name for name in PRESET_FILES if name in requested)


def _sanitized_json_bytes(payload: str, *, name: str) -> bytes:
    try:
        document: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PreparePIIMBError(f"{name} is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise PreparePIIMBError(f"{name} must be a JSON object")

    def reject_raw_fields(value: object) -> None:
        if isinstance(value, Mapping):
            forbidden = {str(key) for key in value} & {"text", "entities"}
            if forbidden:
                raise PreparePIIMBError(f"{name} contains forbidden raw fields: {forbidden}")
            for nested in value.values():
                reject_raw_fields(nested)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for nested in value:
                reject_raw_fields(nested)

    reject_raw_fields(document)
    examples = document.get("examples")
    if examples is not None:
        if not isinstance(examples, Mapping) or set(examples) != {"dev", "test"}:
            raise PreparePIIMBError(f"{name} has an invalid examples object")
        for split, rows in examples.items():
            if not isinstance(rows, list):
                raise PreparePIIMBError(f"{name} examples.{split} must be a list")
            for row in rows:
                if not isinstance(row, Mapping) or set(row) != _EXAMPLE_FIELDS:
                    raise PreparePIIMBError(f"{name} contains an unsafe example row")
    normalized = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return (normalized + "\n").encode("utf-8")


def _atomic_write_set(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transaction = Path(tempfile.mkdtemp(prefix=".prepare-piimb-", dir=output_dir))
    staged = transaction / "staged"
    backups = transaction / "backups"
    staged.mkdir()
    backups.mkdir()
    committed: list[tuple[Path, Path, bool]] = []
    try:
        for filename, payload in payloads.items():
            target = output_dir / filename
            if target.exists() and not target.is_file():
                raise PreparePIIMBError(f"manifest target is not a regular file: {target}")
            with (staged / filename).open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for filename in payloads:
            target = output_dir / filename
            backup = backups / filename
            existed = target.exists()
            if existed:
                os.replace(target, backup)
            try:
                os.replace(staged / filename, target)
            except Exception:
                if existed:
                    os.replace(backup, target)
                raise
            committed.append((target, backup, existed))
    except Exception:
        for target, backup, existed in reversed(committed):
            if existed:
                os.replace(backup, target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(transaction, ignore_errors=True)


def prepare_piimb_manifests(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    presets: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    load_dataset_fn: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> dict[str, Path]:
    """Load PIIMB once, validate every selection, then publish sanitized manifests."""

    tasks, salt, configured_presets = _load_locked_config(Path(config_path))
    selected = _normalize_presets(presets)
    rows = load_piimb(
        tasks=tasks,
        cache_dir=cache_dir,
        streaming=streaming,
        load_dataset_fn=load_dataset_fn,
    )

    payloads = {
        "piimb_task_labels.json": _sanitized_json_bytes(
            task_labels_manifest_json(rows, tasks=tasks, indent=2),
            name="PIIMB task-label manifest",
        )
    }
    for preset in selected:
        selection = build_piimb_selection(
            rows,
            preset=configured_presets[preset],
            tasks=tasks,
            salt=salt,
        )
        payloads[PRESET_FILES[preset]] = _sanitized_json_bytes(
            selection.manifest().to_json(indent=2),
            name=f"PIIMB {preset} manifest",
        )

    destination = Path(output_dir)
    _atomic_write_set(destination, payloads)
    return {filename: destination / filename for filename in payloads}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    written = prepare_piimb_manifests(
        config_path=args.config,
        output_dir=args.output_dir,
        presets=args.presets,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    print(f"wrote {len(written)} sanitized PIIMB manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
