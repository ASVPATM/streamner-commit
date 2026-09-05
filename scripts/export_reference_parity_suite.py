"""Export deterministic cold component fixtures for every synthetic debug case."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from streamner_commit.backends.reference_gliner import ReferenceGLiNERBackend
from streamner_commit.reference.parity import capture_reference_parity, write_parity_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, default=Path("configs/model_lock.json"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/fixtures/streaming_cases.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--local-files-only", action="store_true", default=True)
    network.add_argument("--allow-download", action="store_false", dest="local_files_only")
    return parser.parse_args()


def _object(path: Path, *, name: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(value)


def _model_lock(path: Path) -> tuple[str, str]:
    value = _object(path, name="model lock")
    model_id = value.get("model_id")
    revision = value.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model lock must contain a nonblank model_id")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model lock must contain a 40-character revision")
    return model_id, revision


def main() -> None:
    args = parse_args()
    model_id, revision = _model_lock(args.model_lock)
    fixture_document = _object(args.fixtures, name="fixture document")
    labels = fixture_document.get("labels")
    cases = fixture_document.get("cases")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
        raise ValueError("fixture labels must be a nonempty string list")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture cases must be a nonempty list")

    output_dir = args.output_dir or Path("artifacts/reference") / revision / "parity_suite"
    backend = ReferenceGLiNERBackend.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            raise TypeError("every fixture case must be an object")
        case_id = raw_case.get("id")
        text = raw_case.get("text")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate fixture ID: {case_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"fixture {case_id!r} must contain nonblank text")
        seen.add(case_id)
        fixture = capture_reference_parity(
            backend.raw_model,
            text=text,
            labels=labels,
            model_id=model_id,
            model_revision=revision,
        )
        case_dir = output_dir / case_id
        report = write_parity_fixture(fixture, case_dir)
        records.append(
            {
                "id": case_id,
                "directory": case_id,
                "array_count": report["array_count"],
                "arrays_sha256": report["arrays_sha256"],
                "metadata_sha256": report["metadata_sha256"],
            }
        )
        print(f"captured {case_id}: {report['array_count']} arrays")

    manifest = {
        "schema_version": 1,
        "kind": "synthetic_cold_reference_parity_suite",
        "model_id": model_id,
        "model_revision_sha": revision,
        "fixture_schema_version": fixture_document.get("schema_version"),
        "labels": labels,
        "case_count": len(records),
        "cases": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote suite manifest to {manifest_path}")


if __name__ == "__main__":
    main()
