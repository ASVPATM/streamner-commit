"""Export deterministic streaming oracle traces from the pinned CPU reference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from streamner_commit.backends.reference_gliner import ReferenceGLiNERBackend
from streamner_commit.reference.streaming_parity import (
    capture_streaming_parity_suite,
    write_streaming_parity_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, default=Path("configs/model_lock.json"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/fixtures/streaming_cases.json"),
    )
    parser.add_argument(
        "--chunk-units",
        type=int,
        nargs="+",
        choices=(1, 2, 4),
        default=[1],
        help="Whitespace-delimited chunk schedules to export (default: 1).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    network = parser.add_mutually_exclusive_group()
    network.add_argument("--local-files-only", action="store_true", default=True)
    network.add_argument("--allow-download", action="store_false", dest="local_files_only")
    return parser.parse_args()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(value)


def _locked_model(path: Path) -> tuple[str, str]:
    lock = _json_object(path, name="model lock")
    model_id = lock.get("model_id")
    revision = lock.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model lock must contain a nonblank model_id")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model lock must contain a 40-character revision")
    return model_id, revision


def main() -> None:
    args = parse_args()
    model_id, revision = _locked_model(args.model_lock)
    fixtures = _json_object(args.fixtures, name="fixture document")
    output_path = args.output or (
        Path("artifacts/reference") / revision / "streaming_parity" / "suite.json"
    )
    backend = ReferenceGLiNERBackend.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    payload = capture_streaming_parity_suite(
        backend.raw_model,
        fixture_document=fixtures,
        model_id=model_id,
        model_revision=revision,
        chunk_units=args.chunk_units,
        threshold=args.threshold,
    )
    report = write_streaming_parity_suite(payload, output_path)
    totals = payload["totals"]
    print(
        f"wrote {totals['case_condition_count']} case-conditions, "
        f"{totals['step_count']} steps, and {totals['span_update_count']} updates "
        f"to {report['path']} ({report['sha256']})"
    )


if __name__ == "__main__":
    main()
