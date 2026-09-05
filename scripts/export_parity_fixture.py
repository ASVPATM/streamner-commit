"""Export a tiny deterministic parity fixture from the pinned CPU reference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from streamner_commit.backends.reference_gliner import ReferenceGLiNERBackend
from streamner_commit.reference.parity import (
    DEFAULT_PARITY_LABELS,
    DEFAULT_PARITY_TEXT,
    capture_reference_parity,
    write_parity_fixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, default=Path("configs/model_lock.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--text", default=DEFAULT_PARITY_TEXT)
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Ordered label; repeat to override the two-label default.",
    )
    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Require an already cached checkpoint (default).",
    )
    network.add_argument(
        "--allow-download",
        action="store_false",
        dest="local_files_only",
        help="Explicitly allow Hugging Face downloads.",
    )
    return parser.parse_args()


def read_lock(path: Path) -> tuple[str, str]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("model lock must contain a JSON object")
    model_id = payload.get("model_id")
    revision = payload.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model lock model_id must be a nonblank string")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model lock revision must be a 40-character commit SHA")
    return model_id, revision


def main() -> None:
    args = parse_args()
    model_id, revision = read_lock(args.model_lock)
    output_dir = args.output_dir or Path("artifacts/reference") / revision / "parity"
    labels = tuple(args.labels) if args.labels is not None else DEFAULT_PARITY_LABELS

    backend = ReferenceGLiNERBackend.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    fixture = capture_reference_parity(
        backend.raw_model,
        text=args.text,
        labels=labels,
        model_id=model_id,
        model_revision=revision,
    )
    report = write_parity_fixture(fixture, output_dir)
    print(
        f"wrote {report['array_count']} arrays to {output_dir} "
        f"(npz sha256 {report['arrays_sha256']})"
    )


if __name__ == "__main__":
    main()
