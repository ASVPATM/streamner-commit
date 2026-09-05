"""Capture sanitized project provenance and lock the model revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamner_commit.provenance import capture_provenance, resolve_model_revision

DEFAULT_MODEL_ID = "knowledgator/gliner-stream-pii-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        help="Model ref to resolve; defaults to the Hub default branch",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--model-lock", type=Path, help="Optional model-lock JSON path")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    resolved_revision = resolve_model_revision(args.model_id, args.revision)
    report = capture_provenance(
        repository,
        model_id=args.model_id,
        model_revision=resolved_revision,
    )

    if args.output:
        write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.model_lock:
        write_json(
            args.model_lock,
            {"model_id": args.model_id, "revision": resolved_revision},
        )


if __name__ == "__main__":
    main()
