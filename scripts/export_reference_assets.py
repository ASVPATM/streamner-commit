"""Export the locked StreamingSpan checkpoint into local safe MLX assets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from streamner_commit.reference.exporter import export_reference_assets

DEFAULT_MODEL_LOCK = Path("configs/model_lock.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    parser.add_argument("--model-id", help="Override the model-lock ID")
    parser.add_argument("--revision", help="Override the model-lock commit SHA")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    network = parser.add_mutually_exclusive_group()
    network.add_argument(
        "--local-files-only",
        action="store_true",
        dest="local_files_only",
        help="Require an already-cached snapshot (the default)",
    )
    network.add_argument(
        "--allow-network",
        action="store_false",
        dest="local_files_only",
        help="Allow Hugging Face Hub downloads for missing locked files",
    )
    parser.set_defaults(local_files_only=True)
    return parser.parse_args()


def _locked_model(args: argparse.Namespace) -> tuple[str, str]:
    try:
        payload = json.loads(args.model_lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read model lock: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("model lock must contain a JSON object")
    model_id = args.model_id or payload.get("model_id")
    revision = args.revision or payload.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model ID is missing from arguments and model lock")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model revision is missing from arguments and model lock")
    return model_id, revision


def main() -> None:
    args = parse_args()
    model_id, revision = _locked_model(args)
    result = export_reference_assets(
        output_root=args.output_root,
        model_id=model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
