"""Export the pinned PIIMB smoke oracle with one cold append per sentence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.backends.reference_gliner import ReferenceGLiNERBackend
from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
)
from streamner_commit.reference.piimb_parity import (
    capture_piimb_reference_smoke,
    load_piimb_smoke_manifest,
    write_piimb_reference_smoke,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("experiments/manifests/piimb_smoke.json")
DEFAULT_MODEL_LOCK = Path("configs/model_lock.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    parser.add_argument("--source-jsonl", type=Path, default=Path(PIIMB_SOURCE_FILE))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the exact pinned dataset object and model revision to be downloaded.",
    )
    parser.add_argument(
        "--skip-reference-verification",
        action="store_true",
        help="Skip the observer's public-score cross-check (diagnostic use only).",
    )
    return parser.parse_args(argv)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


def _locked_model(path: Path) -> tuple[str, str]:
    lock = _json_object(path, name="model lock")
    if set(lock) != {"model_id", "revision"}:
        raise ValueError("model lock must contain exactly model_id and revision")
    model_id = lock["model_id"]
    revision = lock["revision"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model lock model_id must be nonblank")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("model lock revision must be a lowercase 40-character commit SHA")
    return model_id, revision


def _resolve_source(
    path: Path,
    *,
    allow_download: bool,
    cache_dir: Path | None,
) -> Path:
    if path.is_file():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"pinned PIIMB source is absent at {path}; pass --allow-download to fetch it"
        )
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=PIIMB_DATASET_ID,
        repo_type="dataset",
        filename=PIIMB_SOURCE_FILE,
        revision=PIIMB_REVISION,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    return Path(downloaded)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_id, revision = _locked_model(args.model_lock)
    manifest = load_piimb_smoke_manifest(args.manifest)
    source_path = _resolve_source(
        args.source_jsonl,
        allow_download=args.allow_download,
        cache_dir=args.cache_dir,
    )
    output_path = args.output or (
        Path("artifacts/reference") / revision / "piimb_parity" / "smoke.json"
    )

    backend = ReferenceGLiNERBackend.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    payload = capture_piimb_reference_smoke(
        backend.raw_model,
        manifest=manifest,
        source_path=source_path,
        model_id=model_id,
        model_revision=revision,
        verify_reference=not args.skip_reference_verification,
    )
    report = write_piimb_reference_smoke(
        payload,
        output_path,
        repository_root=REPOSITORY_ROOT,
    )
    totals = payload["totals"]
    print(
        f"wrote {totals['case_count']} PIIMB cases, {totals['step_count']} steps, "
        f"and {totals['span_update_count']} updates to {report['path']} "
        f"({report['sha256']})"
    )


if __name__ == "__main__":
    main()
