"""Run readable public StreamingSpan traces against the pinned reference model."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.reference import format_public_trace, run_public_trace

DEFAULT_FIXTURES = Path("data/fixtures/streaming_cases.json")
DEFAULT_MODEL_LOCK = Path("configs/model_lock.json")
DEFAULT_LABELS = (
    "person",
    "email address",
    "phone number",
    "street address",
    "credit card number",
    "passport number",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    parser.add_argument("--model-id", help="Override the locked model ID")
    parser.add_argument("--revision", help="Override the locked immutable revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--words-per-chunk", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--example-id", action="append", dest="example_ids")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _locked_model(args: argparse.Namespace) -> tuple[str, str]:
    lock: Mapping[str, object] = {}
    if args.model_lock.exists():
        loaded = _read_json(args.model_lock)
        if not isinstance(loaded, Mapping):
            raise ValueError("model lock must be a JSON object")
        lock = loaded
    model_id = args.model_id or lock.get("model_id")
    revision = args.revision or lock.get("revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model ID is missing; provide --model-id or a valid model lock")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("model revision is missing; provide --revision or a valid model lock")
    return model_id, revision


def _fixture_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    shared_labels: Sequence[str] = DEFAULT_LABELS
    if isinstance(payload, Mapping):
        if "labels" in payload:
            labels = payload["labels"]
            if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
                raise TypeError("fixture labels must be a sequence")
            shared_labels = labels  # type: ignore[assignment]
        rows = payload.get("cases", payload.get("examples"))
    else:
        rows = payload
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("fixtures must be a list or an object containing cases/examples")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"fixture {index} must be an object")
        example_id = row.get("example_id", row.get("id"))
        text = row.get("text", row.get("full_text"))
        labels = row.get("labels", shared_labels)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"fixture {index} has no valid example_id/id")
        if not isinstance(text, str):
            raise ValueError(f"fixture {example_id!r} has no valid text/full_text")
        if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
            raise TypeError(f"fixture {example_id!r} labels must be a sequence")
        normalized.append({"example_id": example_id, "text": text, "labels": list(labels)})
    return normalized


def _load_reference_model(
    model_id: str,
    revision: str,
    *,
    cache_dir: Path | None,
    local_files_only: bool,
) -> object:
    try:
        from gliner import GLiNER  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "GLiNER is unavailable. Run this script with `.venv-reference/bin/python` "
            "and `PYTHONPATH=src`."
        ) from error

    return GLiNER.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        load_tokenizer=True,
        map_location="cpu",
        dtype="fp32",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_id, revision = _locked_model(args)
    fixtures = _fixture_rows(args.fixtures)
    selected = set(args.example_ids or ())
    if selected:
        fixtures = [row for row in fixtures if row["example_id"] in selected]
        missing = selected - {row["example_id"] for row in fixtures}
        if missing:
            raise ValueError(f"unknown example IDs: {sorted(missing)}")
    if not fixtures:
        raise ValueError("no fixtures selected")

    model = _load_reference_model(
        model_id,
        revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    run_id = f"public-{uuid.uuid4().hex}"
    traces = []
    for fixture in fixtures:
        trace = run_public_trace(
            model,  # type: ignore[arg-type]
            text=fixture["text"],
            labels=fixture["labels"],
            example_id=fixture["example_id"],
            run_id=run_id,
            words_per_chunk=args.words_per_chunk,
            threshold=args.threshold,
        )
        traces.append(trace)
        print(format_public_trace(trace))
        print()

    payload = {
        "run_id": run_id,
        "model_id": model_id,
        "model_revision_sha": revision,
        "words_per_chunk": args.words_per_chunk,
        "threshold": args.threshold,
        "traces": [trace.to_dict() for trace in traces],
    }
    # Always prove serializability, even when no output file was requested.
    json.dumps(payload, ensure_ascii=False)
    if args.output:
        _write_json(args.output, payload)


if __name__ == "__main__":
    main()
