"""Build raw StreamingSpan traces for the committed synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.backends.reference_gliner import ReferenceGLiNERBackend
from streamner_commit.chunking import chunk_text_by_words
from streamner_commit.reference.observer import ObservedAppend, StreamingSpanObserver
from streamner_commit.streaming.replay import assert_span_states_close, replay_span_updates
from streamner_commit.types import SpanBoundary, SpanScoreUpdate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("data/fixtures/streaming_cases.json"))
    parser.add_argument("--model-lock", type=Path, default=Path("configs/model_lock.json"))
    parser.add_argument("--words-per-chunk", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def fixture_rows(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    rows = payload.get("cases")
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("fixture document must contain a cases list")
    result: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each fixture must be an object")
        example_id = row.get("id")
        text = row.get("text")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError("fixture id must be a nonblank string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"fixture {example_id!r} text must be nonblank")
        result.append((example_id, text))
    return tuple(result)


def ordered_labels(payload: Mapping[str, Any]) -> tuple[str, ...]:
    labels = payload.get("labels")
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("fixture document must contain an ordered labels list")
    result = tuple(str(label) for label in labels)
    if not result or not all(isinstance(label, str) and label.strip() for label in result):
        raise ValueError("fixture labels must be nonblank strings")
    return result


def serialize_final_state(
    state: Mapping[SpanBoundary, Sequence[float]],
) -> list[dict[str, Any]]:
    return [
        {
            "start_word": boundary.start_word,
            "end_word": boundary.end_word,
            "logits": list(logits),
        }
        for boundary, logits in sorted(state.items())
    ]


def main() -> None:
    args = parse_args()
    fixtures = read_object(args.fixtures)
    lock = read_object(args.model_lock)
    model_id = lock.get("model_id")
    revision = lock.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("model lock must contain string model_id and revision values")

    labels = ordered_labels(fixtures)
    backend = ReferenceGLiNERBackend.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    run_id = f"fixture-raw-{uuid.uuid4().hex}"
    traces: list[dict[str, Any]] = []
    validated_public_scores = 0
    total_updates = 0

    for example_id, text in fixture_rows(fixtures):
        updates: list[SpanScoreUpdate] = []
        observed_steps: list[ObservedAppend] = []
        session_id = f"{run_id}:{example_id}"
        with StreamingSpanObserver(
            backend.raw_model,
            run_id=run_id,
            example_id=example_id,
            session_id=session_id,
            labels=labels,
            threshold=args.threshold,
        ) as observer:
            for step, chunk in enumerate(
                chunk_text_by_words(text, args.words_per_chunk),
                start=1,
            ):
                if not chunk.strip():
                    continue
                observed = observer.append(chunk, step=step)
                updates.extend(observed.span_updates)
                observed_steps.append(observed)
                validated_public_scores += observed.validated_public_score_count

        if not observed_steps:
            raise AssertionError(f"fixture {example_id!r} produced no observed steps")
        final_state = observed_steps[-1].merged_span_logits
        assert_span_states_close(replay_span_updates(updates), final_state)
        total_updates += len(updates)
        traces.append(
            {
                "example_id": example_id,
                "full_text": text,
                "labels": list(labels),
                "steps": [item.snapshot.to_dict() for item in observed_steps],
                "span_updates": [update.to_dict() for update in updates],
                "final_span_state": serialize_final_state(final_state),
                "validated_public_score_count": sum(
                    item.validated_public_score_count for item in observed_steps
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "backend": "gliner-reference",
        "model_id": model_id,
        "model_revision_sha": revision,
        "gliner_version": "0.2.28",
        "dtype": "float32",
        "device": "cpu",
        "words_per_chunk": args.words_per_chunk,
        "threshold": args.threshold,
        "labels": list(labels),
        "trace_count": len(traces),
        "span_update_count": total_updates,
        "validated_public_score_count": validated_public_scores,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(traces)} traces, {total_updates} span updates, "
        f"{validated_public_scores} public-score checks to {args.output}"
    )


if __name__ == "__main__":
    main()
