"""Run a tiny, local-only MLX cold/warm/offline performance characterization."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from streamner_commit.chunking import chunk_text_by_words, word_char_spans
from streamner_commit.performance import (
    PerformanceConfiguration,
    PerformanceReport,
    RuntimeMetadata,
    benchmark_performance,
)
from streamner_commit.types import SnapshotStep, SpanScoreUpdate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ("person", "organization")


def _nonnegative(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return result


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--warmups", type=_nonnegative, default=1)
    parser.add_argument("--repetitions", type=_positive, default=1)
    parser.add_argument("--chunk-words", type=_positive, default=1)
    parser.add_argument(
        "--text-words",
        type=_positive,
        default=4,
        help="synthetic whitespace-delimited update units (default: 4)",
    )
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--context-limit", type=_positive)
    parser.add_argument("--right-context-width", type=_nonnegative)
    return parser.parse_args(argv)


def _synthetic_text(word_count: int) -> str:
    return " ".join("sample" for _ in range(word_count))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _synthetic_trace(
    text: str,
    labels: tuple[str, ...],
    chunk_words: int,
) -> tuple[tuple[SnapshotStep, ...], tuple[SpanScoreUpdate, ...]]:
    """Build a content-safe CPU trace solely for the replay timing category."""

    chunks = tuple(chunk_text_by_words(text, chunk_words))
    offsets = tuple(word_char_spans(text))
    snapshots: list[SnapshotStep] = []
    updates: list[SpanScoreUpdate] = []
    visible_text = ""
    visible_words = 0
    previous_first_probability: float | None = None
    for step, chunk in enumerate(chunks, start=1):
        visible_text += chunk
        next_visible_words = min(step * chunk_words, len(offsets))
        snapshot = SnapshotStep(
            run_id="performance-replay",
            example_id="synthetic",
            step=step,
            chunk=chunk,
            accumulated_text=visible_text,
            visible_char_count=len(visible_text),
            visible_word_count=next_visible_words,
            elapsed_ms=0.0,
            public_entities=(),
        )
        snapshots.append(snapshot)

        scored_words = list(range(visible_words, next_visible_words))
        if step > 1:
            scored_words.insert(0, 0)
        for word_index in scored_words:
            positive_logit = 1.0 + (0.01 * step if word_index == 0 else 0.0)
            logits = (positive_logit,) + tuple(-1.0 for _ in labels[1:])
            probabilities = tuple(_sigmoid(value) for value in logits)
            top_index = max(range(len(labels)), key=probabilities.__getitem__)
            second = max(
                probabilities[:top_index] + probabilities[top_index + 1 :],
                default=0.0,
            )
            start_char, end_char = offsets[word_index]
            is_rescore = word_index == 0 and step > 1
            prior = previous_first_probability if is_rescore else None
            updates.append(
                SpanScoreUpdate(
                    run_id=snapshot.run_id,
                    example_id=snapshot.example_id,
                    step=step,
                    chunk=chunk,
                    visible_char_count=snapshot.visible_char_count,
                    visible_word_count=snapshot.visible_word_count,
                    start_word=word_index,
                    end_word=word_index,
                    start_char=start_char,
                    end_char=end_char,
                    span_text=text[start_char:end_char],
                    logits=logits,
                    probs=probabilities,
                    top_label_index=top_index,
                    top_label=labels[top_index],
                    top_probability=probabilities[top_index],
                    second_probability=second,
                    label_margin=probabilities[top_index] - second,
                    previous_top_probability=prior,
                    top_probability_delta=(
                        None if prior is None else probabilities[top_index] - prior
                    ),
                    update_kind="rescore" if is_rescore else "new",
                    tail_distance_words=(next_visible_words - 1) - word_index,
                )
            )
            if word_index == 0:
                previous_first_probability = probabilities[top_index]
        visible_words = next_visible_words
    return tuple(snapshots), tuple(updates)


def _runtime_metadata(bundle: Any, mx: Any) -> RuntimeMetadata:
    dtypes = sorted({record.dtype for record in bundle.tensors})
    dtype = ",".join(dtypes)
    device = str(mx.default_device())
    return RuntimeMetadata(
        model_revision=bundle.revision,
        dtype=dtype,
        model_device=f"MLX {device}",
        replay_device="CPU (Python host)",
        precision_mode="full-fp32; MLX_ENABLE_TF32=0",
        machine=platform.machine(),
        platform=f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        python_version=platform.python_version(),
        runtime_versions={
            "mlx": version("mlx"),
            "mlx-lm": version("mlx-lm"),
            "streamner-commit": version("streamner-commit"),
            "transformers": version("transformers"),
        },
    )


def _synchronize_prediction(prediction: Any, mx: Any) -> None:
    forward = prediction.forward
    mx.eval(
        forward.qwen_hidden_states,
        forward.labels.prompt.hidden_states,
        forward.labels.contextualized_prompt,
        forward.labels.label_representations,
        forward.pooled_word_states,
        forward.pooled_word_mask,
        forward.projected_label_states,
        forward.span_states,
        forward.logits,
    )


def _execute(args: argparse.Namespace) -> PerformanceReport:
    from streamner_commit.mlx.assets import (
        REFERENCE_MODEL_ID,
        REFERENCE_REVISION,
        load_asset_bundle,
    )
    from streamner_commit.mlx.precision import require_mlx_full_precision

    require_mlx_full_precision()
    text = _synthetic_text(args.text_words)
    labels = tuple(args.labels)
    configuration = PerformanceConfiguration.from_text(
        text,
        labels,
        warmup_count=args.warmups,
        repetition_count=args.repetitions,
        chunk_words=args.chunk_words,
    )
    chunks = tuple(chunk_text_by_words(text, args.chunk_words))
    snapshots, updates = _synthetic_trace(text, labels, args.chunk_words)

    asset_root = (
        PROJECT_ROOT / "artifacts" / "reference" / REFERENCE_REVISION
        if args.asset_root is None
        else (args.asset_root if args.asset_root.is_absolute() else PROJECT_ROOT / args.asset_root)
    )
    if not asset_root.is_dir():
        raise FileNotFoundError(
            "validated local model assets are required; no download was attempted"
        )
    bundle = load_asset_bundle(
        asset_root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    from streamner_commit.backends.mlx_streaming import MLXStreamingBackend
    from streamner_commit.policies.simulator import simulate_commitments
    from streamner_commit.policies.threshold import FixedThreshold

    backend = MLXStreamingBackend.from_asset_bundle(
        bundle,
        context_limit=args.context_limit,
        right_context_width=args.right_context_width,
    )
    import mlx.core as mx

    from streamner_commit.streaming.tracker import build_streaming_observations

    observations = build_streaming_observations(snapshots, updates, labels)

    def cold_full() -> Any:
        return backend.model.predict(text, labels)

    def synchronize_cold(prediction: Any) -> None:
        _synchronize_prediction(prediction, mx)

    def warm_append_run() -> tuple[float, ...]:
        session = backend.start_session(labels)
        try:
            first = session.append(chunks[0])
            if first.is_noop:
                raise RuntimeError("first benchmark append unexpectedly became a no-op")
            elapsed: list[float] = []
            for chunk in chunks[1:]:
                result = session.append(chunk)
                if result.is_noop:
                    raise RuntimeError("warm benchmark append unexpectedly became a no-op")
                # The backend stops this timer only after evaluating new hidden states,
                # cached state, candidate logits, and host score replacements.
                elapsed.append(result.elapsed_ms)
            return tuple(elapsed)
        finally:
            session.clear()

    def policy_replay() -> object:
        return simulate_commitments(FixedThreshold(0.5), observations, labels)

    try:
        return benchmark_performance(
            configuration,
            _runtime_metadata(bundle, mx),
            cold_full_operation=cold_full,
            synchronize_cold=synchronize_cold,
            warm_append_run=warm_append_run,
            policy_replay_operation=policy_replay,
        )
    finally:
        backend.clear_sessions()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = _execute(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"performance benchmark did not complete ({error})", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
