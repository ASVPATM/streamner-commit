"""Generate or exactly reuse immutable PIIMB MLX traces for offline replay."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from streamner_commit.datasets.piimb_trace import load_piimb_trace
from streamner_commit.streaming.trace_pipeline import (
    PRESET_MANIFEST_PATHS,
    TraceModelProvenance,
    TracePipelineResult,
    manifest_path_for_preset,
    run_piimb_trace_pipeline,
    trace_inputs_from_piimb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_MANIFEST_PATHS),
        default="smoke",
        help="checked-in deterministic selection preset (default: smoke)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional sanitized manifest override; its declared preset must match --preset",
    )
    parser.add_argument("--split", choices=("dev", "test", "both"), default="both")
    parser.add_argument(
        "--chunk-words",
        nargs="+",
        type=int,
        choices=(1, 2, 4, 8),
        default=(1,),
        help="one or more distinct update-unit conditions (default: 1)",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("data/test_sentences.jsonl"),
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--right-context-width", type=int)
    return parser.parse_args(argv)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _execute(args: argparse.Namespace) -> TracePipelineResult:
    from streamner_commit.mlx.assets import (
        REFERENCE_MODEL_ID,
        REFERENCE_REVISION,
        load_asset_bundle,
    )
    from streamner_commit.mlx.precision import require_mlx_full_precision

    require_mlx_full_precision()
    manifest_path = (
        manifest_path_for_preset(PROJECT_ROOT, args.preset)
        if args.manifest is None
        else _repo_path(args.manifest)
    )
    reconstructed = load_piimb_trace(
        manifest_path,
        _repo_path(args.source_jsonl),
    )
    # Refuse bad preset/split/label/metadata inputs before loading 2.7 GB of weights.
    trace_inputs_from_piimb(
        reconstructed,
        preset=args.preset,
        split=args.split,
    )

    asset_root = (
        PROJECT_ROOT / "artifacts" / "reference" / REFERENCE_REVISION
        if args.asset_root is None
        else _repo_path(args.asset_root)
    )
    bundle = load_asset_bundle(
        asset_root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    from streamner_commit.backends.mlx_streaming import MLXStreamingBackend

    backend = MLXStreamingBackend.from_asset_bundle(
        bundle,
        context_limit=args.context_limit,
        right_context_width=args.right_context_width,
    )
    provenance = TraceModelProvenance.from_asset_bundle(bundle, backend)
    try:
        return run_piimb_trace_pipeline(
            backend,
            reconstructed,
            provenance,
            project_root=PROJECT_ROOT,
            preset=args.preset,
            split=args.split,
            chunk_words=tuple(args.chunk_words),
        )
    finally:
        backend.clear_sessions()


def _print_sanitized_summary(result: TracePipelineResult) -> None:
    for condition in result.conditions:
        print(
            f"run_id={condition.run.fingerprint.run_id} "
            f"chunk_words={condition.chunk_words} "
            f"examples={condition.example_count} "
            f"steps={condition.step_count} "
            f"span_updates={condition.span_update_count} "
            f"reused={int(condition.reused)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = _execute(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        # Do not echo licensed text or absolute local source paths from nested errors.
        print(
            f"trace generation did not complete ({type(error).__name__})",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    _print_sanitized_summary(result)


if __name__ == "__main__":
    main()
