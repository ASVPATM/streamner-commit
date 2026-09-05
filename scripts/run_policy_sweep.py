"""Replay configured policies over verified development traces without model calls."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from streamner_commit.experiments.config import load_research_config
from streamner_commit.experiments.policies import pilot_policy_specs
from streamner_commit.experiments.sweep import (
    run_checkpointed_development_sweep,
    run_development_sweep,
)
from streamner_commit.experiments.traces import load_trace_conditions, trace_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--trace-root", type=Path, default=Path("results/traces"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/analysis/.sweep_checkpoints"),
        help="ignored local directory for resumable aggregate checkpoints",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=2,
        help="bounded chunk worker processes (default: 2; use 1 for minimum RAM)",
    )
    parser.add_argument(
        "--progress-seconds",
        type=_positive_float,
        default=15.0,
        help="seconds between aggregate progress/ETA reports",
    )
    parser.add_argument("--preset")
    parser.add_argument("--chunk-words", nargs="+", type=int, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run one config per policy on primary-chunk smoke traces; outputs are nonclaim",
    )
    return parser.parse_args(argv)


def _path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def execute(args: argparse.Namespace) -> dict[str, Path]:
    config = load_research_config(_path(args.config))
    preset = args.preset or ("smoke" if args.pilot else config.development_manifest)
    chunks = (
        tuple(args.chunk_words)
        if args.chunk_words is not None
        else ((config.primary_chunk_words,) if args.pilot else config.chunk_words)
    )
    if not args.pilot:
        return dict(
            run_checkpointed_development_sweep(
                config,
                _path(args.trace_root),
                PROJECT_ROOT,
                _path(args.output_dir),
                preset=preset,
                chunk_words=chunks,
                checkpoint_dir=_path(args.checkpoint_dir),
                workers=args.workers,
                progress_seconds=args.progress_seconds,
            )
        )
    conditions = load_trace_conditions(
        _path(args.trace_root),
        config,
        preset=preset,
        split="dev",
        chunk_words=chunks,
    )
    provenance = trace_provenance(conditions, pilot=True, project_root=PROJECT_ROOT)
    return dict(
        run_development_sweep(
            config,
            conditions,
            provenance,
            _path(args.output_dir),
            preset=preset,
            specs=pilot_policy_specs(config),
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    try:
        outputs = execute(parse_args(argv))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(
            f"policy sweep did not complete ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(" ".join(f"{name}={path.name}" for name, path in sorted(outputs.items())))


if __name__ == "__main__":
    main()
