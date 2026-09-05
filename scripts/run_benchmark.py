"""Evaluate frozen policies on verified held-out traces without model calls."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from streamner_commit.experiments.benchmark import run_frozen_benchmark
from streamner_commit.experiments.config import load_research_config
from streamner_commit.experiments.sweep import read_frozen_configs
from streamner_commit.experiments.traces import load_trace_conditions, trace_provenance
from streamner_commit.serialization import project_version

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--trace-root", type=Path, default=Path("results/traces"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--frozen", type=Path, default=Path("results/analysis/frozen_configs.json"))
    parser.add_argument("--preset")
    parser.add_argument("--chunk-words", nargs="+", type=int, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="allow nonclaim validation without a project commit and use 50 bootstrap replicates",
    )
    return parser.parse_args(argv)


def _path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def execute(args: argparse.Namespace) -> dict[str, Path]:
    config = load_research_config(_path(args.config))
    preset = args.preset or ("smoke" if args.pilot else config.held_out_manifest)
    chunks = (
        tuple(args.chunk_words)
        if args.chunk_words is not None
        else ((config.primary_chunk_words,) if args.pilot else config.chunk_words)
    )
    frozen = read_frozen_configs(
        _path(args.frozen),
        config,
        require_final_gate=not args.pilot,
    )
    if frozen.get("preset") != preset:
        raise ValueError("held-out preset differs from the frozen development preset")
    conditions = load_trace_conditions(
        _path(args.trace_root),
        config,
        preset=preset,
        split="test",
        chunk_words=chunks,
        required_project_commit=None if args.pilot else project_version(PROJECT_ROOT)[0],
    )
    provenance = trace_provenance(
        conditions,
        pilot=args.pilot,
        project_root=PROJECT_ROOT,
    )
    return dict(
        run_frozen_benchmark(
            config,
            conditions,
            provenance,
            frozen,
            _path(args.output_dir),
            pilot=args.pilot,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    try:
        outputs = execute(parse_args(argv))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"benchmark did not complete ({type(error).__name__})", file=sys.stderr)
        raise SystemExit(1) from None
    print(" ".join(f"{name}={path.name}" for name, path in sorted(outputs.items())))


if __name__ == "__main__":
    main()
