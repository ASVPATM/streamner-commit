"""Render the three required result tables from the held-out aggregate artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from streamner_commit.publication import PublicationError, generate_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("results/analysis/test_benchmark.parquet"),
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("results/analysis/benchmark_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--selection-mode",
        choices=("matched_quality", "matched_latency"),
        default="matched_quality",
    )
    parser.add_argument("--primary-chunk", type=int, default=1)
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        paths = generate_tables(
            benchmark_path=_project_path(args.benchmark),
            benchmark_manifest_path=_project_path(args.benchmark_manifest),
            output_dir=_project_path(args.output_dir),
            selection_mode=args.selection_mode,
            primary_chunk=args.primary_chunk,
        )
    except PublicationError as error:
        print(f"table generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    for path in paths:
        print(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


if __name__ == "__main__":
    main()
