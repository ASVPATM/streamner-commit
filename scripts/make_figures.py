"""Render the four required figures from sanitized experiment artifacts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from streamner_commit.publication import PublicationError, generate_figures

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, default=Path("results/analysis/dev_pareto.parquet"))
    parser.add_argument(
        "--revision-horizons",
        type=Path,
        default=Path("results/analysis/revision_horizons.parquet"),
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("results/analysis/benchmark_manifest.json"),
    )
    parser.add_argument(
        "--streaming-parity",
        type=Path,
        default=Path("results/parity/parity_report.json"),
    )
    parser.add_argument(
        "--piimb-parity",
        type=Path,
        default=Path("results/parity/piimb_smoke_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--primary-chunk", type=int, default=1)
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        paths = generate_figures(
            pareto_path=_project_path(args.pareto),
            revision_horizons_path=_project_path(args.revision_horizons),
            benchmark_manifest_path=_project_path(args.benchmark_manifest),
            streaming_parity_path=_project_path(args.streaming_parity),
            piimb_parity_path=_project_path(args.piimb_parity),
            output_dir=_project_path(args.output_dir),
            primary_chunk=args.primary_chunk,
        )
    except PublicationError as error:
        print(f"figure generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    for path in paths:
        print(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path)


if __name__ == "__main__":
    main()
