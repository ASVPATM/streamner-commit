"""Run native MLX streaming against an exported deterministic reference suite."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.precision import require_mlx_full_precision
from streamner_commit.mlx.streaming_validation import (
    run_streaming_validation,
    write_streaming_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("artifacts/reference") / REFERENCE_REVISION
    parser.add_argument("--asset-root", type=Path, default=default_root)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--chunk-units",
        type=int,
        nargs="+",
        choices=(1, 2, 4),
        default=[1],
        help="Oracle schedules to validate (default: 1).",
    )
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--right-context-width", type=int)
    return parser.parse_args()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return dict(value)


def main() -> None:
    require_mlx_full_precision()
    args = parse_args()
    asset_root = args.asset_root.resolve()
    oracle_path = (args.oracle or asset_root / "streaming_parity" / "suite.json").resolve()
    output_path = (args.output or asset_root / "mlx_streaming_parity_report.json").resolve()
    oracle = _json_object(oracle_path, name="streaming parity oracle")
    if oracle.get("model_id") != REFERENCE_MODEL_ID:
        raise ValueError("streaming oracle model ID does not match the locked reference")
    if oracle.get("model_revision_sha") != REFERENCE_REVISION:
        raise ValueError("streaming oracle revision does not match the locked reference")

    bundle = load_asset_bundle(
        asset_root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    # Delay the model import until after the cheap identity and asset checks.
    from streamner_commit.backends.mlx_streaming import MLXStreamingBackend

    backend = MLXStreamingBackend.from_asset_bundle(
        bundle,
        context_limit=args.context_limit,
        right_context_width=args.right_context_width,
    )
    report = run_streaming_validation(
        backend,
        oracle,
        chunk_units=args.chunk_units,
    )
    artifact = write_streaming_validation_report(report, output_path)
    totals = report["totals"]
    outcome = "PASS" if report["pass"] else "FAIL"
    print(
        f"{outcome}: {totals['case_count']} cases, {totals['step_count']} steps; "
        f"wrote {artifact['path']} ({artifact['sha256']})"
    )
    if not report["pass"]:
        raise SystemExit("MLX streaming parity gate failed")


if __name__ == "__main__":
    main()
