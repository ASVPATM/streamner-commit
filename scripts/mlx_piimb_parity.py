"""Run the native MLX backend against the pinned 100-case PIIMB smoke oracle."""

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
from streamner_commit.mlx.piimb_validation import (
    run_piimb_validation,
    write_piimb_validation_report,
)
from streamner_commit.mlx.precision import require_mlx_full_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("artifacts/reference") / REFERENCE_REVISION
    parser.add_argument("--asset-root", type=Path, default=default_root)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/parity/piimb_smoke_report.json"),
    )
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--right-context-width", type=int)
    return parser.parse_args()


def _json_object(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("PIIMB oracle must contain a JSON object")
    return dict(value)


def main() -> None:
    require_mlx_full_precision()
    args = parse_args()
    asset_root = args.asset_root.resolve()
    oracle_path = (args.oracle or asset_root / "piimb_parity" / "smoke.json").resolve()
    oracle = _json_object(oracle_path)
    if oracle.get("model_id") != REFERENCE_MODEL_ID:
        raise ValueError("PIIMB oracle model ID does not match the locked reference")
    if oracle.get("model_revision_sha") != REFERENCE_REVISION:
        raise ValueError("PIIMB oracle revision does not match the locked reference")

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
    report = run_piimb_validation(backend, oracle)
    artifact = write_piimb_validation_report(report, args.output.resolve())
    outcome = "PASS" if report["pass"] else "FAIL"
    print(
        f"{outcome}: {report['counts']['cases']} cases, "
        f"{report['counts']['candidate_vectors']} candidates; "
        f"wrote {artifact['path']} ({artifact['sha256']})"
    )
    if not report["pass"]:
        raise SystemExit("MLX PIIMB parity smoke gate failed")


if __name__ == "__main__":
    main()
