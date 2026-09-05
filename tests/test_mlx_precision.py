from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from streamner_commit.mlx.precision import (
    MLXPrecisionError,
    configure_mlx_full_precision,
    require_mlx_full_precision,
)

ROOT = Path(__file__).resolve().parents[1]


def _isolated_import(*, configured: str | None) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if configured is None:
        environment.pop("MLX_ENABLE_TF32", None)
    else:
        environment["MLX_ENABLE_TF32"] = configured
    source = (
        "import os; import streamner_commit.mlx; "
        "print(os.environ['MLX_ENABLE_TF32'])"
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_import_defaults_mlx_to_full_fp32_in_isolated_process() -> None:
    result = _isolated_import(configured=None)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0"


def test_package_import_rejects_incompatible_precision_in_isolated_process() -> None:
    result = _isolated_import(configured="1")

    assert result.returncode != 0
    assert "MLX_ENABLE_TF32 must be exactly '0'" in result.stderr
    assert "before starting Python" in result.stderr


def test_configure_preserves_explicit_safe_value() -> None:
    environment = {"MLX_ENABLE_TF32": "0", "UNRELATED": "kept"}

    configure_mlx_full_precision(environ=environment)

    assert environment == {"MLX_ENABLE_TF32": "0", "UNRELATED": "kept"}


@pytest.mark.parametrize("configured", ["1", "true", "", "00"])
def test_configure_rejects_every_noncanonical_override(configured: str) -> None:
    with pytest.raises(MLXPrecisionError, match="must be exactly '0'"):
        configure_mlx_full_precision(environ={"MLX_ENABLE_TF32": configured})


def test_parity_preflight_detects_environment_mutation() -> None:
    with pytest.raises(MLXPrecisionError, match="parity-critical.*current value is '1'"):
        require_mlx_full_precision(environ={"MLX_ENABLE_TF32": "1"})


def test_parity_preflight_rejects_unset_environment() -> None:
    with pytest.raises(MLXPrecisionError, match="current value is unset"):
        require_mlx_full_precision(environ={})
