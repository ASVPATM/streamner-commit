"""Process-wide numerical-precision invariant for the native MLX runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

MLX_TF32_ENVIRONMENT_VARIABLE = "MLX_ENABLE_TF32"
MLX_FULL_FP32_VALUE = "0"


class MLXPrecisionError(RuntimeError):
    """The process cannot provide the full-FP32 semantics required for parity."""


def _incompatible_message(value: str) -> str:
    return (
        f"{MLX_TF32_ENVIRONMENT_VARIABLE} must be exactly "
        f"{MLX_FULL_FP32_VALUE!r} for reference-compatible full-FP32 MLX inference; "
        f"got {value!r}. Set {MLX_TF32_ENVIRONMENT_VARIABLE}=0 before starting Python "
        "because MLX caches this setting on first matmul-family use."
    )


def configure_mlx_full_precision(
    *,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Default MLX to full FP32, rejecting an explicit incompatible override.

    This function intentionally has no MLX import. The package initializer calls it
    before any project MLX module can import :mod:`mlx.core`.
    """

    target = os.environ if environ is None else environ
    configured = target.get(MLX_TF32_ENVIRONMENT_VARIABLE)
    if configured is None:
        target[MLX_TF32_ENVIRONMENT_VARIABLE] = MLX_FULL_FP32_VALUE
    elif configured != MLX_FULL_FP32_VALUE:
        raise MLXPrecisionError(_incompatible_message(configured))


def require_mlx_full_precision(
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail clearly unless the current process advertises full-FP32 MLX semantics."""

    target = os.environ if environ is None else environ
    configured = target.get(MLX_TF32_ENVIRONMENT_VARIABLE)
    if configured != MLX_FULL_FP32_VALUE:
        rendered = "unset" if configured is None else repr(configured)
        raise MLXPrecisionError(
            f"parity-critical MLX execution requires "
            f"{MLX_TF32_ENVIRONMENT_VARIABLE}=0; current value is {rendered}. "
            f"Set {MLX_TF32_ENVIRONMENT_VARIABLE}=0 before starting Python because "
            "MLX caches this setting on first matmul-family use."
        )


__all__ = [
    "MLX_FULL_FP32_VALUE",
    "MLX_TF32_ENVIRONMENT_VARIABLE",
    "MLXPrecisionError",
    "configure_mlx_full_precision",
    "require_mlx_full_precision",
]
