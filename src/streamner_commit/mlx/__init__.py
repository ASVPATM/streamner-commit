"""Torch-free MLX runtime helpers.

Importing this package must remain safe in the main environment: neither PyTorch nor
GLiNER is imported by the asset loader.
"""

from streamner_commit.mlx.precision import (
    MLX_FULL_FP32_VALUE,
    MLX_TF32_ENVIRONMENT_VARIABLE,
    MLXPrecisionError,
    configure_mlx_full_precision,
    require_mlx_full_precision,
)

# MLX 0.32.2 can route FP32 matmul-family operations through reduced-precision
# hardware by default. Establish the reference-compatible process invariant before
# importing any project module that may import mlx.core.
configure_mlx_full_precision()

from streamner_commit.mlx.assets import (  # noqa: E402 - precision must be configured first
    REFERENCE_MODEL_ID,
    REFERENCE_PARAMETER_COUNT,
    REFERENCE_REVISION,
    REFERENCE_TENSOR_COUNT,
    AssetBundle,
    AssetValidationError,
    TensorRecord,
    load_asset_bundle,
)

__all__ = [
    "MLX_FULL_FP32_VALUE",
    "MLX_TF32_ENVIRONMENT_VARIABLE",
    "MLXPrecisionError",
    "REFERENCE_MODEL_ID",
    "REFERENCE_PARAMETER_COUNT",
    "REFERENCE_REVISION",
    "REFERENCE_TENSOR_COUNT",
    "AssetBundle",
    "AssetValidationError",
    "TensorRecord",
    "configure_mlx_full_precision",
    "load_asset_bundle",
    "require_mlx_full_precision",
]
