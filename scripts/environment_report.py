import importlib.util
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path


def package_version(distribution: str) -> str:
    return version(distribution)


def git_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable (no commit)"


print("macOS:", platform.mac_ver()[0])
print("Architecture:", platform.machine())
print("Python:", platform.python_version())
print("Git commit:", git_commit())

if importlib.util.find_spec("mlx") is not None:
    from streamner_commit.mlx.precision import require_mlx_full_precision

    require_mlx_full_precision()
    import mlx.core as mx

    value = mx.array([1, 2, 3]) * 2
    mx.eval(value)
    print("MLX:", package_version("mlx"))
    print("MLX-LM:", package_version("mlx-lm"))
    print("Transformers:", package_version("transformers"))
    print("MLX_ENABLE_TF32:", os.environ["MLX_ENABLE_TF32"])
    print("MLX compute:", value.tolist() == [2, 4, 6])

if importlib.util.find_spec("torch") is not None:
    import torch  # type: ignore[import-not-found]

    print("PyTorch:", torch.__version__)
    print("GLiNER:", package_version("gliner"))
    print("Transformers:", package_version("transformers"))
    print("PyTorch MPS available:", torch.backends.mps.is_available())
