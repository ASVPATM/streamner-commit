import sys
from importlib.metadata import version

import mlx.core as mx
import mlx_lm
import transformers


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_expected_dependency_versions() -> None:
    assert version("mlx") == "0.32.2"
    assert mlx_lm.__version__ == "0.31.3"
    assert transformers.__version__ == "5.12.1"


def test_mlx_compute() -> None:
    result = mx.array([1, 2, 3]) * 2
    mx.eval(result)
    assert result.tolist() == [2, 4, 6]
