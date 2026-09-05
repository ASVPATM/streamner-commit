"""Strict, selective loading of validated reference tensors into MLX modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten
from safetensors import safe_open

from streamner_commit.mlx.assets import AssetBundle, AssetValidationError

if TYPE_CHECKING:
    import mlx.nn as nn


_COMPONENT_ROOTS = {
    "qwen": "qwen.",
    "label_encoder": "label_encoder.",
    "marker_v2": "marker.",
    "prompt_projection": "prompt_projection.",
}


class MLXWeightError(ValueError):
    """A validated export cannot be assigned exactly to the requested module."""


def module_parameter_names(module: nn.Module) -> frozenset[str]:
    """Return the flattened parameter-name inventory for one MLX module."""

    flattened = tree_flatten(module.parameters(), destination={})
    if not isinstance(flattened, dict):  # pragma: no cover - fixed by destination type
        raise MLXWeightError("MLX returned an unexpected parameter-tree representation")
    return frozenset(flattened)


def load_component_arrays(
    bundle: AssetBundle,
    component: str,
    *,
    strip_component_root: bool = True,
) -> dict[str, mx.array]:
    """Read only one component from a validated safetensors export.

    The :class:`AssetBundle` construction already checks the file hash, header, and
    exhaustive tensor manifest. Selective reads avoid materializing the 2.7 GB
    checkpoint twice merely to initialize one submodule.
    """

    if component not in _COMPONENT_ROOTS:
        choices = ", ".join(sorted(_COMPONENT_ROOTS))
        raise MLXWeightError(f"unknown component {component!r}; expected one of: {choices}")
    records = tuple(record for record in bundle.tensors if record.component == component)
    if not records:
        raise MLXWeightError(f"validated export has no tensors for component {component!r}")

    root = _COMPONENT_ROOTS[component]
    arrays: dict[str, mx.array] = {}
    try:
        with safe_open(bundle.weights_path, framework="numpy") as handle:
            available = frozenset(handle.keys())
            for record in records:
                if record.reference_name not in available:
                    raise MLXWeightError(f"safe checkpoint is missing {record.reference_name!r}")
                destination = record.mlx_key
                if strip_component_root:
                    if not destination.startswith(root):
                        raise MLXWeightError(
                            f"mapped key {destination!r} does not use component root {root!r}"
                        )
                    destination = destination.removeprefix(root)
                if destination in arrays:
                    raise MLXWeightError(f"duplicate MLX destination {destination!r}")
                array = handle.get_tensor(record.reference_name)
                if tuple(array.shape) != record.shape:
                    raise MLXWeightError(
                        f"shape mismatch for {record.reference_name!r}: "
                        f"expected {record.shape}, got {tuple(array.shape)}"
                    )
                arrays[destination] = mx.array(array)
            # MLX construction is lazy; materialize while safetensors' NumPy
            # buffers and file mapping are still alive.
            mx.eval(list(arrays.values()))
    except MLXWeightError:
        raise
    except Exception as exc:
        raise AssetValidationError(f"cannot selectively read safe weights: {exc}") from exc
    return arrays


def strict_load_weights(
    module: nn.Module,
    weights: Mapping[str, mx.array] | Iterable[tuple[str, mx.array]],
) -> None:
    """Load a complete module inventory and reject missing or extra keys."""

    supplied = dict(weights)
    expected = module_parameter_names(module)
    actual = frozenset(supplied)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise MLXWeightError("weight inventory mismatch: " + "; ".join(details))
    try:
        module.load_weights(list(supplied.items()), strict=True)
        mx.eval(module.parameters())
    except Exception as exc:
        raise MLXWeightError(f"MLX rejected the weight inventory: {exc}") from exc


def load_component_into(module: nn.Module, bundle: AssetBundle, component: str) -> None:
    """Selectively read and strictly assign one canonical checkpoint component."""

    strict_load_weights(module, load_component_arrays(bundle, component))


__all__ = [
    "MLXWeightError",
    "load_component_arrays",
    "load_component_into",
    "module_parameter_names",
    "strict_load_weights",
]
