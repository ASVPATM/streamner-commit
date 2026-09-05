"""Strict MLX-LM Qwen3 adapter for the locked StreamingSpan checkpoint.

The reference checkpoint owns the Qwen weights.  This module only supplies the
installed MLX-LM implementation of the *bare* transformer; it deliberately never
constructs a causal-language-model head and never transposes a checkpoint tensor.

MLX imports are deferred until model construction.  Besides making the numerical
diagnostic helper usable without a Metal device, this keeps importing the surrounding
asset and preprocessing modules cheap and side-effect free.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

import numpy as np
from safetensors import safe_open

from streamner_commit.conversion.weight_map import WeightMappingError, map_reference_key
from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    AssetBundle,
)
from streamner_commit.mlx.precision import require_mlx_full_precision

MLX_LM_VERSION = "0.31.3"
QWEN_COMPONENT = "qwen"
QWEN_MLX_PREFIX = "qwen."
QWEN_TENSOR_COUNT = 310


class QwenAdapterError(ValueError):
    """The bundle, runtime, input, or cache violates the locked Qwen contract."""


@dataclass(frozen=True, slots=True)
class QwenConfiguration:
    """The architecture fields consumed by MLX-LM's Qwen3 implementation."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> Self:
        """Read and validate the exact decoder architecture in exported config JSON."""

        if not isinstance(model_config, Mapping):
            raise QwenAdapterError("model config must be a mapping")
        decoder = model_config.get("decoder_config")
        if not isinstance(decoder, Mapping):
            raise QwenAdapterError("model config is missing decoder_config")
        rope_parameters = decoder.get("rope_parameters")
        if not isinstance(rope_parameters, Mapping):
            raise QwenAdapterError(
                "decoder_config.rope_parameters must contain the locked RoPE settings"
            )

        def require_int(name: str) -> int:
            value = decoder.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise QwenAdapterError(f"decoder_config.{name} must be an integer")
            return value

        def require_float(name: str) -> float:
            value = decoder.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise QwenAdapterError(f"decoder_config.{name} must be numeric")
            result = float(value)
            if not np.isfinite(result):
                raise QwenAdapterError(f"decoder_config.{name} must be finite")
            return result

        rope_theta_value = rope_parameters.get("rope_theta")
        if isinstance(rope_theta_value, bool) or not isinstance(rope_theta_value, (int, float)):
            raise QwenAdapterError("decoder_config.rope_parameters.rope_theta must be numeric")
        rope_theta = float(rope_theta_value)
        if not np.isfinite(rope_theta):
            raise QwenAdapterError("decoder_config.rope_parameters.rope_theta must be finite")

        model_type = decoder.get("model_type")
        tie_word_embeddings = decoder.get("tie_word_embeddings")
        configuration = cls(
            model_type=model_type if isinstance(model_type, str) else "",
            hidden_size=require_int("hidden_size"),
            num_hidden_layers=require_int("num_hidden_layers"),
            intermediate_size=require_int("intermediate_size"),
            num_attention_heads=require_int("num_attention_heads"),
            rms_norm_eps=require_float("rms_norm_eps"),
            vocab_size=require_int("vocab_size"),
            num_key_value_heads=require_int("num_key_value_heads"),
            max_position_embeddings=require_int("max_position_embeddings"),
            rope_theta=rope_theta,
            head_dim=require_int("head_dim"),
            tie_word_embeddings=(
                tie_word_embeddings if isinstance(tie_word_embeddings, bool) else False
            ),
        )
        configuration._validate_locked_values(decoder, rope_parameters)
        return configuration

    def _validate_locked_values(
        self,
        decoder: Mapping[str, Any],
        rope_parameters: Mapping[str, Any],
    ) -> None:
        expected: dict[str, Any] = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151_671,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40_960,
            "rope_theta": 1_000_000.0,
            "head_dim": 128,
            "tie_word_embeddings": True,
        }
        for field_name, expected_value in expected.items():
            actual_value = getattr(self, field_name)
            if actual_value != expected_value:
                raise QwenAdapterError(
                    f"locked Qwen {field_name} must be {expected_value!r}, got {actual_value!r}"
                )

        # These fields do not enter ModelArgs directly, but a changed value would mean
        # the installed Qwen3 implementation is no longer the architecture being loaded.
        auxiliary_expected = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
        }
        for name, expected_value in auxiliary_expected.items():
            if decoder.get(name) != expected_value:
                raise QwenAdapterError(
                    f"locked decoder_config.{name} must be {expected_value!r}, "
                    f"got {decoder.get(name)!r}"
                )
        if rope_parameters.get("rope_type") != "default":
            raise QwenAdapterError(
                "locked decoder_config.rope_parameters.rope_type must be 'default'"
            )
        if decoder.get("rope_scaling") not in (None, {}):
            raise QwenAdapterError("the locked Qwen checkpoint does not use RoPE scaling")


@dataclass(frozen=True, slots=True)
class QwenWeightSpec:
    """One direct reference-to-bare-Qwen parameter assignment."""

    reference_name: str
    bare_key: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class NumericalDiagnostics:
    """Backend-neutral numerical comparison over two activation arrays."""

    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    shape_equal: bool
    max_absolute_error: float | None
    mean_absolute_error: float | None
    cosine_similarity: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostics record."""

        return {
            "reference_shape": list(self.reference_shape),
            "candidate_shape": list(self.candidate_shape),
            "shape_equal": self.shape_equal,
            "max_absolute_error": self.max_absolute_error,
            "mean_absolute_error": self.mean_absolute_error,
            "cosine_similarity": self.cosine_similarity,
        }


def numerical_diagnostics(
    reference: np.ndarray[Any, Any] | Sequence[Any],
    candidate: np.ndarray[Any, Any] | Sequence[Any],
) -> NumericalDiagnostics:
    """Compute shape, absolute-error, and flattened-cosine diagnostics.

    This helper intentionally has no MLX dependency.  Shape mismatch is reported rather
    than broadcast, and its numerical fields are ``None`` because an elementwise
    comparison would not be meaningful.
    """

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.dtype.kind not in "biuf" or candidate_array.dtype.kind not in "biuf":
        raise QwenAdapterError("numerical diagnostics require real numeric arrays")
    if reference_array.size == 0 or candidate_array.size == 0:
        raise QwenAdapterError("numerical diagnostics require non-empty arrays")
    if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
        raise QwenAdapterError("numerical diagnostics require finite arrays")

    reference_shape = tuple(int(value) for value in reference_array.shape)
    candidate_shape = tuple(int(value) for value in candidate_array.shape)
    if reference_shape != candidate_shape:
        return NumericalDiagnostics(
            reference_shape=reference_shape,
            candidate_shape=candidate_shape,
            shape_equal=False,
            max_absolute_error=None,
            mean_absolute_error=None,
            cosine_similarity=None,
        )

    reference_flat = reference_array.astype(np.float64, copy=False).reshape(-1)
    candidate_flat = candidate_array.astype(np.float64, copy=False).reshape(-1)
    absolute_error = np.abs(reference_flat - candidate_flat)
    reference_norm = float(np.linalg.norm(reference_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 1.0 if reference_norm == candidate_norm else 0.0
    else:
        cosine = float(np.dot(reference_flat, candidate_flat) / (reference_norm * candidate_norm))
        # Floating-point reduction can put an otherwise valid cosine a few ulps outside
        # its mathematical range.
        cosine = min(1.0, max(-1.0, cosine))
    return NumericalDiagnostics(
        reference_shape=reference_shape,
        candidate_shape=candidate_shape,
        shape_equal=True,
        max_absolute_error=float(np.max(absolute_error)),
        mean_absolute_error=float(np.mean(absolute_error)),
        cosine_similarity=cosine,
    )


def qwen_weight_spec(bundle: AssetBundle) -> tuple[QwenWeightSpec, ...]:
    """Select the 310 Qwen tensors and prove their direct canonical assignments."""

    records = sorted(
        (record for record in bundle.tensors if record.component == QWEN_COMPONENT),
        key=lambda record: record.mlx_key,
    )
    if len(records) != QWEN_TENSOR_COUNT:
        raise QwenAdapterError(
            f"locked Qwen inventory must contain {QWEN_TENSOR_COUNT} tensors, got {len(records)}"
        )

    specifications: list[QwenWeightSpec] = []
    bare_keys: set[str] = set()
    for record in records:
        if record.transform != "none":
            raise QwenAdapterError(
                f"Qwen tensor {record.reference_name} requests forbidden transform "
                f"{record.transform!r}"
            )
        try:
            canonical = map_reference_key(record.reference_name)
        except WeightMappingError as exc:
            raise QwenAdapterError(str(exc)) from exc
        if canonical.component != QWEN_COMPONENT or canonical.mlx_key != record.mlx_key:
            raise QwenAdapterError(
                f"Qwen tensor {record.reference_name} disagrees with the canonical map"
            )
        if not record.mlx_key.startswith(QWEN_MLX_PREFIX):
            raise QwenAdapterError(
                f"Qwen destination lacks {QWEN_MLX_PREFIX!r} prefix: {record.mlx_key}"
            )
        bare_key = record.mlx_key.removeprefix(QWEN_MLX_PREFIX)
        if not bare_key or bare_key.startswith("model.") or "lm_head" in bare_key:
            raise QwenAdapterError(f"invalid bare Qwen destination: {bare_key!r}")
        if bare_key in bare_keys:
            raise QwenAdapterError(f"duplicate bare Qwen destination: {bare_key}")
        bare_keys.add(bare_key)
        specifications.append(
            QwenWeightSpec(
                reference_name=record.reference_name,
                bare_key=bare_key,
                shape=record.shape,
                dtype=record.dtype,
            )
        )
    return tuple(specifications)


def _runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    require_mlx_full_precision()
    try:
        installed_version = version("mlx-lm")
    except PackageNotFoundError as exc:  # pragma: no cover - environment failure
        raise QwenAdapterError("mlx-lm is not installed") from exc
    if installed_version != MLX_LM_VERSION:
        raise QwenAdapterError(f"mlx-lm version must be {MLX_LM_VERSION}, got {installed_version}")
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import KVCache, make_prompt_cache
        from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment failure
        raise QwenAdapterError(f"MLX Qwen runtime is unavailable: {exc}") from exc
    return mx, tree_flatten, KVCache, make_prompt_cache, ModelArgs, Qwen3Model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenAdapter:
    """Loaded bare Qwen3 transformer with strict weights and native KV caches."""

    def __init__(
        self,
        *,
        model: Any,
        configuration: QwenConfiguration,
        kv_cache_class: type[Any],
        make_prompt_cache: Any,
        mx: Any,
        tree_flatten: Any,
    ) -> None:
        self._model = model
        self.configuration = configuration
        self._kv_cache_class = kv_cache_class
        self._make_prompt_cache = make_prompt_cache
        self._mx = mx
        self._tree_flatten = tree_flatten

    @classmethod
    def from_asset_bundle(cls, bundle: AssetBundle) -> Self:
        """Construct, strict-load, freeze, and materialize the locked bare model."""

        if bundle.model_id != REFERENCE_MODEL_ID:
            raise QwenAdapterError(
                f"Qwen assets must come from {REFERENCE_MODEL_ID}, got {bundle.model_id}"
            )
        if bundle.revision != REFERENCE_REVISION:
            raise QwenAdapterError(
                f"Qwen asset revision must be {REFERENCE_REVISION}, got {bundle.revision}"
            )
        configuration = QwenConfiguration.from_model_config(bundle.config)
        specifications = qwen_weight_spec(bundle)
        mx, tree_flatten, kv_cache_class, make_prompt_cache, model_args, model_class = _runtime()
        arguments = model_args(
            model_type=configuration.model_type,
            hidden_size=configuration.hidden_size,
            num_hidden_layers=configuration.num_hidden_layers,
            intermediate_size=configuration.intermediate_size,
            num_attention_heads=configuration.num_attention_heads,
            rms_norm_eps=configuration.rms_norm_eps,
            vocab_size=configuration.vocab_size,
            num_key_value_heads=configuration.num_key_value_heads,
            max_position_embeddings=configuration.max_position_embeddings,
            rope_theta=configuration.rope_theta,
            head_dim=configuration.head_dim,
            tie_word_embeddings=configuration.tie_word_embeddings,
            rope_scaling=None,
        )
        model = model_class(arguments)
        if hasattr(model, "lm_head"):
            raise QwenAdapterError("bare Qwen3 model unexpectedly exposes an LM head")

        parameter_rows = dict(tree_flatten(model.parameters()))
        expected_shapes = {spec.bare_key: spec.shape for spec in specifications}
        actual_shapes = {
            name: tuple(int(dimension) for dimension in value.shape)
            for name, value in parameter_rows.items()
        }
        if set(actual_shapes) != set(expected_shapes):
            missing = sorted(set(expected_shapes) - set(actual_shapes))
            unexpected = sorted(set(actual_shapes) - set(expected_shapes))
            raise QwenAdapterError(
                "bare Qwen parameter keys disagree with the checkpoint; "
                f"missing={missing}, unexpected={unexpected}"
            )
        shape_mismatches = {
            name: {"checkpoint": expected_shapes[name], "model": actual_shapes[name]}
            for name in expected_shapes
            if expected_shapes[name] != actual_shapes[name]
        }
        if shape_mismatches:
            raise QwenAdapterError(
                f"bare Qwen parameter shapes disagree with the checkpoint: {shape_mismatches}"
            )

        actual_hash = _sha256(bundle.weights_path)
        if actual_hash != bundle.weights_sha256:
            raise QwenAdapterError(
                "model.safetensors changed after asset validation: "
                f"expected {bundle.weights_sha256}, got {actual_hash}"
            )
        specification_by_reference = {
            specification.reference_name: specification for specification in specifications
        }
        weights: list[tuple[str, Any]] = []
        with safe_open(bundle.weights_path, framework="np") as handle:
            available = set(handle.keys())
            missing_reference = sorted(set(specification_by_reference) - available)
            if missing_reference:
                raise QwenAdapterError(
                    f"safe weights are missing Qwen tensors: {missing_reference}"
                )
            for reference_name, specification in specification_by_reference.items():
                array = handle.get_tensor(reference_name)
                if tuple(array.shape) != specification.shape:
                    raise QwenAdapterError(
                        f"safe weight {reference_name} changed shape: "
                        f"expected {specification.shape}, got {tuple(array.shape)}"
                    )
                # Direct assignment is intentional: MLX Linear and the checkpoint both
                # use [out, in].  There is no transpose or other conversion step.
                weights.append((specification.bare_key, mx.array(array)))
            # ``mx.array`` construction is lazy.  Materialize while the safe-file handle
            # and its NumPy buffers are unquestionably still alive.
            mx.eval([value for _, value in weights])

        try:
            model.load_weights(weights, strict=True)
        except ValueError as exc:
            raise QwenAdapterError(f"strict Qwen weight load failed: {exc}") from exc
        model.eval()
        model.freeze()
        # MLX is lazy.  Materialize the direct assignments before the safe-file handle
        # and its NumPy views leave this method.
        mx.eval(model.parameters())
        return cls(
            model=model,
            configuration=configuration,
            kv_cache_class=kv_cache_class,
            make_prompt_cache=make_prompt_cache,
            mx=mx,
            tree_flatten=tree_flatten,
        )

    @property
    def model(self) -> Any:
        """Return the bare installed ``Qwen3Model`` (never an LM wrapper)."""

        return self._model

    def create_cache(self) -> list[Any]:
        """Create one distinct native, non-rotating MLX-LM KV cache per layer."""

        cache = list(self._make_prompt_cache(self._model))
        self.validate_cache(cache, expected_offset=0)
        return cache

    def cache_offsets(self, cache: Sequence[Any]) -> tuple[int, ...]:
        """Validate a cache and return its per-layer offsets."""

        self.validate_cache(cache)
        return tuple(int(item.offset) for item in cache)

    def validate_cache(
        self,
        cache: Sequence[Any],
        *,
        expected_offset: int | None = None,
    ) -> int:
        """Validate native cache count, type, identity, offsets, and K/V shapes.

        Returns the common token offset shared by all layers.
        """

        if not isinstance(cache, Sequence) or isinstance(cache, (str, bytes)):
            raise QwenAdapterError("Qwen cache must be a sequence")
        if len(cache) != self.configuration.num_hidden_layers:
            raise QwenAdapterError(
                "Qwen cache count must equal the decoder layer count: "
                f"expected {self.configuration.num_hidden_layers}, got {len(cache)}"
            )
        if len({id(item) for item in cache}) != len(cache):
            raise QwenAdapterError("each Qwen layer must own a distinct KV cache")

        offsets: list[int] = []
        for index, item in enumerate(cache):
            if not isinstance(item, self._kv_cache_class):
                raise QwenAdapterError(
                    f"Qwen cache layer {index} must be a native KVCache, got {type(item).__name__}"
                )
            offset = item.offset
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise QwenAdapterError(f"Qwen cache layer {index} has invalid offset {offset!r}")
            offsets.append(offset)
            if item.size() != offset:
                raise QwenAdapterError(f"Qwen cache layer {index} size and offset disagree")
            if item.empty():
                if offset != 0:
                    raise QwenAdapterError(f"Qwen cache layer {index} is empty at nonzero offset")
                continue
            if item.keys is None or item.values is None:
                raise QwenAdapterError(
                    f"Qwen cache layer {index} reports storage without K/V arrays"
                )
            key_shape = tuple(int(value) for value in item.keys.shape)
            value_shape = tuple(int(value) for value in item.values.shape)
            if len(key_shape) != 4 or len(value_shape) != 4:
                raise QwenAdapterError(f"Qwen cache layer {index} K/V must be rank four")
            if key_shape[:2] != value_shape[:2] or key_shape[2] != value_shape[2]:
                raise QwenAdapterError(f"Qwen cache layer {index} K/V shapes disagree")
            if key_shape[1] != self.configuration.num_key_value_heads:
                raise QwenAdapterError(
                    f"Qwen cache layer {index} has {key_shape[1]} KV heads, "
                    f"expected {self.configuration.num_key_value_heads}"
                )
            if key_shape[2] < offset:
                raise QwenAdapterError(
                    f"Qwen cache layer {index} capacity is smaller than its offset"
                )
            if key_shape[3] != self.configuration.head_dim:
                raise QwenAdapterError(
                    f"Qwen cache layer {index} key width is {key_shape[3]}, "
                    f"expected {self.configuration.head_dim}"
                )
            if value_shape[3] != self.configuration.head_dim:
                raise QwenAdapterError(
                    f"Qwen cache layer {index} value width is {value_shape[3]}, "
                    f"expected {self.configuration.head_dim}"
                )

        if len(set(offsets)) != 1:
            raise QwenAdapterError(f"Qwen cache layer offsets disagree: {offsets}")
        common_offset = offsets[0]
        if expected_offset is not None:
            if isinstance(expected_offset, bool) or not isinstance(expected_offset, int):
                raise QwenAdapterError("expected cache offset must be an integer")
            if common_offset != expected_offset:
                raise QwenAdapterError(
                    f"Qwen cache offset must be {expected_offset}, got {common_offset}"
                )
        return common_offset

    def cold_hidden_states(self, input_ids: Any) -> Any:
        """Return materialized final normalized hidden states without a cache."""

        identifiers = self._validated_input_ids(input_ids)
        hidden_states = self._model(identifiers, cache=None)
        self._mx.eval(hidden_states)
        self._validate_hidden_shape(hidden_states, identifiers)
        return hidden_states

    def cached_hidden_states(self, input_ids: Any, cache: Sequence[Any]) -> Any:
        """Append IDs to native K/V state and return materialized new-token states."""

        identifiers = self._validated_input_ids(input_ids)
        start_offset = self.validate_cache(cache)
        hidden_states = self._model(identifiers, cache=cache)
        # Cache assignments are lazy too.  Evaluate both the output and the state before
        # reporting its new offset or allowing the caller to time the append.
        cache_arrays = [value for _, value in self._tree_flatten([item.state for item in cache])]
        self._mx.eval(hidden_states, *cache_arrays)
        expected_offset = start_offset + int(identifiers.shape[1])
        self.validate_cache(cache, expected_offset=expected_offset)
        self._validate_hidden_shape(hidden_states, identifiers)
        return hidden_states

    def _validated_input_ids(self, input_ids: Any) -> Any:
        try:
            numpy_ids = np.asarray(input_ids)
        except (TypeError, ValueError) as exc:
            raise QwenAdapterError(f"input IDs cannot be converted to an array: {exc}") from exc
        if numpy_ids.dtype.kind not in "iu":
            raise QwenAdapterError("input IDs must have an integer dtype")
        if numpy_ids.ndim != 2:
            raise QwenAdapterError(
                f"input IDs must have shape [batch, tokens], got {numpy_ids.shape}"
            )
        if numpy_ids.shape[0] < 1 or numpy_ids.shape[1] < 1:
            raise QwenAdapterError("input IDs must contain at least one batch row and token")
        minimum = int(numpy_ids.min())
        maximum = int(numpy_ids.max())
        if minimum < 0 or maximum >= self.configuration.vocab_size:
            raise QwenAdapterError(
                f"input IDs must be in [0, {self.configuration.vocab_size}); "
                f"observed [{minimum}, {maximum}]"
            )
        return self._mx.array(numpy_ids, dtype=self._mx.int32)

    def _validate_hidden_shape(self, hidden_states: Any, input_ids: Any) -> None:
        expected = (
            int(input_ids.shape[0]),
            int(input_ids.shape[1]),
            self.configuration.hidden_size,
        )
        actual = tuple(int(value) for value in hidden_states.shape)
        if actual != expected:
            raise QwenAdapterError(f"Qwen hidden-state shape must be {expected}, got {actual}")
