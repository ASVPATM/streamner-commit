"""Deterministic, graph-free parity capture for the pinned reference model.

This module intentionally avoids importing ``torch`` or ``gliner`` at import
time.  The capture entry point is reference-environment-only, while the safe
NPZ and metadata helpers remain usable from the main MLX environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from streamner_commit.backends.reference_gliner import require_pinned_gliner

PARITY_SCHEMA_VERSION = 1
ARRAYS_FILENAME = "parity_arrays.npz"
METADATA_FILENAME = "parity_metadata.json"
DEFAULT_PARITY_TEXT = "Ada emailed Jo."
DEFAULT_PARITY_LABELS = ("person", "email address")

REQUIRED_ARRAY_NAMES = frozenset(
    {
        "input_ids",
        "attention_mask",
        "label_attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask",
        "label_token_positions",
        "separator_token_positions",
        "qwen_final_hidden_states",
        "label_encoder_input_hidden_states",
        "prompt_input_hidden_states",
        "prompt_input_ids",
        "prompt_attention_mask",
        "contextualized_prompt_hidden_states",
        "label_representations_pre_projection",
        "label_mask",
        "pooled_word_states",
        "pooled_word_mask",
        "contextualized_word_states",
        "contextualized_word_mask",
        "marker_v2_span_representations",
        "label_representations_post_projection",
        "raw_logits",
    }
)


class ReferenceParityError(RuntimeError):
    """Raised when the pinned model violates a parity-capture invariant."""


@dataclass(frozen=True, slots=True)
class ParityFixture:
    """Detached reference arrays and deterministic, path-free metadata."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def validated_arrays(self) -> dict[str, np.ndarray]:
        """Return copied, normalized arrays after safe-serialization checks."""

        return validate_arrays(self.arrays, required=REQUIRED_ARRAY_NAMES)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    payload = contiguous.dtype.str.encode("ascii") + b"\0"
    payload += json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    payload += b"\0" + contiguous.tobytes(order="C")
    return _sha256(payload)


def _validate_array_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("array names must be nonempty strings")
    if not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ValueError(f"unsafe parity array name: {name!r}")
    return name


def validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    required: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Normalize arrays and reject pickle-requiring or non-finite values."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a nonempty mapping")
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"parity fixture is missing required arrays: {', '.join(missing)}")

    normalized: dict[str, np.ndarray] = {}
    for raw_name, raw_array in arrays.items():
        name = _validate_array_name(raw_name)
        array = np.asarray(raw_array)
        if array.dtype.hasobject or array.dtype.kind in {"O", "S", "U", "V"}:
            raise ValueError(f"{name} has unsafe or non-numeric dtype {array.dtype}")
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        normalized[name] = np.ascontiguousarray(array).copy()
    return normalized


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize numeric arrays as a byte-stable, pickle-free NPZ archive."""

    normalized = validate_arrays(arrays)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(normalized):
            npy = io.BytesIO()
            np.lib.format.write_array(
                npy,
                normalized[name],
                version=(2, 0),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy.getvalue())
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_parity_fixture(fixture: ParityFixture, output_dir: Path) -> dict[str, Any]:
    """Write deterministic NPZ/JSON fixture files and return artifact metadata."""

    arrays = fixture.validated_arrays()
    npz_payload = deterministic_npz_bytes(arrays)
    array_manifest = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "numel": int(array.size),
            "nbytes": int(array.nbytes),
            "sha256": _array_digest(array),
        }
        for name, array in sorted(arrays.items())
    }
    metadata = dict(fixture.metadata)
    metadata.update(
        {
            "schema_version": PARITY_SCHEMA_VERSION,
            "arrays_file": {
                "filename": ARRAYS_FILENAME,
                "sha256": _sha256(npz_payload),
                "size_bytes": len(npz_payload),
            },
            "arrays": array_manifest,
        }
    )
    metadata_payload = _json_bytes(metadata)

    output_dir = Path(output_dir)
    arrays_path = output_dir / ARRAYS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    _atomic_write(arrays_path, npz_payload)
    _atomic_write(metadata_path, metadata_payload)
    return {
        "arrays_path": arrays_path,
        "metadata_path": metadata_path,
        "arrays_sha256": _sha256(npz_payload),
        "metadata_sha256": _sha256(metadata_payload),
        "array_count": len(arrays),
    }


def _host_array(value: object, *, name: str) -> np.ndarray:
    """Immediately detach one tensor, move it to CPU, and own its storage."""

    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise ReferenceParityError(f"{name} is not a detachable tensor")
    host = detach()
    cpu = getattr(host, "cpu", None)
    if not callable(cpu):
        raise ReferenceParityError(f"{name} cannot be copied to CPU")
    host = cpu()
    contiguous = getattr(host, "contiguous", None)
    if callable(contiguous):
        host = contiguous()
    to_numpy = getattr(host, "numpy", None)
    if not callable(to_numpy):
        raise ReferenceParityError(f"{name} cannot be converted to a NumPy array")
    try:
        array = np.asarray(to_numpy())
    except TypeError:
        # NumPy cannot represent torch.bfloat16.  Preserve values in float32 and
        # record all reference captures under the enforced CPU/fp32 load path.
        to_float = getattr(host, "float", None)
        if not callable(to_float):
            raise
        array = np.asarray(to_float().numpy())
    if array.dtype.hasobject:
        raise ReferenceParityError(f"{name} produced an object array")
    result = np.ascontiguousarray(array).copy()
    if result.dtype.kind in {"f", "c"} and not np.isfinite(result).all():
        raise ReferenceParityError(f"{name} contains non-finite values")
    return result


def _pair(output: object, *, name: str) -> tuple[object, object]:
    if not isinstance(output, tuple | list) or len(output) < 2:
        raise ReferenceParityError(f"{name} must return at least two values")
    return output[0], output[1]


class _HookCapture:
    """Own short-lived forward hooks and enforce one firing per cold path."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.classes: dict[str, str] = {}
        self._handles: list[object] = []

    def add_forward(
        self,
        path: str,
        module: object,
        capture: Callable[[object, tuple[object, ...], object], None],
    ) -> None:
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise ReferenceParityError(f"{path} does not support forward hooks")
        self.counts[path] = 0
        self.classes[path] = type(module).__name__

        def hook(hooked: object, inputs: tuple[object, ...], output: object) -> None:
            self.counts[path] += 1
            capture(hooked, inputs, output)

        self._handles.append(register(hook))

    def add_pre_forward(
        self,
        path: str,
        module: object,
        capture: Callable[[object, tuple[object, ...]], None],
    ) -> None:
        register = getattr(module, "register_forward_pre_hook", None)
        if not callable(register):
            raise ReferenceParityError(f"{path} does not support forward pre-hooks")
        self.counts[path] = 0
        self.classes[path] = type(module).__name__

        def hook(hooked: object, inputs: tuple[object, ...]) -> None:
            self.counts[path] += 1
            capture(hooked, inputs)

        self._handles.append(register(hook))

    def put(self, name: str, value: object) -> None:
        if name in self.arrays:
            raise ReferenceParityError(f"capture attempted to overwrite {name}")
        self.arrays[name] = _host_array(value, name=name)

    def assert_once(self) -> None:
        failures = {path: count for path, count in self.counts.items() if count != 1}
        if failures:
            details = ", ".join(f"{path}={count}" for path, count in sorted(failures.items()))
            raise ReferenceParityError(
                f"each hooked component must fire exactly once during cold capture: {details}"
            )

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {
            path: {
                "class": self.classes[path],
                "expected_firings": 1,
                "observed_firings": self.counts[path],
            }
            for path in sorted(self.counts)
        }

    def close(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()


def _plain_config(config: object) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = {key: value for key, value in vars(config).items() if not key.startswith("_")}
    if not isinstance(raw, Mapping):
        raise ReferenceParityError("model config did not serialize to an object")
    return json.loads(json.dumps(raw, default=str, allow_nan=False))


def _config_snapshot(config: object) -> tuple[dict[str, Any], str]:
    full = _plain_config(config)
    canonical = json.dumps(full, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    keys = (
        "model_type",
        "model_name",
        "hidden_size",
        "max_width",
        "right_context_width",
        "span_mode",
        "subtoken_pooling",
        "words_splitter_type",
        "label_token",
        "sep_token",
        "class_token_index",
        "sep_token_index",
        "embed_ent_token",
        "vocab_size",
        "dropout",
    )
    snapshot = {key: full.get(key) for key in keys}
    for key in ("decoder_config", "labels_encoder_config", "span_encoder_config"):
        snapshot[key] = full.get(key)
    return snapshot, _sha256(canonical)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence of strings")
    normalized = tuple(labels)
    if not normalized or not all(isinstance(label, str) and label.strip() for label in normalized):
        raise ValueError("labels must contain nonblank strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("labels must be unique and ordered")
    return normalized


def _require_component(parent: object, name: str, *, expected_class: str | None = None) -> object:
    component = getattr(parent, name, None)
    if component is None:
        raise ReferenceParityError(f"reference model is missing component {name}")
    if expected_class is not None and type(component).__name__ != expected_class:
        raise ReferenceParityError(
            f"{name} must be {expected_class}, found {type(component).__name__}"
        )
    return component


def capture_reference_parity(
    model: object,
    *,
    text: str = DEFAULT_PARITY_TEXT,
    labels: Sequence[str] = DEFAULT_PARITY_LABELS,
    model_id: str,
    model_revision: str,
) -> ParityFixture:
    """Capture one exact cold StreamingSpan forward from the pinned CPU oracle."""

    require_pinned_gliner()
    if type(model).__name__ != "StreamingSpanGLiNER":
        raise ReferenceParityError(f"expected StreamingSpanGLiNER, found {type(model).__name__}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonblank string")
    ordered_labels = _normalize_labels(labels)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonblank string")
    if not isinstance(model_revision, str) or len(model_revision) != 40:
        raise ValueError("model_revision must be a 40-character commit SHA")

    internal = _require_component(model, "model", expected_class="StreamingSpanModel")
    token_rep_layer = _require_component(internal, "token_rep_layer", expected_class="Decoder")
    labels_encoder = _require_component(
        internal,
        "labels_encoder",
        expected_class="StreamingSpanLabelsEncoder",
    )
    label_context = _require_component(labels_encoder, "encoder")
    span_rep_layer = _require_component(internal, "span_rep_layer", expected_class="SpanRepLayer")
    marker = _require_component(span_rep_layer, "span_rep_layer", expected_class="SpanMarkerV2")
    prompt_projection = _require_component(internal, "prompt_rep_layer")

    prepare_inputs = getattr(model, "prepare_inputs", None)
    collate = getattr(model, "_collate_session_tokens", None)
    create_cache = getattr(model, "_create_session_cache", None)
    processor = getattr(model, "data_processor", None)
    if not all(callable(value) for value in (prepare_inputs, collate, create_cache)):
        raise ReferenceParityError("pinned cold preprocessing helpers are unavailable")
    if processor is None:
        raise ReferenceParityError("reference model has no data processor")

    prepare_inputs_call = cast(Callable[..., Any], prepare_inputs)
    collate_call = cast(Callable[..., Any], collate)
    create_cache_call = cast(Callable[..., Any], create_cache)
    internal_call = cast(Callable[..., Any], internal)

    words_rows, starts_rows, ends_rows = prepare_inputs_call([text])
    if len(words_rows) != 1 or len(starts_rows) != 1 or len(ends_rows) != 1:
        raise ReferenceParityError("prepare_inputs did not return one row")
    words = list(words_rows[0])
    char_starts = list(starts_rows[0])
    char_ends = list(ends_rows[0])
    if not words or not (len(words) == len(char_starts) == len(char_ends)):
        raise ReferenceParityError("word and character metadata are inconsistent")
    for word, start, end in zip(words, char_starts, char_ends, strict=True):
        if text[start:end] != word:
            raise ReferenceParityError("word splitter offsets do not round-trip to the input text")

    prepare_serialized = getattr(processor, "prepare_inputs", None)
    if not callable(prepare_serialized):
        raise ReferenceParityError("streaming processor prepare_inputs is unavailable")
    serialized_rows, prompt_lengths = prepare_serialized(
        [words],
        list(ordered_labels),
        include_prompt=True,
    )
    if len(serialized_rows) != 1 or len(prompt_lengths) != 1:
        raise ReferenceParityError("streaming processor did not serialize one prompt row")
    serialized_words = list(serialized_rows[0])
    prompt_length = int(prompt_lengths[0])
    if serialized_words[prompt_length:] != words:
        raise ReferenceParityError("serialized input text suffix differs from split words")
    prompt_words = serialized_words[:prompt_length]

    batch = collate_call(words, list(ordered_labels), include_prompt=True)
    if not isinstance(batch, Mapping):
        raise ReferenceParityError("cold collator did not return a mapping")
    torch = import_module("torch")
    device = getattr(model, "device", "cpu")
    if str(device) != "cpu":
        raise ReferenceParityError(f"reference parity capture requires CPU, found {device}")
    model_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    captures = _HookCapture()

    def capture_qwen(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        hidden, _cache = _pair(output, name="token_rep_layer")
        captures.put("qwen_final_hidden_states", hidden)

    def capture_label_input(module: object, inputs: tuple[object, ...]) -> None:
        if len(inputs) < 3:
            raise ReferenceParityError("labels_encoder pre-hook expected three positional inputs")
        captures.put("label_encoder_input_hidden_states", inputs[0])
        slice_prompt = getattr(module, "_slice_prompt", None)
        if not callable(slice_prompt):
            raise ReferenceParityError("labels_encoder._slice_prompt is unavailable")
        prompt_hidden, prompt_ids, prompt_mask = slice_prompt(inputs[0], inputs[1], inputs[2])
        captures.put("prompt_input_hidden_states", prompt_hidden)
        captures.put("prompt_input_ids", prompt_ids)
        captures.put("prompt_attention_mask", prompt_mask)

    def capture_context(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        captures.put("contextualized_prompt_hidden_states", output)

    def capture_labels(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        label_representations, label_mask = _pair(output, name="labels_encoder")
        captures.put("label_representations_pre_projection", label_representations)
        captures.put("label_mask", label_mask)

    def capture_span(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        span_representations, contextualized_words = _pair(output, name="span_rep_layer")
        captures.put("marker_v2_span_representations", span_representations)
        captures.put("contextualized_word_states", contextualized_words)

    def capture_projection(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        captures.put("label_representations_post_projection", output)

    captures.add_forward("model.token_rep_layer", token_rep_layer, capture_qwen)
    captures.add_pre_forward("model.labels_encoder", labels_encoder, capture_label_input)
    captures.add_forward("model.labels_encoder.encoder", label_context, capture_context)
    captures.add_forward("model.labels_encoder.output", labels_encoder, capture_labels)
    captures.add_forward("model.span_rep_layer", span_rep_layer, capture_span)
    captures.add_forward("model.prompt_rep_layer", prompt_projection, capture_projection)

    token_projection = getattr(internal, "token_projection", None)
    if token_projection is not None:
        captures.add_forward(
            "model.token_projection",
            token_projection,
            lambda _module, _inputs, output: captures.put(
                "token_projected_hidden_states",
                output,
            ),
        )

    was_training = bool(getattr(internal, "training", False))
    eval_method = getattr(internal, "eval", None)
    train_method = getattr(internal, "train", None)
    try:
        if callable(eval_method):
            eval_method()
        with torch.inference_mode():
            output = internal_call(**model_batch, past_key_values=create_cache_call())
        captures.assert_once()
    finally:
        captures.close()
        if was_training and callable(train_method):
            train_method(True)

    arrays = {
        "input_ids": _host_array(model_batch["input_ids"], name="input_ids"),
        "attention_mask": _host_array(model_batch["attention_mask"], name="attention_mask"),
        "label_attention_mask": _host_array(
            model_batch["attention_mask"],
            name="label_attention_mask",
        ),
        "words_mask": _host_array(model_batch["words_mask"], name="words_mask"),
        "text_lengths": _host_array(model_batch["text_lengths"], name="text_lengths"),
        "span_idx": _host_array(model_batch["span_idx"], name="span_idx"),
        "span_mask": _host_array(model_batch["span_mask"], name="span_mask"),
        **captures.arrays,
        "raw_logits": _host_array(getattr(output, "logits", None), name="raw_logits"),
        "pooled_word_states": _host_array(
            getattr(output, "past_word_embeddings", None),
            name="pooled_word_states",
        ),
        "pooled_word_mask": _host_array(
            getattr(output, "past_word_mask", None),
            name="pooled_word_mask",
        ),
        "contextualized_word_mask": _host_array(
            getattr(output, "mask", None),
            name="contextualized_word_mask",
        ),
    }

    input_ids = arrays["input_ids"]
    config = getattr(model, "config", None)
    if config is None:
        raise ReferenceParityError("reference model has no config")
    class_token_index = int(getattr(config, "class_token_index", -1))
    sep_token_index = int(getattr(config, "sep_token_index", -1))
    if class_token_index < 0 or sep_token_index < 0:
        raise ReferenceParityError("reference special-token indices are unset")
    arrays["label_token_positions"] = np.argwhere(input_ids == class_token_index).astype(
        np.int64,
        copy=False,
    )
    arrays["separator_token_positions"] = np.argwhere(input_ids == sep_token_index).astype(
        np.int64,
        copy=False,
    )
    if arrays["label_token_positions"].shape[0] != len(ordered_labels):
        raise ReferenceParityError("label-token count differs from ordered label count")
    if arrays["separator_token_positions"].shape[0] != 1:
        raise ReferenceParityError("cold prompt must contain exactly one separator token")

    tokenizer = getattr(processor, "transformer_tokenizer", None)
    convert_ids = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert_ids):
        raise ReferenceParityError("reference tokenizer cannot convert IDs back to tokens")
    valid_ids = input_ids[0, arrays["attention_mask"][0].astype(bool)].tolist()
    tokenizer_tokens = [str(token) for token in convert_ids(valid_ids)]

    snapshot, config_sha256 = _config_snapshot(config)
    metadata = {
        "fixture_kind": "cold_reference_parity",
        "model_id": model_id,
        "model_revision_sha": model_revision,
        "gliner_version": require_pinned_gliner(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "device": "cpu",
        "load_dtype": "float32",
        "text": text,
        "labels": list(ordered_labels),
        "word_tokens": words,
        "word_char_starts": char_starts,
        "word_char_ends": char_ends,
        "serialized_prompt": "".join(prompt_words),
        "serialized_prompt_words": prompt_words,
        "serialized_input_words": serialized_words,
        "prompt_word_length": prompt_length,
        "tokenizer_tokens": tokenizer_tokens,
        "special_tokens": {
            "label_token": str(getattr(config, "label_token", "")),
            "label_token_id": class_token_index,
            "separator_token": str(getattr(config, "sep_token", "")),
            "separator_token_id": sep_token_index,
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size_with_added_tokens": len(cast(Any, tokenizer)),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "add_special_tokens": False,
            "is_split_into_words": True,
        },
        "config_sha256": config_sha256,
        "config_snapshot": snapshot,
        "component_classes": {
            "wrapper": type(model).__name__,
            "model": type(internal).__name__,
            "token_rep_layer": type(token_rep_layer).__name__,
            "labels_encoder": type(labels_encoder).__name__,
            "label_context_encoder": type(label_context).__name__,
            "span_rep_layer": type(span_rep_layer).__name__,
            "marker": type(marker).__name__,
            "prompt_projection": type(prompt_projection).__name__,
        },
        "hook_firing_invariants": captures.metadata(),
        "capture_contract": {
            "inference_mode": True,
            "model_eval": True,
            "all_arrays_detached_to_cpu_immediately": True,
            "graphs_serialized": False,
            "pickle_allowed": False,
            "cold_forward_count": 1,
        },
    }
    return ParityFixture(
        arrays=validate_arrays(arrays, required=REQUIRED_ARRAY_NAMES),
        metadata=metadata,
    )


__all__ = [
    "ARRAYS_FILENAME",
    "DEFAULT_PARITY_LABELS",
    "DEFAULT_PARITY_TEXT",
    "METADATA_FILENAME",
    "PARITY_SCHEMA_VERSION",
    "ParityFixture",
    "ReferenceParityError",
    "capture_reference_parity",
    "deterministic_npz_bytes",
    "validate_arrays",
    "write_parity_fixture",
]
