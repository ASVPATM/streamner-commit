"""Lazy, reference-environment-only access to the pinned GLiNER oracle.

This module deliberately contains no top-level ``gliner`` or ``torch`` import.
It is therefore safe to import while running the project's main MLX environment.
Only :meth:`ReferenceGLiNERBackend.from_pretrained` resolves the reference
dependency and loads checkpoint weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Protocol, Self

from streamner_commit.backends.base import EntitySnapshot

PINNED_GLINER_VERSION = "0.2.28"
DEFAULT_MODEL_ID = "knowledgator/gliner-stream-pii-v1.0"
DEFAULT_THRESHOLD = 0.5


class ReferenceBackendError(RuntimeError):
    """Raised when the isolated GLiNER reference backend cannot be used safely."""


class _ReferenceModel(Protocol):
    def inference(self, texts: list[str], labels: list[str], **kwargs: Any) -> Any: ...

    def clear_session(self, session_id: str | list[str]) -> None: ...

    def clear_sessions(self) -> None: ...


class _GLiNERFactory(Protocol):
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _ReferenceModel: ...


def require_pinned_gliner() -> str:
    """Return the installed GLiNER version or raise with environment guidance."""

    try:
        installed = distribution_version("gliner")
    except PackageNotFoundError as error:
        raise ReferenceBackendError(
            "GLiNER is not installed. Run reference work with "
            "`PYTHONPATH=src .venv-reference/bin/python ...`."
        ) from error

    if installed != PINNED_GLINER_VERSION:
        raise ReferenceBackendError(
            f"GLiNER {PINNED_GLINER_VERSION} is required; found {installed}. "
            "Do not run the reference backend against a different release."
        )
    return installed


def _load_gliner_factory() -> type[_GLiNERFactory]:
    require_pinned_gliner()
    module = import_module("gliner")
    factory = getattr(module, "GLiNER", None)
    if factory is None or not callable(getattr(factory, "from_pretrained", None)):
        raise ReferenceBackendError(
            "The installed gliner package does not expose GLiNER.from_pretrained"
        )
    return factory


class ReferenceGLiNERBackend:
    """Thin adapter around one loaded v0.2.28 StreamingSpan model.

    Direct construction accepts an already-loaded model, which keeps unit tests
    independent of both PyTorch and checkpoint files.  Production callers should
    use :meth:`from_pretrained`, which enforces the reference version and loads on
    CPU in float32 by default.
    """

    def __init__(self, model: _ReferenceModel) -> None:
        if not callable(getattr(model, "inference", None)):
            raise TypeError("model must provide an inference method")
        if not callable(getattr(model, "clear_session", None)):
            raise TypeError("model must provide a clear_session method")
        self._model = model

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        **model_kwargs: Any,
    ) -> Self:
        """Load the pinned oracle lazily using deterministic CPU/fp32 defaults."""

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-blank string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be null or a non-blank string")
        if "map_location" in model_kwargs or "dtype" in model_kwargs:
            raise TypeError("map_location and dtype are fixed to cpu and float32")

        factory = _load_gliner_factory()
        load_options: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
            "map_location": "cpu",
            "dtype": "float32",
            **model_kwargs,
        }
        if cache_dir is not None:
            load_options["cache_dir"] = cache_dir
        model = factory.from_pretrained(model_id, **load_options)
        return cls(model)

    @property
    def raw_model(self) -> _ReferenceModel:
        """Expose the pinned model for later reference-only instrumentation."""

        return self._model

    def inference(self, texts: list[str], labels: list[str], **kwargs: Any) -> Any:
        """Call the public GLiNER inference API without altering its result."""

        return self._model.inference(texts, labels, **kwargs)

    def append(
        self,
        chunk: str,
        labels: Sequence[str],
        session_id: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        recompute: bool = False,
    ) -> EntitySnapshot:
        """Append one exact chunk and return its complete public snapshot."""

        _require_text("chunk", chunk, allow_empty=True)
        normalized_labels = _normalize_labels(labels)
        _require_text("session_id", session_id)
        normalized_threshold = _normalize_threshold(threshold)
        _require_bool("flat_ner", flat_ner)
        _require_bool("multi_label", multi_label)
        _require_bool("return_class_probs", return_class_probs)
        _require_bool("recompute", recompute)

        output = self._model.inference(
            [chunk],
            normalized_labels,
            session_id=[session_id],
            threshold=normalized_threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
            recompute=recompute,
        )
        return _single_snapshot(output)

    def infer_full(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
    ) -> EntitySnapshot:
        """Run cold full-text inference without assigning a session ID."""

        _require_text("text", text, allow_empty=True)
        normalized_labels = _normalize_labels(labels)
        normalized_threshold = _normalize_threshold(threshold)
        _require_bool("flat_ner", flat_ner)
        _require_bool("multi_label", multi_label)
        _require_bool("return_class_probs", return_class_probs)

        output = self._model.inference(
            [text],
            normalized_labels,
            threshold=normalized_threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            return_class_probs=return_class_probs,
        )
        return _single_snapshot(output)

    def clear_session(self, session_id: str) -> None:
        """Clear one cached reference session."""

        _require_text("session_id", session_id)
        self._model.clear_session(session_id)

    def clear_sessions(self) -> None:
        """Clear every cached session when the loaded model exposes that API."""

        clear_all = getattr(self._model, "clear_sessions", None)
        if not callable(clear_all):
            raise ReferenceBackendError("The loaded model does not expose clear_sessions")
        clear_all()


def _single_snapshot(output: Any) -> EntitySnapshot:
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], list):
        raise ReferenceBackendError("GLiNER inference must return one snapshot per input text")
    snapshot = output[0]
    if not all(isinstance(entity, Mapping) for entity in snapshot):
        raise ReferenceBackendError("Every GLiNER entity must be a mapping")
    return [dict(entity) for entity in snapshot]


def _normalize_labels(labels: Sequence[str]) -> list[str]:
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence of strings")
    normalized = list(labels)
    if not normalized:
        raise ValueError("labels must not be empty")
    if not all(isinstance(label, str) and label.strip() for label in normalized):
        raise ValueError("labels must contain only non-blank strings")
    return normalized


def _normalize_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError("threshold must be a real number")
    normalized = float(threshold)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    return normalized


def _require_text(name: str, value: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be non-blank")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_THRESHOLD",
    "PINNED_GLINER_VERSION",
    "ReferenceBackendError",
    "ReferenceGLiNERBackend",
    "require_pinned_gliner",
]
