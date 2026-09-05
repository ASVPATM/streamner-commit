from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from streamner_commit.backends import reference_gliner
from streamner_commit.backends.base import StreamingNERBackend


class FakeModel:
    def __init__(self, output: Any | None = None) -> None:
        self.output = output if output is not None else [[]]
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []
        self.cleared: list[str | list[str]] = []
        self.clear_all_count = 0

    def inference(self, texts: list[str], labels: list[str], **kwargs: Any) -> Any:
        self.calls.append((texts, labels, kwargs))
        return self.output

    def clear_session(self, session_id: str | list[str]) -> None:
        self.cleared.append(session_id)

    def clear_sessions(self) -> None:
        self.clear_all_count += 1


def test_module_reload_does_not_import_gliner_or_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    attempted: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", maxsplit=1)[0] in {"gliner", "torch"}:
            attempted.append(name)
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(reference_gliner)

    assert attempted == []


def test_missing_gliner_has_reference_environment_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> str:
        raise reference_gliner.PackageNotFoundError("gliner")

    monkeypatch.setattr(reference_gliner, "distribution_version", missing)

    with pytest.raises(reference_gliner.ReferenceBackendError, match=".venv-reference"):
        reference_gliner.require_pinned_gliner()


def test_wrong_gliner_version_is_a_hard_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference_gliner, "distribution_version", lambda _name: "0.2.29")

    with pytest.raises(reference_gliner.ReferenceBackendError, match="0.2.28.*0.2.29"):
        reference_gliner.require_pinned_gliner()


def test_from_pretrained_is_lazy_and_uses_cpu_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel()
    captured: dict[str, Any] = {}

    class FakeFactory:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return fake_model

    imported: list[str] = []

    def fake_import(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(GLiNER=FakeFactory)

    monkeypatch.setattr(reference_gliner, "distribution_version", lambda _name: "0.2.28")
    monkeypatch.setattr(reference_gliner, "import_module", fake_import)

    backend = reference_gliner.ReferenceGLiNERBackend.from_pretrained(
        revision="revision-sha",
        local_files_only=True,
        cache_dir="model-cache",
        strict=True,
    )

    assert imported == ["gliner"]
    assert captured == {
        "model_id": reference_gliner.DEFAULT_MODEL_ID,
        "kwargs": {
            "revision": "revision-sha",
            "local_files_only": True,
            "map_location": "cpu",
            "dtype": "float32",
            "strict": True,
            "cache_dir": "model-cache",
        },
    }
    assert backend.raw_model is fake_model


@pytest.mark.parametrize("reserved", [{"map_location": "mps"}, {"dtype": "float16"}])
def test_reference_load_cannot_override_cpu_float32(reserved: dict[str, str]) -> None:
    with pytest.raises(TypeError, match="fixed to cpu and float32"):
        reference_gliner.ReferenceGLiNERBackend.from_pretrained(**reserved)


def test_loaded_module_must_expose_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference_gliner, "distribution_version", lambda _name: "0.2.28")
    monkeypatch.setattr(reference_gliner, "import_module", lambda _name: SimpleNamespace())

    with pytest.raises(reference_gliner.ReferenceBackendError, match="GLiNER.from_pretrained"):
        reference_gliner.ReferenceGLiNERBackend.from_pretrained()


def test_append_calls_streaming_session_api_and_returns_complete_snapshot() -> None:
    entity = {"start": 9, "end": 14, "text": "Sarah", "label": "person", "score": 0.8}
    model = FakeModel([[entity]])
    backend = reference_gliner.ReferenceGLiNERBackend(model)

    snapshot = backend.append(
        " Sarah",
        ("person", "email address"),
        "session-1",
        threshold=0.6,
        return_class_probs=True,
    )

    assert snapshot == [entity]
    assert snapshot is not model.output[0]
    assert model.calls == [
        (
            [" Sarah"],
            ["person", "email address"],
            {
                "session_id": ["session-1"],
                "threshold": 0.6,
                "flat_ner": True,
                "multi_label": False,
                "return_class_probs": True,
                "recompute": False,
            },
        )
    ]


def test_full_inference_is_cold_and_has_no_session_id() -> None:
    model = FakeModel([[]])
    backend = reference_gliner.ReferenceGLiNERBackend(model)

    assert backend.infer_full("No PII here.", ["person"]) == []

    texts, labels, kwargs = model.calls[0]
    assert texts == ["No PII here."]
    assert labels == ["person"]
    assert "session_id" not in kwargs
    assert kwargs == {
        "threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "return_class_probs": False,
    }


def test_clear_session_and_clear_sessions_delegate() -> None:
    model = FakeModel()
    backend = reference_gliner.ReferenceGLiNERBackend(model)

    backend.clear_session("session-1")
    backend.clear_sessions()

    assert model.cleared == ["session-1"]
    assert model.clear_all_count == 1


@pytest.mark.parametrize("output", [[], [[], []], {}, [["not-a-mapping"]]])
def test_invalid_reference_output_shape_is_rejected(output: Any) -> None:
    backend = reference_gliner.ReferenceGLiNERBackend(FakeModel(output))

    with pytest.raises(reference_gliner.ReferenceBackendError):
        backend.infer_full("text", ["person"])


@pytest.mark.parametrize("labels", [[], [""], ["person", "  "], "person"])
def test_invalid_labels_are_rejected(labels: Any) -> None:
    backend = reference_gliner.ReferenceGLiNERBackend(FakeModel())

    with pytest.raises((TypeError, ValueError)):
        backend.append("chunk", labels, "session")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_thresholds_are_rejected(threshold: float) -> None:
    backend = reference_gliner.ReferenceGLiNERBackend(FakeModel())

    with pytest.raises(ValueError, match="between 0 and 1"):
        backend.infer_full("text", ["person"], threshold=threshold)


def test_backend_matches_structural_protocol() -> None:
    backend: StreamingNERBackend = reference_gliner.ReferenceGLiNERBackend(FakeModel())
    assert backend.infer_full("text", ["person"]) == []
