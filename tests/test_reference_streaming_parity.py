from __future__ import annotations

import builtins
import importlib
import math
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.chunking import word_char_spans
from streamner_commit.reference import streaming_parity
from streamner_commit.reference.observer import ObservedAppend
from streamner_commit.types import PublicEntity, SnapshotStep, SpanBoundary, SpanScoreUpdate

LABELS = ("person", "email address")
REVISION = "e871777dc4b3b688747a0433fff8d94a36fcc7b0"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class FakeCacheState:
    def __init__(
        self,
        *,
        text: str,
        labels: tuple[str, ...],
        scores: dict[SpanBoundary, tuple[float, ...]],
    ) -> None:
        spans = word_char_spans(text)
        self.text = text
        self.tokens = [text[start:end] for start, end in spans]
        self.char_starts = [start for start, _ in spans]
        self.char_ends = [end for _, end in spans]
        self.labels = labels
        self.span_logits = {boundary.to_tuple(): logits for boundary, logits in scores.items()}


class FakeCache:
    def __init__(self) -> None:
        self.states: dict[str, FakeCacheState] = {}

    def get(self, session_id: str) -> FakeCacheState | None:
        return self.states.get(session_id)


class FakeModel:
    def __init__(self) -> None:
        self._session_cache = FakeCache()
        self.clear_calls: list[str] = []

    def clear_session(self, session_id: str) -> None:
        self.clear_calls.append(session_id)
        self._session_cache.states.pop(session_id, None)


class FakeObserver:
    def __init__(
        self,
        model: FakeModel,
        *,
        run_id: str,
        example_id: str,
        session_id: str,
        labels: tuple[str, ...],
        **_kwargs: Any,
    ) -> None:
        self.model = model
        self.run_id = run_id
        self.example_id = example_id
        self.session_id = session_id
        self.labels = labels
        self.text = ""
        self.scores: dict[SpanBoundary, tuple[float, ...]] = {}

    def __enter__(self) -> FakeObserver:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.model.clear_session(self.session_id)

    def _update(
        self,
        *,
        boundary: SpanBoundary,
        step: int,
        chunk: str,
        words: list[str],
        starts: list[int],
        ends: list[int],
    ) -> SpanScoreUpdate:
        previous = self.scores.get(boundary)
        magnitude = float(step + boundary.end_word + 1)
        logits = (magnitude, -magnitude)
        probs = (_sigmoid(magnitude), _sigmoid(-magnitude))
        previous_top = None if previous is None else max(_sigmoid(value) for value in previous)
        self.scores[boundary] = logits
        start_char = starts[boundary.start_word]
        end_char = ends[boundary.end_word]
        return SpanScoreUpdate(
            run_id=self.run_id,
            example_id=self.example_id,
            step=step,
            chunk=chunk,
            visible_char_count=len(self.text),
            visible_word_count=len(words),
            start_word=boundary.start_word,
            end_word=boundary.end_word,
            start_char=start_char,
            end_char=end_char,
            span_text=self.text[start_char:end_char],
            logits=logits,
            probs=probs,
            top_label_index=0,
            top_label=self.labels[0],
            top_probability=probs[0],
            second_probability=probs[1],
            label_margin=probs[0] - probs[1],
            previous_top_probability=previous_top,
            top_probability_delta=None if previous_top is None else probs[0] - previous_top,
            update_kind="new" if previous is None else "rescore",
            tail_distance_words=(len(words) - 1) - boundary.end_word,
        )

    def append(self, chunk: str, *, step: int) -> ObservedAppend:
        self.text += chunk
        spans = word_char_spans(self.text)
        words = [self.text[start:end] for start, end in spans]
        starts = [start for start, _ in spans]
        ends = [end for _, end in spans]
        boundaries = [SpanBoundary(0, 0)]
        if len(words) > 1:
            boundaries.append(SpanBoundary(0, len(words) - 1))
        updates = tuple(
            self._update(
                boundary=boundary,
                step=step,
                chunk=chunk,
                words=words,
                starts=starts,
                ends=ends,
            )
            for boundary in boundaries
        )
        self.model._session_cache.states[self.session_id] = FakeCacheState(
            text=self.text,
            labels=self.labels,
            scores=self.scores,
        )
        entity = PublicEntity(
            start_char=starts[0],
            end_char=ends[0],
            label=self.labels[0],
            text=words[0],
            score=_sigmoid(self.scores[SpanBoundary(0, 0)][0]),
        )
        snapshot = SnapshotStep(
            run_id=self.run_id,
            example_id=self.example_id,
            step=step,
            chunk=chunk,
            accumulated_text=self.text,
            visible_char_count=len(self.text),
            visible_word_count=len(words),
            elapsed_ms=987.654,
            public_entities=(entity,),
        )
        return ObservedAppend(
            snapshot=snapshot,
            span_updates=updates,
            merged_span_logits=dict(self.scores),
            validated_public_score_count=1,
        )


@pytest.fixture
def fake_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(streaming_parity.observer_module, "StreamingSpanObserver", FakeObserver)


def _fixture_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fictional": True,
        "labels": list(LABELS),
        "cases": [{"id": "person", "text": "Ada Lovelace"}],
    }


def _capture(model: FakeModel, schedules: list[int]) -> dict[str, Any]:
    return streaming_parity.capture_streaming_parity_suite(
        model,
        fixture_document=_fixture_document(),
        model_id="example/pinned-model",
        model_revision=REVISION,
        chunk_units=schedules,
        verify_reference=False,
    )


def test_suite_captures_exact_schedule_coordinates_updates_and_final_state(
    fake_observer: None,
) -> None:
    payload = _capture(FakeModel(), [1])

    assert payload["chunk_unit_schedules"] == [1]
    assert payload["totals"] == {
        "case_condition_count": 1,
        "step_count": 2,
        "span_update_count": 3,
        "final_span_count": 2,
        "validated_public_score_count": 2,
    }
    case = payload["conditions"][0]["cases"][0]
    assert case["chunks"] == ["Ada ", "Lovelace"]
    assert case["steps"][0]["chunk"] == "Ada "
    assert case["steps"][1]["accumulated_text"] == "Ada Lovelace"
    assert case["steps"][1]["model_words"] == ["Ada", "Lovelace"]
    assert case["steps"][1]["word_char_starts"] == [0, 4]
    assert case["steps"][1]["word_char_ends"] == [3, 12]
    assert case["steps"][1]["labels"] == list(LABELS)
    assert case["steps"][1]["updated_boundaries"] == [[0, 0], [0, 1]]
    rescore = case["steps"][1]["span_updates"][0]
    assert rescore["update_kind"] == "rescore"
    assert len(rescore["logits"]) == len(LABELS)
    assert len(rescore["probs"]) == len(LABELS)
    assert case["steps"][1]["public_entities"][0]["text"] == "Ada"
    final = case["final_state"]
    assert final["accumulated_text"] == "Ada Lovelace"
    assert final["model_words"] == ["Ada", "Lovelace"]
    assert final["span_count"] == 2
    assert all(len(row["logits"]) == len(LABELS) for row in final["span_scores"])
    assert all(len(row["probs"]) == len(LABELS) for row in final["span_scores"])


def test_schedules_are_canonical_and_export_is_byte_deterministic(
    fake_observer: None,
    tmp_path: Path,
) -> None:
    first = _capture(FakeModel(), [4, 1, 2, 1])
    second = _capture(FakeModel(), [2, 4, 1])

    assert first == second
    assert first["chunk_unit_schedules"] == [1, 2, 4]
    assert [condition["chunk_units"] for condition in first["conditions"]] == [1, 2, 4]
    encoded = streaming_parity.deterministic_json_bytes(first)
    assert encoded == streaming_parity.deterministic_json_bytes(second)
    assert b"elapsed_ms" not in encoded

    output = tmp_path / "oracle" / "suite.json"
    report_one = streaming_parity.write_streaming_parity_suite(first, output)
    bytes_one = output.read_bytes()
    report_two = streaming_parity.write_streaming_parity_suite(second, output)
    assert output.read_bytes() == bytes_one
    assert report_one == report_two
    assert report_one["size_bytes"] == len(bytes_one)
    assert len(report_one["sha256"]) == 64


def test_single_chunk_mode_captures_exact_text_once_and_is_explicit(
    fake_observer: None,
) -> None:
    result = streaming_parity.capture_streaming_case(
        FakeModel(),
        case_id="one-append",
        text="Ada Lovelace wrote notes",
        labels=LABELS,
        chunk_units=None,
        run_id="unit-single",
        verify_reference=False,
        single_chunk=True,
    )

    assert result["chunk_mode"] == "single"
    assert result["chunk_units"] is None
    assert result["chunks"] == ["Ada Lovelace wrote notes"]
    assert result["step_count"] == 1
    assert result["steps"][0]["chunk"] == "Ada Lovelace wrote notes"


@pytest.mark.parametrize(
    ("chunk_units", "single_chunk"),
    [(None, False), (1, True), (None, 1)],
)
def test_single_chunk_arguments_fail_closed(
    chunk_units: int | None,
    single_chunk: object,
) -> None:
    with pytest.raises((TypeError, streaming_parity.StreamingParityError)):
        streaming_parity.capture_streaming_case(
            FakeModel(),
            case_id="invalid",
            text="Ada",
            labels=LABELS,
            chunk_units=chunk_units,
            run_id="unit-invalid",
            verify_reference=False,
            single_chunk=single_chunk,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("schedules", [[], [3], [0], [True], [1.5]])
def test_invalid_schedules_fail_closed(schedules: list[object]) -> None:
    with pytest.raises((TypeError, streaming_parity.StreamingParityError)):
        streaming_parity.normalize_chunk_units(schedules)  # type: ignore[arg-type]


def test_duplicate_fixture_ids_fail_before_observation(fake_observer: None) -> None:
    fixtures = _fixture_document()
    fixtures["cases"] = [
        {"id": "same", "text": "Ada"},
        {"id": "same", "text": "Jo"},
    ]

    with pytest.raises(streaming_parity.StreamingParityError, match="duplicate fixture ID"):
        streaming_parity.capture_streaming_parity_suite(
            FakeModel(),
            fixture_document=fixtures,
            model_id="example/pinned-model",
            model_revision=REVISION,
            verify_reference=False,
        )


def test_module_reload_never_imports_torch_or_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    attempted: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", maxsplit=1)[0] in {"torch", "gliner"}:
            attempted.append(name)
            raise AssertionError(f"unexpected eager reference import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(streaming_parity)

    assert attempted == []
