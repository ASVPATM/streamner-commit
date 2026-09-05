import json

import pytest

from streamner_commit.reference import format_public_trace, run_public_trace


class IncrementingClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        value = self.current
        self.current += 0.001
        return value


class FakeStreamingModel:
    def __init__(self) -> None:
        self.session_text: dict[str, str] = {}
        self.events: list[tuple[str, object]] = []

    def inference(self, texts: str | list[str], labels: list[str], **kwargs: object) -> object:
        assert isinstance(texts, list)
        text = texts[0]
        session_ids = kwargs.get("session_id")
        if session_ids is None:
            self.events.append(("cold", text))
            visible = text
        else:
            assert isinstance(session_ids, list)
            session_id = session_ids[0]
            self.events.append(("append", text))
            visible = self.session_text.get(session_id, "") + text
            self.session_text[session_id] = visible

        entities: list[dict[str, object]] = []
        if visible.startswith("Ada"):
            end = 13 if visible.startswith("Ada  Lovelace") else 3
            entities.append(
                {
                    "start": 0,
                    "end": end,
                    "text": visible[:end],
                    "label": labels[0],
                    "score": 0.875,
                }
            )
        return [entities]

    def clear_session(self, session_id: str | list[str]) -> None:
        assert isinstance(session_id, str)
        self.events.append(("clear", session_id))
        self.session_text.pop(session_id, None)


def test_public_trace_records_exact_snapshots_then_clears_before_cold_run() -> None:
    model = FakeStreamingModel()
    trace = run_public_trace(
        model,
        text="Ada  Lovelace\nwrites.",
        labels=["person"],
        example_id="multi-space",
        run_id="run-1",
        session_id="session-1",
        words_per_chunk=1,
        clock=IncrementingClock(),
    )

    assert [snapshot.chunk for snapshot in trace.snapshots] == [
        "Ada  ",
        "Lovelace\n",
        "writes.",
    ]
    assert [snapshot.accumulated_text for snapshot in trace.snapshots] == [
        "Ada  ",
        "Ada  Lovelace\n",
        "Ada  Lovelace\nwrites.",
    ]
    assert [snapshot.elapsed_ms for snapshot in trace.snapshots] == pytest.approx([1.0] * 3)
    assert [event[0] for event in model.events] == ["append", "append", "append", "clear", "cold"]
    assert model.session_text == {}
    assert trace.cold_full.full_text == "Ada  Lovelace\nwrites."
    assert trace.cold_elapsed_ms == pytest.approx(1.0)
    json.dumps(trace.to_dict())


def test_public_trace_accepts_public_entity_objects() -> None:
    from streamner_commit.types import PublicEntity

    class EntityModel(FakeStreamingModel):
        def inference(self, texts: str | list[str], labels: list[str], **kwargs: object) -> object:
            assert isinstance(texts, list)
            return [[PublicEntity(0, 3, labels[0], "Ada", 0.9)]]

    trace = run_public_trace(
        EntityModel(),
        text="Ada",
        labels=["person"],
        example_id="entity-object",
        run_id="run-1",
        session_id="session-1",
    )
    assert trace.snapshots[0].public_entities[0].text == "Ada"


def test_invalid_public_offsets_fail_and_still_clear_session() -> None:
    class InvalidOffsetModel(FakeStreamingModel):
        def inference(self, texts: str | list[str], labels: list[str], **kwargs: object) -> object:
            self.events.append(("append", texts))
            return [[{"start": 0, "end": 4, "text": "Ada!", "label": labels[0], "score": 0.9}]]

    model = InvalidOffsetModel()
    with pytest.raises(ValueError, match="beyond accumulated text"):
        run_public_trace(
            model,
            text="Ada",
            labels=["person"],
            example_id="bad-offset",
            run_id="run-1",
            session_id="session-1",
        )
    assert model.events[-1] == ("clear", "session-1")
    assert not any(event[0] == "cold" for event in model.events)


def test_model_failure_still_clears_session() -> None:
    class FailingModel(FakeStreamingModel):
        def inference(self, texts: str | list[str], labels: list[str], **kwargs: object) -> object:
            raise RuntimeError("model failed")

    model = FailingModel()
    with pytest.raises(RuntimeError, match="model failed"):
        run_public_trace(
            model,
            text="Ada",
            labels=["person"],
            example_id="failure",
            run_id="run-1",
            session_id="session-1",
        )
    assert model.events == [("clear", "session-1")]


def test_whitespace_only_text_skips_session_append_but_runs_cold() -> None:
    model = FakeStreamingModel()
    trace = run_public_trace(
        model,
        text=" \t\n",
        labels=["person"],
        example_id="blank",
        run_id="run-1",
        session_id="session-1",
    )
    assert trace.snapshots == ()
    assert model.events == [("clear", "session-1"), ("cold", " \t\n")]


def test_trace_validates_labels_and_single_row_model_output() -> None:
    model = FakeStreamingModel()
    with pytest.raises(ValueError, match="unique"):
        run_public_trace(
            model,
            text="Ada",
            labels=["person", "person"],
            example_id="duplicate-labels",
        )

    class TwoRowModel(FakeStreamingModel):
        def inference(self, texts: str | list[str], labels: list[str], **kwargs: object) -> object:
            return [[], []]

    with pytest.raises(ValueError, match="exactly one batch row"):
        run_public_trace(
            TwoRowModel(),
            text="Ada",
            labels=["person"],
            example_id="bad-batch",
            run_id="run-1",
            session_id="session-1",
        )


def test_plain_text_formatter_contains_steps_offsets_and_cold_result() -> None:
    trace = run_public_trace(
        FakeStreamingModel(),
        text="Ada",
        labels=["person"],
        example_id="format",
        run_id="run-1",
        session_id="session-1",
        clock=IncrementingClock(),
    )
    output = format_public_trace(trace)
    assert "[step 01]" in output
    assert 'text: "Ada"' in output
    assert "person" in output
    assert "[0:3]" in output
    assert "[cold full]" in output
