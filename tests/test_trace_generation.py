from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from streamner_commit.backends.mlx_streaming import (
    MLXStreamingBackend,
    StreamingAppendResult,
    StreamingSpanUpdate,
    StreamingStateMetadata,
)
from streamner_commit.chunking import chunk_text_by_words, word_char_spans
from streamner_commit.streaming.replay import replay_span_updates
from streamner_commit.streaming.trace_generation import (
    TRACE_CHUNK_WORDS,
    TraceGenerationError,
    TraceInputExample,
    generate_condition_traces,
    generate_example_trace,
    span_state_sha256,
)
from streamner_commit.types import ColdFullResult, GoldEntity, PublicEntity, SpanBoundary


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class FakeTraceSession:
    def __init__(
        self,
        backend: FakeTraceBackend,
        labels: tuple[str, ...],
        session_id: str,
    ) -> None:
        self.backend = backend
        self.labels = labels
        self.session_id = session_id
        self.text = ""
        self.append_calls: list[str] = []
        self.history: dict[tuple[int, int], np.ndarray[Any, Any]] = {}
        self.closed = False

    @property
    def historical_logits(self) -> Mapping[tuple[int, int], np.ndarray[Any, Any]]:
        return MappingProxyType(self.history.copy())

    def append(self, chunk: str) -> StreamingAppendResult:
        if self.closed:
            raise RuntimeError("session already closed")
        self.append_calls.append(chunk)
        self.backend.all_appends.append(chunk)
        if self.backend.fail_on_append == len(self.append_calls):
            raise RuntimeError("injected append failure")

        prior_word_count = len(word_char_spans(self.text))
        self.text += chunk
        coordinates = word_char_spans(self.text)
        updates: list[StreamingSpanUpdate] = []
        generated: list[tuple[int, int]] = [
            (word_index, word_index)
            for word_index in range(prior_word_count, len(coordinates))
        ]
        # Revisit exactly one old boundary. Other historical entries remain cached
        # and intentionally have no event in this append.
        if prior_word_count and (0, 0) not in generated:
            generated.insert(0, (0, 0))
        for start_word, end_word in generated:
            boundary = start_word, end_word
            base = float(len(self.append_calls) + start_word + 1)
            logits = (base, -base)
            probabilities = (_sigmoid(base), _sigmoid(-base))
            kind = "rescore" if boundary in self.history else "new"
            updates.append(
                StreamingSpanUpdate(
                    start_word=start_word,
                    end_word=end_word,
                    logits=logits,
                    probs=probabilities,
                    update_kind=kind,
                )
            )
            self.history[boundary] = np.asarray(logits, dtype=np.float64)

        tokens = tuple(self.text[start:end] for start, end in coordinates)
        starts = tuple(start for start, _end in coordinates)
        ends = tuple(end for _start, end in coordinates)
        state = StreamingStateMetadata(
            session_id=self.session_id,
            accumulated_text=self.text,
            word_tokens=tokens,
            word_char_starts=starts,
            word_char_ends=ends,
            labels=self.labels,
            token_count=len(tokens) + len(self.labels),
            cache_offset=len(tokens) + len(self.labels),
            word_count=len(tokens),
            historical_span_count=len(self.history),
            context_limit=512,
            is_initialized=True,
            is_released=False,
        )
        return StreamingAppendResult(
            public_entities=(),
            span_updates=tuple(updates),
            elapsed_ms=float(len(self.append_calls)),
            state=state,
        )

    def clear(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.backend.active_sessions.pop(self.session_id, None)
        self.backend.cleared_sessions.append(self.session_id)


class FakeTraceBackend:
    def __init__(self, *, fail_on_append: int | None = None, fail_cold: bool = False) -> None:
        self.fail_on_append = fail_on_append
        self.fail_cold = fail_cold
        self.active_sessions: dict[str, FakeTraceSession] = {}
        self.cleared_sessions: list[str] = []
        self.sessions: list[FakeTraceSession] = []
        self.all_appends: list[str] = []
        self.cold_calls: list[tuple[str, str, tuple[str, ...]]] = []

    def start_session(
        self,
        labels: Sequence[str],
        *,
        session_id: str,
        **_kwargs: object,
    ) -> FakeTraceSession:
        assert not self.active_sessions, "prior example leaked a live session"
        session = FakeTraceSession(self, tuple(labels), session_id)
        self.active_sessions[session_id] = session
        self.sessions.append(session)
        return session

    def infer_full_trace(
        self,
        text: str,
        labels: Sequence[str],
        *,
        example_id: str,
        **_kwargs: object,
    ) -> ColdFullResult:
        assert not self.active_sessions, "warm state remained live during cold inference"
        self.cold_calls.append((example_id, text, tuple(labels)))
        if self.fail_cold:
            raise RuntimeError("injected cold failure")
        coordinates = word_char_spans(text)
        raw_state = {
            SpanBoundary(start_word, end_word): tuple(
                float(index + 1) for index in range(len(labels))
            )
            for start_word in range(len(coordinates))
            for end_word in range(start_word, min(start_word + 2, len(coordinates)))
        }
        return ColdFullResult(
            example_id=example_id,
            full_text=text,
            public_entities=(),
            raw_final_span_state=raw_state,
        )


def _example(example_id: str = "example-1", text: str = "Ada,  Bob\nwrites.") -> TraceInputExample:
    start = text.find("Bob")
    gold = (
        (GoldEntity(example_id, start, start + 3, "person", "Bob"),) if start >= 0 else ()
    )
    return TraceInputExample(
        example_id=example_id,
        text=text,
        labels=("person", "email address"),
        gold_entities=gold,
        metadata={"task_name": "synthetic", "nested": {"rank": [1, 2]}},
    )


@pytest.mark.parametrize("chunk_words", TRACE_CHUNK_WORDS)
def test_trace_uses_exact_chunks_authoritative_model_words_and_one_cold_call(
    chunk_words: int,
) -> None:
    backend = FakeTraceBackend()
    example = _example()

    trace = generate_example_trace(
        backend,  # type: ignore[arg-type]
        example,
        run_id="run-1",
        chunk_words=chunk_words,
    )

    expected_chunks = tuple(chunk_text_by_words(example.text, chunk_words))
    assert tuple(snapshot.chunk for snapshot in trace.snapshots) == expected_chunks
    assert "".join(backend.all_appends) == example.text
    assert trace.snapshots[-1].accumulated_text == example.text
    assert trace.snapshots[-1].visible_word_count == len(word_char_spans(example.text))
    assert trace.gold_entities == example.gold_entities
    assert example.text[
        trace.gold_entities[0].start_char : trace.gold_entities[0].end_char
    ] == trace.gold_entities[0].text
    assert backend.cold_calls == [(example.example_id, example.text, example.labels)]
    assert not backend.active_sessions
    assert backend.sessions[0].closed

    replayed = replay_span_updates(trace.span_updates)
    assert trace.final_state_sha256 == span_state_sha256(replayed)
    assert trace.cold_full.full_text == example.text
    assert len(trace.cold_full.raw_final_span_state) > len(replayed)


def test_updates_derive_rank_margin_delta_tail_and_omit_cached_candidates() -> None:
    backend = FakeTraceBackend()
    trace = generate_example_trace(
        backend,  # type: ignore[arg-type]
        _example(text="Ada, Bob writes."),
        run_id="run-1",
        chunk_words=1,
    )

    rescored = [update for update in trace.span_updates if update.update_kind == "rescore"]
    assert rescored
    first_rescore = rescored[0]
    assert first_rescore.previous_top_probability is not None
    assert first_rescore.top_probability_delta == pytest.approx(
        first_rescore.top_probability - first_rescore.previous_top_probability
    )
    assert first_rescore.label_margin == pytest.approx(
        first_rescore.top_probability - first_rescore.second_probability
    )
    assert first_rescore.top_label == trace.labels[first_rescore.top_label_index]
    assert first_rescore.tail_distance_words == (
        first_rescore.visible_word_count - 1 - first_rescore.end_word
    )
    assert first_rescore.span_text == trace.text[
        first_rescore.start_char : first_rescore.end_char
    ]

    final_boundaries = set(replay_span_updates(trace.span_updates))
    last_step_boundaries = {
        update.boundary for update in trace.span_updates if update.step == trace.snapshots[-1].step
    }
    assert final_boundaries - last_step_boundaries, "cached spans should not be emitted as updates"


def test_empty_public_snapshots_are_retained_and_do_not_trigger_model_reruns() -> None:
    backend = FakeTraceBackend()
    example = _example(text="One two three four")
    trace = generate_example_trace(
        backend,  # type: ignore[arg-type]
        example,
        run_id="run-1",
        chunk_words=2,
    )

    assert len(trace.snapshots) == 2
    assert all(not snapshot.public_entities for snapshot in trace.snapshots)
    assert len(backend.sessions) == 1
    assert len(backend.all_appends) == 2
    assert len(backend.cold_calls) == 1


def test_append_and_cold_failures_never_leave_warm_state_live() -> None:
    failing_append = FakeTraceBackend(fail_on_append=2)
    with pytest.raises(RuntimeError, match="injected append"):
        generate_example_trace(
            failing_append,  # type: ignore[arg-type]
            _example(text="One two three"),
            run_id="run-1",
            chunk_words=1,
        )
    assert not failing_append.active_sessions
    assert failing_append.sessions[0].closed
    assert not failing_append.cold_calls

    failing_cold = FakeTraceBackend(fail_cold=True)
    with pytest.raises(RuntimeError, match="injected cold"):
        generate_example_trace(
            failing_cold,  # type: ignore[arg-type]
            _example(text="One two"),
            run_id="run-1",
            chunk_words=1,
        )
    assert not failing_cold.active_sessions
    assert failing_cold.sessions[0].closed


def test_condition_runner_rejects_duplicates_and_does_not_leak_between_examples() -> None:
    backend = FakeTraceBackend()
    traces = generate_condition_traces(
        backend,  # type: ignore[arg-type]
        (_example("one", "One two"), _example("two", "Three four")),
        run_id="run-1",
        chunk_words=1,
    )
    assert [trace.example_id for trace in traces] == ["one", "two"]
    assert len(backend.cleared_sessions) == 2
    assert len(backend.cold_calls) == 2

    duplicate_backend = FakeTraceBackend()
    with pytest.raises(ValueError, match="duplicate example_id"):
        generate_condition_traces(
            duplicate_backend,  # type: ignore[arg-type]
            (_example("same", "One"), _example("same", "Two")),
            run_id="run-2",
            chunk_words=1,
        )
    assert not duplicate_backend.sessions


def test_example_preserves_gold_and_freezes_json_metadata() -> None:
    metadata = {"task_name": "pii", "nested": {"values": [1, 2]}}
    example = TraceInputExample(
        example_id="example-1",
        text="Ada",
        labels=("person", "person"),
        gold_entities=(GoldEntity("example-1", 0, 3, "person", "Ada"),),
        metadata=metadata,
    )
    metadata["task_name"] = "changed"
    assert example.labels == ("person",)
    assert example.metadata["task_name"] == "pii"
    assert example.to_dict()["metadata"] == {
        "task_name": "pii",
        "nested": {"values": [1, 2]},
    }

    with pytest.raises(ValueError, match="source slice"):
        TraceInputExample(
            example_id="example-1",
            text="Ada",
            labels=("person",),
            gold_entities=(GoldEntity("example-1", 0, 3, "person", "Bob"),),
        )


@pytest.mark.parametrize("chunk_words", [0, 3, 16, True])
def test_only_locked_chunk_conditions_are_accepted(chunk_words: object) -> None:
    with pytest.raises((TypeError, ValueError), match="chunk_words"):
        generate_example_trace(
            FakeTraceBackend(),  # type: ignore[arg-type]
            _example(),
            run_id="run-1",
            chunk_words=chunk_words,  # type: ignore[arg-type]
        )


class RawColdModel:
    def __init__(self) -> None:
        self.preprocessor = object()
        self.qwen = SimpleNamespace(
            configuration=SimpleNamespace(max_position_embeddings=64),
        )
        self.prompt_projection = object()
        self.marker = object()
        self.decoder = object()
        self.forward_preprocessed = object()
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        **_kwargs: object,
    ) -> SimpleNamespace:
        ordered = tuple(labels)
        self.calls.append((text, ordered))
        prepared = SimpleNamespace(
            text=text,
            labels=ordered,
            word_tokens=("Ada", "Jo"),
            span_idx=np.asarray([[[0, 0], [0, 1], [1, 1], [0, 0]]], dtype=np.int64),
            span_mask=np.asarray([[True, True, True, False]], dtype=np.bool_),
        )
        forward = SimpleNamespace(
            preprocessing=prepared,
            logits=np.asarray([[[[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [9.0, 9.0]]]]),
        )
        return SimpleNamespace(
            forward=forward,
            entities=(PublicEntity(0, 3, "person", "Ada", 0.9),),
        )


def test_backend_cold_trace_materializes_valid_raw_map_without_session_or_sentinel() -> None:
    model = RawColdModel()
    backend = MLXStreamingBackend(model)

    result = backend.infer_full_trace(
        "Ada Jo",
        ("person", "email", "person"),
        example_id="example-1",
    )

    assert model.calls == [("Ada Jo", ("person", "email"))]
    assert result.full_text == "Ada Jo"
    assert result.public_entities == (PublicEntity(0, 3, "person", "Ada", 0.9),)
    assert result.raw_final_span_state == {
        SpanBoundary(0, 0): (1.0, -1.0),
        SpanBoundary(0, 1): (2.0, -2.0),
        SpanBoundary(1, 1): (3.0, -3.0),
    }
    assert backend._sessions == {}


def test_runner_rejects_replay_divergence_and_still_clears_session() -> None:
    backend = FakeTraceBackend()
    original_start = backend.start_session

    def corrupting_start(*args: object, **kwargs: object) -> FakeTraceSession:
        session = original_start(*args, **kwargs)  # type: ignore[arg-type]
        original_append = session.append

        def corrupting_append(chunk: str) -> StreamingAppendResult:
            result = original_append(chunk)
            if len(session.append_calls) == 2:
                session.history[(0, 0)] = np.asarray((99.0, -99.0))
            return result

        session.append = corrupting_append  # type: ignore[method-assign]
        return session

    backend.start_session = corrupting_start  # type: ignore[method-assign]
    with pytest.raises(TraceGenerationError, match="replay differs"):
        generate_example_trace(
            backend,  # type: ignore[arg-type]
            _example(text="One two"),
            run_id="run-1",
            chunk_words=1,
        )
    assert not backend.active_sessions
    assert backend.sessions[0].closed
