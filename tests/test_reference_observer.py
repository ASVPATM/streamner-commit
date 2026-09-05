import math

import numpy as np
import pytest

from streamner_commit.reference.observer import (
    ObserverInvariantError,
    StreamingSpanObserver,
    observer_record_json,
)
from streamner_commit.types import SpanBoundary


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)
        self.detach_calls = 0
        self.cpu_calls = 0

    def detach(self) -> "FakeTensor":
        self.detach_calls += 1
        return self

    def cpu(self) -> "FakeTensor":
        self.cpu_calls += 1
        return self

    def tolist(self) -> object:
        return self.values.tolist()


class FakeHookHandle:
    def __init__(self, hooks: list[object], hook: object) -> None:
        self.hooks = hooks
        self.hook = hook
        self.removed = False

    def remove(self) -> None:
        self.removed = True
        self.hooks.remove(self.hook)


class FakeStreamingSpanModel:
    def __init__(self) -> None:
        self.hooks: list[object] = []
        self.handles: list[FakeHookHandle] = []

    def register_forward_hook(self, hook: object) -> FakeHookHandle:
        self.hooks.append(hook)
        handle = FakeHookHandle(self.hooks, hook)
        self.handles.append(handle)
        return handle

    def emit(self, output: object, *, times: int = 1) -> None:
        for _ in range(times):
            for hook in tuple(self.hooks):
                hook(self, (), output)  # type: ignore[operator]


class FakeCache:
    def __init__(self) -> None:
        self.states: dict[str, object] = {}

    def get(self, session_id: str) -> object | None:
        return self.states.get(session_id)


class FakeState:
    def __init__(
        self,
        *,
        text: str,
        tokens: list[str],
        char_starts: list[int],
        char_ends: list[int],
        span_logits: dict[tuple[int, int], object],
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.char_starts = char_starts
        self.char_ends = char_ends
        self.span_logits = span_logits
        self.labels = ("person", "email address")


class FakeOutput:
    def __init__(self, logits: object, span_idx: object, span_mask: object) -> None:
        self.logits = FakeTensor(logits)
        self.span_idx = FakeTensor(span_idx)
        self.span_mask = FakeTensor(span_mask)

    @property
    def words_embedding(self) -> object:
        raise AssertionError("observer must not access embeddings")


class FakeWrapper:
    def __init__(self) -> None:
        self.model = FakeStreamingSpanModel()
        self._session_cache = FakeCache()
        self.full_text = ""
        self.call = 0
        self.clear_calls: list[str] = []
        self.hook_times = 1
        self.public_score_override: float | None = None
        self.outputs: list[FakeOutput] = []

    def inference(self, texts: list[str], labels: list[str], **kwargs: object) -> object:
        session_ids = kwargs.get("session_id")
        assert isinstance(session_ids, list)
        session_id = session_ids[0]
        self.call += 1
        self.full_text += texts[0]
        if self.call == 1:
            output = FakeOutput(
                logits=[[[[2.0, -2.0]]]],
                span_idx=[[[0, 0]]],
                span_mask=[[1]],
            )
            state = FakeState(
                text=self.full_text,
                tokens=["Ada"],
                char_starts=[0],
                char_ends=[3],
                span_logits={(0, 0): FakeTensor([2.0, -2.0])},
            )
            end = 3
            score = 1.0 / (1.0 + math.exp(-2.0))
        else:
            output = FakeOutput(
                logits=[[[1.0, -1.0], [3.0, -3.0], [9.0, 9.0]]],
                span_idx=[[[0, 0], [0, 1], [1, 1]]],
                span_mask=[[1, 1, 0]],
            )
            state = FakeState(
                text=self.full_text,
                tokens=["Ada", "Lovelace"],
                char_starts=[0, 4],
                char_ends=[3, 12],
                span_logits={
                    (0, 0): FakeTensor([1.0, -1.0]),
                    (0, 1): FakeTensor([3.0, -3.0]),
                },
            )
            end = 12
            score = 1.0 / (1.0 + math.exp(-3.0))
        self.outputs.append(output)
        self.model.emit(output, times=self.hook_times)
        self._session_cache.states[session_id] = state
        return [
            [
                {
                    "start": 0,
                    "end": end,
                    "text": self.full_text[:end],
                    "label": labels[0],
                    "score": self.public_score_override or score,
                }
            ]
        ]

    def clear_session(self, session_id: str) -> None:
        self.clear_calls.append(session_id)
        self._session_cache.states.pop(session_id, None)


def make_observer(model: FakeWrapper) -> StreamingSpanObserver:
    return StreamingSpanObserver(
        model,
        run_id="run-1",
        example_id="example-1",
        session_id="session-1",
        labels=["person", "email address"],
        verify_reference=False,
        clock=lambda: 1.0,
    )


def test_observer_emits_actual_new_and_rescore_updates_and_copies_state() -> None:
    model = FakeWrapper()
    observer = make_observer(model)

    first = observer.append("Ada", step=1)
    second = observer.append(" Lovelace", step=2)

    assert [update.update_kind for update in first.span_updates] == ["new"]
    assert [update.boundary for update in second.span_updates] == [
        SpanBoundary(0, 0),
        SpanBoundary(0, 1),
    ]
    assert [update.update_kind for update in second.span_updates] == ["rescore", "new"]
    assert second.span_updates[0].previous_top_probability == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0))
    )
    assert second.span_updates[0].tail_distance_words == 1
    assert second.span_updates[1].span_text == "Ada Lovelace"
    assert second.span_updates[1].logits == (3.0, -3.0)
    assert second.validated_public_score_count == 1
    assert second.snapshot.accumulated_text == "Ada Lovelace"
    assert second.snapshot.visible_word_count == 2
    assert len(second.merged_span_logits) == 2
    assert "merged_span_logits" in observer_record_json(second)

    # Returned state is a host-value copy, not a live view of CacheState tensors.
    cached = model._session_cache.get("session-1")
    assert cached is not None
    cached.span_logits[(0, 1)].values[:] = 99.0  # type: ignore[attr-defined]
    assert second.merged_span_logits[SpanBoundary(0, 1)] == (3.0, -3.0)

    observer.close()
    assert model.clear_calls == ["session-1"]


def test_hooked_tensors_detach_and_copy_to_cpu_and_handles_are_removed() -> None:
    model = FakeWrapper()
    observer = make_observer(model)
    observer.append("Ada", step=1)

    output = model.outputs[0]
    for tensor in (output.logits, output.span_idx, output.span_mask):
        assert tensor.detach_calls == 1
        assert tensor.cpu_calls == 1
    assert model.model.hooks == []
    assert model.model.handles[0].removed


def test_masked_forward_candidates_do_not_emit_updates() -> None:
    model = FakeWrapper()
    observer = make_observer(model)
    observer.append("Ada", step=1)
    second = observer.append(" Lovelace", step=2)
    assert SpanBoundary(1, 1) not in {update.boundary for update in second.span_updates}


def test_public_score_mismatch_fails_and_clears_session() -> None:
    model = FakeWrapper()
    model.public_score_override = 0.5
    observer = make_observer(model)
    with pytest.raises(ObserverInvariantError, match="does not match sigmoid"):
        observer.append("Ada", step=1)
    assert model.clear_calls == ["session-1"]
    with pytest.raises(RuntimeError, match="closed"):
        observer.append(" again", step=2)


@pytest.mark.parametrize("hook_times", [0, 2])
def test_unexpected_forward_hook_count_fails_without_choosing_a_capture(hook_times: int) -> None:
    model = FakeWrapper()
    model.hook_times = hook_times
    observer = make_observer(model)
    with pytest.raises(ObserverInvariantError, match="exactly one"):
        observer.append("Ada", step=1)
    assert model.model.hooks == []
    assert model.clear_calls == ["session-1"]


def test_observer_rejects_existing_session_and_nonincreasing_steps() -> None:
    model = FakeWrapper()
    model._session_cache.states["occupied"] = object()
    with pytest.raises(ValueError, match="new, empty"):
        StreamingSpanObserver(
            model,
            run_id="run-1",
            example_id="example-1",
            session_id="occupied",
            labels=["person"],
            verify_reference=False,
        )

    observer = make_observer(model)
    observer.append("Ada", step=1)
    with pytest.raises(ValueError, match="increase"):
        observer.append(" again", step=1)
