from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from streamner_commit.backends.mlx_streaming import (
    ClearedSessionError,
    InvalidatedSessionError,
    MLXStreamingBackend,
)
from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.decoder import SpanDecoder
from streamner_commit.mlx.preprocessing import ColdPreprocessor, PreprocessingConfig


class FakeEncoding(dict[str, np.ndarray[Any, Any]]):
    def __init__(self, token_ids: list[int], word_ids: list[int]) -> None:
        super().__init__(
            input_ids=np.asarray([token_ids], dtype=np.int64),
            attention_mask=np.ones((1, len(token_ids)), dtype=np.int64),
        )
        self._word_ids = tuple(word_ids)

    def word_ids(self, batch_index: int = 0) -> list[int]:
        if batch_index != 0:
            raise IndexError(batch_index)
        return list(self._word_ids)


class RecordingTokenizer:
    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.ids = {
            "person": 1,
            "email": 2,
            "address": 3,
            "Ada": 4,
            "em": 5,
            "ailed": 6,
            "Jo": 7,
            ".": 8,
            "X": 9,
            "<<LABEL>>": 90,
            "<<SEP>>": 91,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, 99)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        reverse = {value: key for key, value in self.ids.items()}
        return [reverse.get(token_id, "[UNK]") for token_id in token_ids]

    def __call__(
        self,
        rows: list[list[str]],
        *,
        is_split_into_words: bool,
        return_tensors: str,
        truncation: bool,
        padding: str,
        add_special_tokens: bool,
    ) -> FakeEncoding:
        assert is_split_into_words and return_tensors == "np"
        assert not truncation and padding == "longest" and not add_special_tokens
        words = tuple(rows[0])
        self.calls.append(words)
        token_ids: list[int] = []
        word_ids: list[int] = []
        for word_index, word in enumerate(words):
            pieces: tuple[str, ...]
            if word == "email address":
                pieces = ("email", "address")
            elif word == "emailed":
                pieces = ("em", "ailed")
            else:
                pieces = (word,)
            token_ids.extend(self.convert_tokens_to_ids(piece) for piece in pieces)
            word_ids.extend([word_index] * len(pieces))
        return FakeEncoding(token_ids, word_ids)


@dataclass
class FakeConfiguration:
    hidden_size: int = 4
    max_position_embeddings: int = 64


@dataclass
class FakeCache:
    offset: int = 0
    keys: object | None = None
    values: object | None = None


class FakeQwen:
    def __init__(self) -> None:
        self.configuration = FakeConfiguration()
        self.calls: list[np.ndarray[Any, Any]] = []
        self.offsets_before: list[int] = []
        self.caches: list[list[FakeCache]] = []
        self.fail_after_mutation = False

    def create_cache(self) -> list[FakeCache]:
        cache = [FakeCache(), FakeCache()]
        self.caches.append(cache)
        return cache

    def validate_cache(
        self,
        cache: list[FakeCache],
        *,
        expected_offset: int | None = None,
    ) -> int:
        if len(cache) != 2:
            raise ValueError("expected two native layers")
        offsets = {layer.offset for layer in cache}
        if len(offsets) != 1:
            raise ValueError("cache offsets differ")
        offset = offsets.pop()
        if expected_offset is not None and offset != expected_offset:
            raise ValueError(f"expected {expected_offset}, got {offset}")
        return offset

    def cached_hidden_states(self, input_ids: object, cache: list[FakeCache]) -> Any:
        import mlx.core as mx

        identifiers = np.asarray(input_ids, dtype=np.int32)
        self.calls.append(identifiers.copy())
        start = self.validate_cache(cache)
        self.offsets_before.append(start)
        for layer in cache:
            layer.offset += identifiers.shape[1]
            layer.keys = object()
            layer.values = object()
        if self.fail_after_mutation:
            raise RuntimeError("synthetic post-mutation failure")
        values = mx.array(identifiers, dtype=mx.float32)[..., None]
        return mx.broadcast_to(values, (*identifiers.shape, self.configuration.hidden_size))


class IdentityProjection:
    def __init__(self) -> None:
        self.inputs: list[Any] = []

    def __call__(self, value: Any) -> Any:
        self.inputs.append(value)
        return value


class FakeMarker:
    def __init__(self, max_width: int) -> None:
        self.max_width = max_width
        self.word_capacities: list[int] = []

    def __call__(self, words: Any, span_idx: Any, _mask: Any) -> Any:
        import mlx.core as mx

        self.word_capacities.append(int(words.shape[1]))
        start_count = int(span_idx.shape[1]) // self.max_width
        base = mx.ones((1, start_count, self.max_width, int(words.shape[-1])))
        boundary_signal = span_idx[..., 1].reshape(1, start_count, self.max_width, 1)
        return base + boundary_signal.astype(base.dtype) * 0.01


class FakeColdModel:
    def __init__(self) -> None:
        self.tokenizer = RecordingTokenizer()
        self.preprocessor = ColdPreprocessor(
            PreprocessingConfig(
                max_width=3,
                label_token="<<LABEL>>",
                label_token_id=90,
                separator_token="<<SEP>>",
                separator_token_id=91,
            ),
            self.tokenizer,
        )
        self.qwen = FakeQwen()
        self.prompt_projection = IdentityProjection()
        self.marker = FakeMarker(max_width=3)
        self.decoder = SpanDecoder()
        self.forward_calls = 0

    def forward_preprocessed(
        self,
        prepared: Any,
        *,
        qwen_hidden_states: Any,
    ) -> Any:
        import mlx.core as mx

        from streamner_commit.mlx.span_rep import score_spans
        from streamner_commit.mlx.word_pooling import pool_first_subtokens

        self.forward_calls += 1
        pooled, pooled_mask = pool_first_subtokens(
            qwen_hidden_states,
            mx.array(prepared.words_mask, dtype=mx.int32),
            mx.array(prepared.attention_mask, dtype=mx.int32),
            mx.array(prepared.text_lengths, dtype=mx.int32),
        )
        safe_spans = (
            mx.array(prepared.span_idx, dtype=mx.int32)
            * mx.array(prepared.span_mask, dtype=mx.bool_)[..., None]
        )
        span_states = self.marker(pooled, safe_spans, pooled_mask)
        label_count = len(prepared.labels)
        labels = mx.stack(
            [mx.full((4,), float(index + 1)) for index in range(label_count)],
            axis=0,
        )[None, ...]
        label_mask = mx.ones((1, label_count), dtype=mx.bool_)
        projected = self.prompt_projection(labels)
        logits = score_spans(span_states, projected)
        # Deliberately retain spare capacity. Warm append must compact this prefix
        # before concatenation so candidate coordinates remain absolute word indices.
        padded_words = mx.concatenate([pooled, mx.zeros((1, 2, 4))], axis=1)
        padded_mask = mx.concatenate([pooled_mask, mx.zeros((1, 2), dtype=mx.bool_)], axis=1)
        mx.eval(padded_words, padded_mask, labels, logits)
        return SimpleNamespace(
            pooled_word_states=padded_words,
            pooled_word_mask=padded_mask,
            labels=SimpleNamespace(label_representations=labels, label_mask=label_mask),
            logits=logits,
        )

    def infer_full(self, text: str, labels: Sequence[str], **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"text": text, "labels": list(labels)}]


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _backend(
    *,
    context_limit: int = 64,
) -> tuple[MLXStreamingBackend, FakeColdModel]:
    model = FakeColdModel()
    return (
        MLXStreamingBackend(
            model,
            context_limit=context_limit,
            right_context_width=12,
            clock=StepClock(),
        ),
        model,
    )


def test_cold_then_warm_uses_native_cache_current_ids_and_complete_history() -> None:
    backend, model = _backend()
    session = backend.start_session(["person", "email address", "person"])

    cold = session.append("Ada ")
    warm = session.append("emailed Jo.")

    assert cold.state.labels == ("person", "email address")
    assert cold.state.accumulated_text == "Ada "
    assert cold.state.word_tokens == ("Ada",)
    assert cold.state.token_count == model.qwen.calls[0].shape[1]
    assert cold.state.cache_offset == cold.state.token_count
    assert all(update.update_kind == "new" for update in cold.span_updates)
    assert cold.public_snapshot == cold.public_entities
    assert cold.raw_updated_span_scores == cold.span_updates

    # The second Qwen call contains only current no-prompt IDs; the native offset
    # carries position/RoPE history.
    np.testing.assert_array_equal(model.qwen.calls[1], [[5, 6, 7, 8]])
    assert model.qwen.offsets_before[1] == cold.state.token_count
    assert warm.state.accumulated_text == "Ada emailed Jo."
    assert warm.state.word_tokens == ("Ada", "emailed", "Jo", ".")
    assert warm.state.word_char_starts == (0, 4, 12, 14)
    assert warm.state.word_char_ends == (3, 11, 14, 15)
    assert warm.state.token_count == cold.state.token_count + 4
    assert warm.state.cache_offset == warm.state.token_count
    assert warm.state.historical_span_count >= cold.state.historical_span_count
    assert any(update.update_kind == "rescore" for update in warm.span_updates)
    assert all(len(update.logits) == 2 and len(update.probs) == 2 for update in warm.span_updates)
    assert warm.public_entities
    assert len(session.historical_logits) == warm.state.historical_span_count

    # Cold cached two spare word slots; warm marker still sees exactly 1 + 3 words.
    assert model.marker.word_capacities[-1] == 4
    # The same pre-projection prompt object is reused and only projected on demand.
    assert model.prompt_projection.inputs[0] is model.prompt_projection.inputs[1]


def test_blank_is_empty_public_noop_before_and_after_initialization() -> None:
    backend, model = _backend()
    session = backend.start_session(["person"])

    initial = session.append(" \t\n")
    assert initial.is_noop
    assert initial.public_entities == ()
    assert initial.span_updates == ()
    assert not initial.state.is_initialized
    assert model.qwen.calls == []

    cold = session.append("Ada")
    calls = len(model.qwen.calls)
    blank = session.append("")
    assert blank.is_noop
    assert blank.public_entities == ()
    assert blank.span_updates == ()
    assert blank.state == cold.state
    assert len(model.qwen.calls) == calls


def test_context_preflight_failure_does_not_mutate_or_release_session() -> None:
    backend, model = _backend(context_limit=4)
    session = backend.start_session(["person"])
    cold = session.append("Ada")
    assert cold.state.token_count == 4
    cache = model.qwen.caches[0]
    before_offsets = [layer.offset for layer in cache]
    before_calls = len(model.qwen.calls)

    with pytest.raises(ValueError, match="context limit"):
        session.append(" X")

    assert [layer.offset for layer in cache] == before_offsets
    assert len(model.qwen.calls) == before_calls
    assert not session.is_cleared
    assert session.state_metadata == cold.state


def test_post_mutation_error_releases_state_and_invalidates_session() -> None:
    backend, model = _backend()
    session = backend.start_session(["person"])
    session.append("Ada")
    cache_layers = tuple(model.qwen.caches[0])
    model.qwen.fail_after_mutation = True

    with pytest.raises(RuntimeError, match="post-mutation"):
        session.append(" X")

    assert session.is_cleared
    assert session.state_metadata.is_released
    assert all(
        layer.offset == 0 and layer.keys is None and layer.values is None for layer in cache_layers
    )
    with pytest.raises(InvalidatedSessionError, match="invalidated"):
        session.append("again")


def test_clear_is_idempotent_and_backend_owns_distinct_sessions() -> None:
    backend, _model = _backend()
    first = backend.start_session(["person"], session_id="one")
    second = backend.start_session(["person"], session_id="two")
    with pytest.raises(ValueError, match="already exists"):
        backend.start_session(["person"], session_id="one")

    first.append("Ada")
    first.clear()
    first.clear()
    with pytest.raises(ClearedSessionError, match="cleared"):
        first.append(" X")
    second.append("Ada")
    backend.clear_sessions()
    assert second.is_cleared


def test_import_boundary_does_not_load_reference_frameworks() -> None:
    assert "torch" not in sys.modules
    assert "gliner" not in sys.modules


@pytest.mark.skipif(
    os.environ.get("STREAMNER_RUN_MLX_STREAMING_SMOKE") != "1",
    reason="set STREAMNER_RUN_MLX_STREAMING_SMOKE=1 for the real 2.7 GB checkpoint smoke",
)
def test_real_checkpoint_cold_and_warm_smoke() -> None:
    root = Path("artifacts/reference") / REFERENCE_REVISION
    if not (root / "export_manifest.json").is_file():
        pytest.skip("ignored reference export is unavailable")
    bundle = load_asset_bundle(
        root,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    backend = MLXStreamingBackend.from_asset_bundle(bundle)
    with backend.start_session(["person", "email address"]) as session:
        cold = session.append("Ada ")
        warm = session.append("emailed Jo.")
        assert cold.state.word_tokens == ("Ada",)
        assert warm.state.word_tokens == ("Ada", "emailed", "Jo", ".")
        assert warm.state.cache_offset == warm.state.token_count
        assert warm.span_updates
