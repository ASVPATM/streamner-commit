"""Complete native MLX cold inference for the locked StreamingSpan checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import mlx.core as mx

from streamner_commit.mlx.assets import AssetBundle
from streamner_commit.mlx.context_encoder import (
    DenseDebertaV2Encoder,
    LabelEncoderConfig,
    load_label_encoder_weights,
)
from streamner_commit.mlx.decoder import DecodedSpan, SpanDecoder
from streamner_commit.mlx.labels_encoder import LabelEncoderOutput, encode_labels
from streamner_commit.mlx.preprocessing import (
    ColdPreprocessingResult,
    ColdPreprocessor,
)
from streamner_commit.mlx.qwen_adapter import QwenAdapter
from streamner_commit.mlx.span_rep import MarkerV2, ProjectionMLP, score_spans
from streamner_commit.mlx.weights import load_component_into
from streamner_commit.mlx.word_pooling import pool_first_subtokens
from streamner_commit.types import PublicEntity


class ColdModelError(ValueError):
    """The exported configuration cannot support the locked cold inference path."""


@dataclass(frozen=True, slots=True)
class ColdForwardOutput:
    """Materialized component boundaries for parity and decoding."""

    preprocessing: ColdPreprocessingResult
    qwen_hidden_states: mx.array
    labels: LabelEncoderOutput
    pooled_word_states: mx.array
    pooled_word_mask: mx.array
    projected_label_states: mx.array
    span_states: mx.array
    logits: mx.array


@dataclass(frozen=True, slots=True)
class ColdPrediction:
    """One cold model output in both word and public character coordinates."""

    forward: ColdForwardOutput
    decoded_spans: tuple[DecodedSpan, ...]
    entities: tuple[PublicEntity, ...]


class MLXColdModel:
    """Own all 371 inference tensors and execute an exact one-shot cold path."""

    def __init__(
        self,
        *,
        preprocessor: ColdPreprocessor,
        qwen: QwenAdapter,
        label_encoder: DenseDebertaV2Encoder,
        prompt_projection: ProjectionMLP,
        marker: MarkerV2,
        decoder: SpanDecoder | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.qwen = qwen
        self.label_encoder = label_encoder
        self.prompt_projection = prompt_projection
        self.marker = marker
        self.decoder = decoder or SpanDecoder()

    @classmethod
    def from_asset_bundle(cls, bundle: AssetBundle) -> Self:
        """Validate configuration and strict-load every inference tensor exactly once."""

        config = bundle.config
        hidden_size = config.get("hidden_size")
        max_width = config.get("max_width")
        dropout = config.get("dropout")
        if hidden_size != 1024 or max_width != 12 or dropout != 0.3:
            raise ColdModelError(
                "locked cold model requires hidden_size=1024, max_width=12, dropout=0.3"
            )
        label_config_value = config.get("labels_encoder_config")
        if not isinstance(label_config_value, dict):
            raise ColdModelError("labels_encoder_config must be an object")

        preprocessor = ColdPreprocessor.from_asset_bundle(bundle)
        qwen = QwenAdapter.from_asset_bundle(bundle)
        label_encoder = DenseDebertaV2Encoder(LabelEncoderConfig.from_dict(label_config_value))
        load_label_encoder_weights(label_encoder, bundle)
        label_encoder.eval()
        label_encoder.freeze()

        prompt_projection = ProjectionMLP(hidden_size, hidden_size, float(dropout))
        load_component_into(prompt_projection, bundle, "prompt_projection")
        prompt_projection.eval()
        prompt_projection.freeze()

        marker = MarkerV2(hidden_size, max_width, float(dropout))
        load_component_into(marker, bundle, "marker_v2")
        marker.eval()
        marker.freeze()
        return cls(
            preprocessor=preprocessor,
            qwen=qwen,
            label_encoder=label_encoder,
            prompt_projection=prompt_projection,
            marker=marker,
        )

    def forward(self, text: str, labels: Sequence[str]) -> ColdForwardOutput:
        """Execute the full cold graph and retain explicit parity boundaries."""

        prepared = self.preprocessor.preprocess(text, labels)
        return self.forward_preprocessed(prepared)

    def forward_preprocessed(
        self,
        prepared: ColdPreprocessingResult,
        *,
        qwen_hidden_states: mx.array | None = None,
    ) -> ColdForwardOutput:
        """Run the post-tokenization cold graph, optionally using cached-Qwen output.

        Supplying hidden states is used when a streaming session initializes its native
        KV cache. The same exact downstream graph is shared with stateless cold inference.
        """

        if not isinstance(prepared, ColdPreprocessingResult):
            raise TypeError("prepared must be a ColdPreprocessingResult")
        input_ids = mx.array(prepared.input_ids, dtype=mx.int32)
        attention_mask = mx.array(prepared.attention_mask, dtype=mx.int32)
        label_attention_mask = mx.array(
            prepared.label_attention_mask,
            dtype=mx.int32,
        )
        words_mask = mx.array(prepared.words_mask, dtype=mx.int32)
        text_lengths = mx.array(prepared.text_lengths, dtype=mx.int32)
        span_idx = mx.array(prepared.span_idx, dtype=mx.int32)
        span_mask = mx.array(prepared.span_mask, dtype=mx.bool_)

        qwen_hidden = (
            self.qwen.cold_hidden_states(input_ids)
            if qwen_hidden_states is None
            else qwen_hidden_states
        )
        expected_hidden_shape = (*input_ids.shape, 1024)
        if qwen_hidden.shape != expected_hidden_shape:
            raise ColdModelError(
                f"Qwen hidden states must have shape {expected_hidden_shape}, "
                f"got {qwen_hidden.shape}"
            )
        pooled_words, pooled_mask = pool_first_subtokens(
            qwen_hidden,
            words_mask,
            attention_mask,
            text_lengths,
        )
        label_output = encode_labels(
            self.label_encoder,
            qwen_hidden,
            input_ids,
            label_attention_mask,
            separator_token_id=prepared.separator_token_id,
            label_token_id=prepared.label_token_id,
        )
        safe_span_idx = span_idx * span_mask[..., None]
        span_states = self.marker(pooled_words, safe_span_idx, pooled_mask)
        projected_labels = self.prompt_projection(label_output.label_representations)
        logits = score_spans(span_states, projected_labels)
        mx.eval(
            qwen_hidden,
            label_output.prompt.hidden_states,
            label_output.contextualized_prompt,
            label_output.label_representations,
            pooled_words,
            pooled_mask,
            span_states,
            projected_labels,
            logits,
        )
        return ColdForwardOutput(
            preprocessing=prepared,
            qwen_hidden_states=qwen_hidden,
            labels=label_output,
            pooled_word_states=pooled_words,
            pooled_word_mask=pooled_mask,
            projected_label_states=projected_labels,
            span_states=span_states,
            logits=logits,
        )

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> ColdPrediction:
        """Run cold inference and return reference-compatible public entities."""

        output = self.forward(text, labels)
        prepared = output.preprocessing
        decoded = self.decoder.decode(
            output.logits,
            prepared.span_idx,
            prepared.span_mask,
            prepared.labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )[0]
        entities = self.decoder.to_public_entities(
            decoded,
            text,
            prepared.word_char_starts,
            prepared.word_char_ends,
        )
        return ColdPrediction(output, decoded, entities)

    def infer_full(
        self,
        text: str,
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
    ) -> list[dict[str, Any]]:
        """Satisfy the backend cold-call shape with JSON-safe entity dictionaries."""

        if return_class_probs:
            raise NotImplementedError("class-probability dictionaries are not needed by the study")
        prediction = self.predict(
            text,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        return [entity.to_dict() for entity in prediction.entities]


__all__ = [
    "ColdForwardOutput",
    "ColdModelError",
    "ColdPrediction",
    "MLXColdModel",
]
