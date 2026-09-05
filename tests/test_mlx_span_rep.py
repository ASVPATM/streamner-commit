from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from streamner_commit.mlx.assets import (
    REFERENCE_MODEL_ID,
    REFERENCE_REVISION,
    load_asset_bundle,
)
from streamner_commit.mlx.qwen_adapter import numerical_diagnostics
from streamner_commit.mlx.span_rep import (
    MarkerV2,
    ProjectionMLP,
    SpanRepresentationError,
    score_spans,
)
from streamner_commit.mlx.weights import load_component_into, module_parameter_names
from streamner_commit.mlx.word_pooling import pool_first_subtokens

ASSET_ROOT = Path("artifacts/reference") / REFERENCE_REVISION


def test_marker_parameter_names_match_canonical_weight_map() -> None:
    marker = MarkerV2(hidden_size=2, max_width=2)
    expected = {
        f"{projection}.layers.{layer}.{parameter}"
        for projection in (
            "project_start",
            "project_end",
            "project_context",
            "out_project",
        )
        for layer in (0, 3)
        for parameter in ("weight", "bias")
    }
    assert module_parameter_names(marker) == expected


def test_marker_shape_and_score_operation() -> None:
    marker = MarkerV2(hidden_size=2, max_width=2, dropout=0.0)
    marker.eval()
    words = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    spans = mx.array([[[0, 0], [0, 1], [1, 1], [1, 2]]])
    span_states = marker(words, spans, mx.array([[True, True, True]]))
    labels = mx.array([[[0.5, -0.5], [2.0, 1.0]]])
    logits = score_spans(span_states, labels)
    mx.eval(span_states, logits)

    assert span_states.shape == (1, 2, 2, 2)
    assert logits.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(
        np.asarray(logits),
        np.einsum("blkd,bcd->blkc", np.asarray(span_states), np.asarray(labels)),
        # Metal's fused einsum uses a slightly different reduction order.
        rtol=2e-3,
        atol=1e-3,
    )


def test_marker_rejects_candidate_count_not_divisible_by_max_width() -> None:
    marker = MarkerV2(hidden_size=2, max_width=2)
    with pytest.raises(SpanRepresentationError, match="divisible"):
        marker(mx.zeros((1, 2, 2)), mx.zeros((1, 3, 2)))


@pytest.mark.skipif(not ASSET_ROOT.is_dir(), reason="ignored reference assets not exported")
def test_real_marker_projection_and_logit_parity() -> None:
    bundle = load_asset_bundle(
        ASSET_ROOT,
        expected_model_id=REFERENCE_MODEL_ID,
        expected_revision=REFERENCE_REVISION,
        strict_reference=True,
    )
    marker = MarkerV2()
    marker.eval()
    load_component_into(marker, bundle, "marker_v2")
    projection = ProjectionMLP(1024, 1024, 0.3)
    projection.eval()
    load_component_into(projection, bundle, "prompt_projection")

    with np.load(ASSET_ROOT / "parity" / "parity_arrays.npz", allow_pickle=False) as fixture:
        pooled, pooled_mask = pool_first_subtokens(
            mx.array(fixture["qwen_final_hidden_states"]),
            mx.array(fixture["words_mask"]),
            mx.array(fixture["attention_mask"]),
            mx.array(fixture["text_lengths"]),
        )
        safe_spans = mx.array(fixture["span_idx"]) * mx.array(fixture["span_mask"])[..., None]
        spans = marker(
            mx.array(fixture["contextualized_word_states"]),
            safe_spans,
            mx.array(fixture["contextualized_word_mask"]),
        )
        labels = projection(mx.array(fixture["label_representations_pre_projection"]))
        logits = score_spans(spans, labels)
        mx.eval(pooled, pooled_mask, spans, labels, logits)

        np.testing.assert_array_equal(pooled, fixture["pooled_word_states"])
        np.testing.assert_array_equal(pooled_mask, fixture["pooled_word_mask"])
        span_diagnostics = numerical_diagnostics(
            fixture["marker_v2_span_representations"], np.asarray(spans)
        )
        label_diagnostics = numerical_diagnostics(
            fixture["label_representations_post_projection"], np.asarray(labels)
        )
        logit_diagnostics = numerical_diagnostics(fixture["raw_logits"], np.asarray(logits))

    assert span_diagnostics.cosine_similarity is not None
    assert span_diagnostics.cosine_similarity > 0.999999
    assert span_diagnostics.max_absolute_error is not None
    assert span_diagnostics.max_absolute_error < 0.05
    assert label_diagnostics.cosine_similarity is not None
    assert label_diagnostics.cosine_similarity > 0.999999
    assert label_diagnostics.max_absolute_error is not None
    assert label_diagnostics.max_absolute_error < 0.01
    assert logit_diagnostics.cosine_similarity is not None
    assert logit_diagnostics.cosine_similarity > 0.99999
    assert logit_diagnostics.mean_absolute_error is not None
    assert logit_diagnostics.mean_absolute_error < 0.2
    assert logit_diagnostics.max_absolute_error is not None
    assert logit_diagnostics.max_absolute_error < 8.0
