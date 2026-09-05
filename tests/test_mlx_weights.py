from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from streamner_commit.mlx.weights import MLXWeightError, strict_load_weights


def test_strict_weight_loader_accepts_exact_inventory() -> None:
    layer = nn.Linear(2, 1)
    strict_load_weights(
        layer,
        {
            "weight": mx.array([[2.0, 3.0]]),
            "bias": mx.array([4.0]),
        },
    )
    output = layer(mx.array([[1.0, 1.0]]))
    mx.eval(output)
    assert output.item() == 9.0


def test_strict_weight_loader_rejects_missing_or_extra_names() -> None:
    layer = nn.Linear(2, 1)
    with pytest.raises(MLXWeightError, match="missing"):
        strict_load_weights(layer, {"weight": mx.zeros((1, 2))})
    with pytest.raises(MLXWeightError, match="extra"):
        strict_load_weights(
            layer,
            {
                "weight": mx.zeros((1, 2)),
                "bias": mx.zeros((1,)),
                "unexpected": mx.zeros((1,)),
            },
        )
