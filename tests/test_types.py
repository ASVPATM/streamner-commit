import json
from dataclasses import FrozenInstanceError

import pytest

from streamner_commit.types import (
    ColdFullResult,
    GoldEntity,
    PublicEntity,
    SnapshotStep,
    SpanBoundary,
    SpanScoreUpdate,
)


def make_update(**overrides: object) -> SpanScoreUpdate:
    values: dict[str, object] = {
        "run_id": "run-1",
        "example_id": "example-1",
        "step": 2,
        "chunk": " Lovelace",
        "visible_char_count": 12,
        "visible_word_count": 2,
        "start_word": 0,
        "end_word": 1,
        "start_char": 0,
        "end_char": 12,
        "span_text": "Ada Lovelace",
        "logits": [1.0, -1.0],
        "probs": [0.75, 0.25],
        "top_label_index": 0,
        "top_label": "person",
        "top_probability": 0.75,
        "second_probability": 0.25,
        "label_margin": 0.5,
        "previous_top_probability": 0.65,
        "top_probability_delta": 0.10,
        "update_kind": "rescore",
        "tail_distance_words": 0,
    }
    values.update(overrides)
    return SpanScoreUpdate(**values)  # type: ignore[arg-type]


def test_entity_records_are_immutable_and_validate_offsets() -> None:
    gold = GoldEntity("example-1", 4, 12, "person", "Lovelace")
    public = PublicEntity(4, 12, "person", "Lovelace", 0.9)

    assert gold.to_dict()["end_char"] == 12
    assert public.score == 0.9
    with pytest.raises(FrozenInstanceError):
        public.score = 0.8  # type: ignore[misc]
    with pytest.raises(ValueError, match="span length"):
        GoldEntity("example-1", 4, 11, "person", "Lovelace")
    with pytest.raises(ValueError, match="between 0 and 1"):
        PublicEntity(0, 3, "person", "Ada", 1.1)


def test_span_boundary_uses_inclusive_ordered_word_offsets() -> None:
    assert SpanBoundary(2, 2).to_tuple() == (2, 2)
    assert SpanBoundary(1, 2) < SpanBoundary(2, 2)
    with pytest.raises(ValueError, match="greater than or equal"):
        SpanBoundary(2, 1)


def test_span_score_update_normalizes_vectors_and_is_json_safe() -> None:
    update = make_update()

    assert update.boundary == SpanBoundary(0, 1)
    assert update.logits == (1.0, -1.0)
    assert update.probs == (0.75, 0.25)
    encoded = json.dumps(update.to_dict())
    assert json.loads(encoded)["logits"] == [1.0, -1.0]
    with pytest.raises(FrozenInstanceError):
        update.step = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"update_kind": "cached"}, "update_kind"),
        ({"end_word": 2}, "visible word"),
        ({"tail_distance_words": 1}, "tail_distance_words"),
        ({"top_label_index": 1}, "top_probability must equal"),
        ({"second_probability": 0.1}, "second_probability"),
        ({"label_margin": 0.4}, "label_margin"),
        ({"previous_top_probability": None}, "both be set or both be null"),
        ({"top_probability_delta": 0.2}, "top_probability_delta must equal"),
        ({"probs": [0.75]}, "equal lengths"),
    ],
)
def test_span_score_update_rejects_inconsistent_derived_fields(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_update(**overrides)


def test_first_score_observation_can_have_no_previous_probability() -> None:
    update = make_update(
        previous_top_probability=None,
        top_probability_delta=None,
        update_kind="new",
    )
    assert update.previous_top_probability is None
    assert update.top_probability_delta is None


def test_snapshot_validates_prefix_counts_and_nested_entities() -> None:
    snapshot = SnapshotStep(
        run_id="run-1",
        example_id="example-1",
        step=2,
        chunk=" Lovelace",
        accumulated_text="Ada Lovelace",
        visible_char_count=12,
        visible_word_count=2,
        elapsed_ms=1.25,
        public_entities=[PublicEntity(0, 12, "person", "Ada Lovelace", 0.9)],
    )

    assert isinstance(snapshot.public_entities, tuple)
    assert json.loads(json.dumps(snapshot.to_dict()))["public_entities"][0]["label"] == "person"
    with pytest.raises(ValueError, match="visible_char_count"):
        SnapshotStep(
            run_id="run-1",
            example_id="example-1",
            step=2,
            chunk=" Lovelace",
            accumulated_text="Ada Lovelace",
            visible_char_count=11,
            visible_word_count=2,
            elapsed_ms=1.25,
            public_entities=[],
        )


def test_cold_full_result_freezes_and_serializes_tuple_keyed_raw_state() -> None:
    source_state = {(1, 1): [0.1, 0.2], SpanBoundary(0, 0): [0.3, 0.4]}
    result = ColdFullResult(
        example_id="example-1",
        full_text="Ada Lovelace",
        public_entities=[PublicEntity(0, 12, "person", "Ada Lovelace", 0.9)],
        raw_final_span_state=source_state,
    )
    source_state[(2, 2)] = [9.0, 9.0]

    assert tuple(result.raw_final_span_state) == (SpanBoundary(1, 1), SpanBoundary(0, 0))
    assert SpanBoundary(2, 2) not in result.raw_final_span_state
    with pytest.raises(TypeError):
        result.raw_final_span_state[SpanBoundary(2, 2)] = (1.0, 1.0)  # type: ignore[index]
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["raw_final_span_state"] == [
        {"end_word": 0, "logits": [0.3, 0.4], "start_word": 0},
        {"end_word": 1, "logits": [0.1, 0.2], "start_word": 1},
    ]


def test_cold_full_result_rejects_mixed_score_vector_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        ColdFullResult(
            example_id="example-1",
            full_text="Ada Lovelace",
            public_entities=[],
            raw_final_span_state={(0, 0): [0.1], (0, 1): [0.2, 0.3]},
        )
