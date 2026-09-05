from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from streamner_commit.mlx.streaming_validation import (
    StreamingValidationError,
    deterministic_json_bytes,
    run_streaming_validation,
    write_streaming_validation_report,
)

LABELS = ("person", "email")
REVISION = "1" * 40


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _reference_update(
    boundary: tuple[int, int],
    logits: tuple[float, ...],
    *,
    previous_top: float | None = None,
) -> dict[str, Any]:
    probs = tuple(_sigmoid(value) for value in logits)
    top = max(range(len(logits)), key=logits.__getitem__)
    second = max((index for index in range(len(logits)) if index != top), key=probs.__getitem__)
    top_probability = probs[top]
    return {
        "start_word": boundary[0],
        "end_word": boundary[1],
        "logits": list(logits),
        "probs": list(probs),
        "top_label_index": top,
        "top_label": LABELS[top],
        "top_probability": top_probability,
        "second_probability": probs[second],
        "label_margin": top_probability - probs[second],
        "previous_top_probability": previous_top,
        "top_probability_delta": (None if previous_top is None else top_probability - previous_top),
        "update_kind": "new" if previous_top is None else "rescore",
    }


@dataclass(frozen=True)
class FakeEntity:
    start_char: int
    end_char: int
    label: str
    text: str
    score: float


@dataclass(frozen=True)
class FakeUpdate:
    start_word: int
    end_word: int
    logits: tuple[float, ...]
    probs: tuple[float, ...]
    update_kind: str

    @property
    def boundary(self) -> tuple[int, int]:
        return self.start_word, self.end_word


@dataclass(frozen=True)
class FakeState:
    accumulated_text: str
    word_tokens: tuple[str, ...]
    word_char_starts: tuple[int, ...]
    word_char_ends: tuple[int, ...]
    labels: tuple[str, ...]
    word_count: int
    historical_span_count: int


@dataclass(frozen=True)
class FakeResult:
    public_entities: tuple[FakeEntity, ...]
    span_updates: tuple[FakeUpdate, ...]
    state: FakeState
    elapsed_ms: float


class FakeSession:
    def __init__(self, plan: tuple[tuple[str, FakeResult], ...]) -> None:
        self._plan = plan
        self._index = 0
        self.cleared = False

    def append(self, chunk: str) -> FakeResult:
        expected_chunk, result = self._plan[self._index]
        assert chunk == expected_chunk
        self._index += 1
        return result

    def clear(self) -> None:
        self.cleared = True


class FakeBackend:
    def __init__(self, plan: tuple[tuple[str, FakeResult], ...]) -> None:
        self.plan = plan
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.sessions: list[FakeSession] = []

    def start_session(self, labels: list[str], **kwargs: Any) -> FakeSession:
        self.calls.append((tuple(labels), kwargs))
        session = FakeSession(self.plan)
        self.sessions.append(session)
        return session


def _candidate_update(reference: dict[str, Any]) -> FakeUpdate:
    return FakeUpdate(
        start_word=reference["start_word"],
        end_word=reference["end_word"],
        logits=tuple(reference["logits"]),
        probs=tuple(reference["probs"]),
        update_kind=reference["update_kind"],
    )


def _entity(score: float, *, label: str = "person", text: str = "Ada") -> dict[str, Any]:
    return {"start_char": 0, "end_char": len(text), "label": label, "text": text, "score": score}


def _suite(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "synthetic_streaming_reference_parity_suite",
        "model_id": "test/model",
        "model_revision_sha": REVISION,
        "threshold": 0.5,
        "labels": list(LABELS),
        "conditions": [{"chunk_units": 1, "case_count": 1, "cases": [case]}],
    }


def _two_step_fixture() -> tuple[dict[str, Any], tuple[tuple[str, FakeResult], ...]]:
    first = _reference_update((0, 0), (1.0, -1.0))
    first_probs = tuple(first["probs"])
    rescored = _reference_update((0, 0), (1.1, -1.0), previous_top=max(first_probs))
    email = _reference_update((1, 1), (-1.0, 1.5))
    ada = _entity(first_probs[0])
    final_ada = _entity(rescored["probs"][0])
    final_email = {
        "start_char": 4,
        "end_char": 11,
        "label": "email",
        "text": "emailed",
        "score": email["probs"][1],
    }
    steps = [
        {
            "step": 1,
            "chunk": "Ada ",
            "accumulated_text": "Ada ",
            "visible_char_count": 4,
            "visible_word_count": 1,
            "model_words": ["Ada"],
            "word_char_starts": [0],
            "word_char_ends": [3],
            "labels": list(LABELS),
            "updated_boundaries": [[0, 0]],
            "span_updates": [first],
            "public_entities": [ada],
        },
        {
            "step": 2,
            "chunk": "emailed Jo.",
            "accumulated_text": "Ada emailed Jo.",
            "visible_char_count": 15,
            "visible_word_count": 4,
            "model_words": ["Ada", "emailed", "Jo", "."],
            "word_char_starts": [0, 4, 12, 14],
            "word_char_ends": [3, 11, 14, 15],
            "labels": list(LABELS),
            "updated_boundaries": [[0, 0], [1, 1]],
            "span_updates": [rescored, email],
            "public_entities": [final_ada, final_email],
        },
    ]
    case = {
        "id": "two-step",
        "full_text": "Ada emailed Jo.",
        "labels": list(LABELS),
        "chunk_units": 1,
        "chunks": [step["chunk"] for step in steps],
        "step_count": 2,
        "span_update_count": 3,
        "steps": steps,
        "final_state": {
            "accumulated_text": "Ada emailed Jo.",
            "model_words": ["Ada", "emailed", "Jo", "."],
            "word_char_starts": [0, 4, 12, 14],
            "word_char_ends": [3, 11, 14, 15],
            "labels": list(LABELS),
            "span_count": 2,
            "span_scores": [rescored, email],
            "public_entities": [final_ada, final_email],
        },
    }
    plan = (
        (
            "Ada ",
            FakeResult(
                public_entities=(FakeEntity(**ada),),
                span_updates=(_candidate_update(first),),
                state=FakeState("Ada ", ("Ada",), (0,), (3,), LABELS, 1, 1),
                elapsed_ms=123.0,
            ),
        ),
        (
            "emailed Jo.",
            FakeResult(
                public_entities=(FakeEntity(**final_ada), FakeEntity(**final_email)),
                span_updates=(_candidate_update(rescored), _candidate_update(email)),
                state=FakeState(
                    "Ada emailed Jo.",
                    ("Ada", "emailed", "Jo", "."),
                    (0, 4, 12, 14),
                    (3, 11, 14, 15),
                    LABELS,
                    4,
                    2,
                ),
                elapsed_ms=456.0,
            ),
        ),
    )
    return _suite(case), plan


def _single_step_fixture(
    reference_logits: tuple[float, float],
    candidate_logits: tuple[float, float],
    *,
    reference_entity: bool,
    candidate_entity: bool,
) -> tuple[dict[str, Any], tuple[tuple[str, FakeResult], ...]]:
    update = _reference_update((0, 0), reference_logits)
    candidate_probs = tuple(_sigmoid(value) for value in candidate_logits)
    candidate = FakeUpdate(0, 0, candidate_logits, candidate_probs, "new")
    reference_top = max(range(2), key=reference_logits.__getitem__)
    candidate_top = max(range(2), key=candidate_logits.__getitem__)
    reference_entities = (
        [_entity(update["probs"][reference_top], label=LABELS[reference_top])]
        if reference_entity
        else []
    )
    candidate_entities = (
        (
            FakeEntity(
                0,
                3,
                LABELS[candidate_top],
                "Ada",
                candidate_probs[candidate_top],
            ),
        )
        if candidate_entity
        else ()
    )
    step = {
        "step": 1,
        "chunk": "Ada",
        "accumulated_text": "Ada",
        "visible_char_count": 3,
        "visible_word_count": 1,
        "model_words": ["Ada"],
        "word_char_starts": [0],
        "word_char_ends": [3],
        "labels": list(LABELS),
        "updated_boundaries": [[0, 0]],
        "span_updates": [update],
        "public_entities": reference_entities,
    }
    case = {
        "id": "one-step",
        "full_text": "Ada",
        "labels": list(LABELS),
        "chunk_units": 1,
        "chunks": ["Ada"],
        "step_count": 1,
        "span_update_count": 1,
        "steps": [step],
        "final_state": {
            "accumulated_text": "Ada",
            "model_words": ["Ada"],
            "word_char_starts": [0],
            "word_char_ends": [3],
            "labels": list(LABELS),
            "span_count": 1,
            "span_scores": [update],
            "public_entities": reference_entities,
        },
    }
    result = FakeResult(
        candidate_entities,
        (candidate,),
        FakeState("Ada", ("Ada",), (0,), (3,), LABELS, 1, 1),
        999.0,
    )
    return _suite(case), (("Ada", result),)


def test_exact_streaming_report_covers_rescore_delta_and_is_deterministic(
    tmp_path: Path,
) -> None:
    oracle, plan = _two_step_fixture()
    backend = FakeBackend(plan)

    report = run_streaming_validation(backend, oracle)

    assert report["pass"] is True
    assert report["totals"] == {
        "condition_count": 1,
        "case_count": 1,
        "step_count": 2,
        "exact_step_count": 2,
        "material_reason_count": 0,
        "near_top_tie_count": 0,
        "near_threshold_count": 0,
    }
    assert report["numerical"]["vector_count"] == 3
    assert report["numerical"]["delta_maximum_absolute_error"] == 0.0
    assert report["numerical"]["top_label_agreement"] == 1.0
    assert backend.calls == [
        (
            LABELS,
            {"threshold": 0.5, "flat_ner": True, "multi_label": False},
        )
    ]
    assert backend.sessions[0].cleared is True
    data = deterministic_json_bytes(report)
    assert b"elapsed_ms" not in data
    artifact = write_streaming_validation_report(report, tmp_path / "report.json")
    assert (tmp_path / "report.json").read_bytes() == data
    assert artifact["sha256"] == hashlib.sha256(data).hexdigest()


def test_exact_state_mismatch_is_material() -> None:
    oracle, plan = _two_step_fixture()
    chunk, last = plan[-1]
    bad_state = replace(last.state, word_char_starts=(0, 5, 12, 14))
    backend = FakeBackend((*plan[:-1], (chunk, replace(last, state=bad_state))))

    report = run_streaming_validation(backend, oracle)

    assert report["pass"] is False
    case = report["conditions"][0]["cases"][0]
    assert "exact:word_char_starts" in case["material_reasons"]
    assert "final_exact:word_char_starts" in case["material_reasons"]


def test_negligible_top_tie_is_categorized_without_hiding_it() -> None:
    oracle, plan = _single_step_fixture(
        (-100.0, -100.05),
        (-100.05, -100.0),
        reference_entity=False,
        candidate_entity=False,
    )

    report = run_streaming_validation(FakeBackend(plan), oracle)

    assert report["pass"] is True
    assert report["totals"]["near_top_tie_count"] == 1
    disagreement = report["conditions"][0]["cases"][0]["categorized_disagreements"][0]
    assert disagreement["category"] == "near_top_tie"
    assert report["numerical"]["top_label_agreement"] == 0.0


def test_near_threshold_identity_difference_is_categorized() -> None:
    reference_probability = 0.5002
    candidate_probability = 0.4998
    oracle, plan = _single_step_fixture(
        (math.log(reference_probability / (1.0 - reference_probability)), -10.0),
        (math.log(candidate_probability / (1.0 - candidate_probability)), -10.0),
        reference_entity=True,
        candidate_entity=False,
    )

    report = run_streaming_validation(FakeBackend(plan), oracle)

    assert report["pass"] is True
    assert report["totals"]["near_threshold_count"] == 2  # append snapshot and final
    case = report["conditions"][0]["cases"][0]
    assert case["steps"][0]["decoded"]["identity_match"] is False
    assert case["final"]["decoded"]["identity_match"] is False


def test_material_decoded_difference_fails_and_missing_schedule_is_rejected() -> None:
    oracle, plan = _single_step_fixture(
        (1.0, -1.0),
        (1.0, -1.0),
        reference_entity=True,
        candidate_entity=False,
    )
    report = run_streaming_validation(FakeBackend(plan), oracle)
    assert report["pass"] is False
    assert "decoded_identity" in report["conditions"][0]["cases"][0]["material_reasons"]

    with pytest.raises(StreamingValidationError, match="lacks requested chunk schedules"):
        run_streaming_validation(FakeBackend(plan), oracle, chunk_units=(2,))
