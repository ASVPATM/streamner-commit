from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_LICENSE,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SPLIT,
    PIIMB_SUBSET,
)
from streamner_commit.mlx.assets import REFERENCE_MODEL_ID, REFERENCE_REVISION
from streamner_commit.mlx.piimb_validation import (
    PIIMBValidationError,
    deterministic_piimb_json_bytes,
    run_piimb_validation,
    validate_sanitized_piimb_report,
    write_piimb_validation_report,
)

REVISION = "1" * 40
PIIMB_REPORT = (
    Path(__file__).resolve().parents[1] / "results" / "parity" / "piimb_smoke_report.json"
)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _reference_update(
    labels: tuple[str, ...],
    logits: tuple[float, ...],
    *,
    probs: tuple[float, ...] | None = None,
) -> dict[str, object]:
    probabilities = probs or tuple(_sigmoid(value) for value in logits)
    top = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "start_word": 0,
        "end_word": 0,
        "logits": list(logits),
        "probs": list(probabilities),
        "top_label_index": top,
        "top_label": labels[top],
        "top_probability": probabilities[top],
        "second_probability": sorted(probabilities, reverse=True)[1] if len(labels) > 1 else 0.0,
        "label_margin": (
            probabilities[top] - sorted(probabilities, reverse=True)[1]
            if len(labels) > 1
            else probabilities[top]
        ),
        "previous_top_probability": None,
        "top_probability_delta": None,
        "update_kind": "new",
    }


@dataclass(frozen=True)
class FakeUpdate:
    start_word: int
    end_word: int
    logits: tuple[float, ...]
    probs: tuple[float, ...]
    update_kind: str = "new"

    @property
    def boundary(self) -> tuple[int, int]:
        return self.start_word, self.end_word


@dataclass(frozen=True)
class FakeEntity:
    start_char: int
    end_char: int
    label: str
    text: str
    score: float


@dataclass(frozen=True)
class FakeState:
    accumulated_text: str
    word_tokens: tuple[str, ...]
    word_char_starts: tuple[int, ...]
    word_char_ends: tuple[int, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class FakeResult:
    state: FakeState
    span_updates: tuple[FakeUpdate, ...]
    public_entities: tuple[FakeEntity, ...]


class FakeSession:
    def __init__(self, chunk: str, result: FakeResult) -> None:
        self.chunk = chunk
        self.result = result
        self.cleared = False

    def append(self, chunk: str) -> FakeResult:
        assert chunk == self.chunk
        return self.result

    def clear(self) -> None:
        self.cleared = True


class FakeBackend:
    def __init__(self, plans: dict[tuple[str, ...], list[tuple[str, FakeResult]]]) -> None:
        self.plans = plans
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
        self.sessions: list[FakeSession] = []

    def start_session(self, labels: list[str], **kwargs: object) -> FakeSession:
        key = tuple(labels)
        self.calls.append((key, kwargs))
        chunk, result = self.plans[key].pop(0)
        session = FakeSession(chunk, result)
        self.sessions.append(session)
        return session


def _metadata(selection: dict[str, object]) -> str:
    fields = {
        name: selection[name]
        for name in (
            "uid",
            "source_row_index",
            "task_name",
            "source_dataset",
            "source_uid",
            "parent_id",
            "sentence_index",
            "language",
        )
    }
    encoded = json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _case(
    index: int,
    task: str,
    labels: tuple[str, ...],
    reference: dict[str, object],
    candidate: FakeUpdate,
    *,
    split: str,
    reference_entities: tuple[FakeEntity, ...] = (),
    candidate_entities: tuple[FakeEntity, ...] = (),
    candidate_words: tuple[str, ...] = ("Ada",),
) -> tuple[dict[str, object], tuple[str, FakeResult]]:
    selection: dict[str, object] = {
        "selection_index": index,
        "benchmark_split": split,
        "uid": f"uid-{index}",
        "source_row_index": 100 + index,
        "task_name": task,
        "source_dataset": "synthetic-source",
        "source_uid": f"source-{index}",
        "parent_id": f"parent-{index}",
        "sentence_index": 0,
        "language": "en",
    }
    selection["metadata_sha256"] = _metadata(selection)
    entities = [entity.__dict__ for entity in reference_entities]
    step = {
        "step": 1,
        "chunk": "Ada",
        "accumulated_text": "Ada",
        "model_words": ["Ada"],
        "word_char_starts": [0],
        "word_char_ends": [3],
        "labels": list(labels),
        "updated_boundaries": [[0, 0]],
        "span_updates": [reference],
        "public_entities": entities,
    }
    trace = {
        "labels": list(labels),
        "chunks": ["Ada"],
        "steps": [step],
        "final_state": {
            "accumulated_text": "Ada",
            "model_words": ["Ada"],
            "word_char_starts": [0],
            "word_char_ends": [3],
            "labels": list(labels),
            "span_scores": [reference],
            "public_entities": entities,
        },
    }
    result = FakeResult(
        state=FakeState("Ada", candidate_words, (0,), (3,), labels),
        span_updates=(candidate,),
        public_entities=candidate_entities,
    )
    return {"selection": selection, "trace": trace}, ("Ada", result)


def _oracle_and_backend(
    specifications: list[
        tuple[
            str,
            tuple[str, ...],
            dict[str, object],
            FakeUpdate,
            tuple[FakeEntity, ...],
            tuple[FakeEntity, ...],
            tuple[str, ...],
        ]
    ],
) -> tuple[dict[str, object], FakeBackend, tuple[str, ...]]:
    tasks = tuple(specification[0] for specification in specifications)
    task_labels = {specification[0]: list(specification[1]) for specification in specifications}
    cases: list[dict[str, object]] = []
    plans: dict[tuple[str, ...], list[tuple[str, FakeResult]]] = {}
    split_counts = {"dev": 0, "test": 0}
    for index, (task, labels, reference, candidate, ref_entities, mlx_entities, words) in enumerate(
        specifications
    ):
        split = "dev" if index == 0 else "test"
        split_counts[split] += 1
        case, plan = _case(
            index,
            task,
            labels,
            reference,
            candidate,
            split=split,
            reference_entities=ref_entities,
            candidate_entities=mlx_entities,
            candidate_words=words,
        )
        cases.append(case)
        plans.setdefault(labels, []).append(plan)
    oracle = {
        "schema_version": 1,
        "kind": "piimb_reference_parity_smoke",
        "backend": "reference",
        "model_id": "test/model",
        "model_revision_sha": REVISION,
        "gliner_version": "0.2.28",
        "device": "cpu",
        "dtype": "float32",
        "dataset": {
            "id": PIIMB_DATASET_ID,
            "subset": PIIMB_SUBSET,
            "revision": PIIMB_REVISION,
            "split": PIIMB_SPLIT,
            "license": PIIMB_LICENSE,
            "source_file": {
                "path": PIIMB_SOURCE_FILE,
                "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
                "sha256": PIIMB_SOURCE_SHA256,
                "rows": PIIMB_SOURCE_ROW_COUNT,
            },
        },
        "threshold": 0.5,
        "capture": {"chunk_mode": "single", "appends_per_case": 1},
        "selection": {
            "preset": "smoke",
            "manifest_sha256": "a" * 64,
            "task_labels_sha256": "b" * 64,
            "case_count": len(cases),
            "split_counts": split_counts,
            "task_counts": {task: 1 for task in tasks},
        },
        "task_labels": task_labels,
        "totals": {"case_count": len(cases)},
        "cases": cases,
    }
    return oracle, FakeBackend(plans), tasks


def _specification(
    task: str,
    labels: tuple[str, ...],
    reference_logits: tuple[float, ...],
    candidate_logits: tuple[float, ...] | None = None,
    *,
    reference_probs: tuple[float, ...] | None = None,
    candidate_probs: tuple[float, ...] | None = None,
    reference_entities: tuple[FakeEntity, ...] = (),
    candidate_entities: tuple[FakeEntity, ...] = (),
    words: tuple[str, ...] = ("Ada",),
) -> tuple[
    str,
    tuple[str, ...],
    dict[str, object],
    FakeUpdate,
    tuple[FakeEntity, ...],
    tuple[FakeEntity, ...],
    tuple[str, ...],
]:
    candidate_values = candidate_logits or reference_logits
    candidate_probabilities = candidate_probs or tuple(
        _sigmoid(value) for value in candidate_values
    )
    return (
        task,
        labels,
        _reference_update(labels, reference_logits, probs=reference_probs),
        FakeUpdate(0, 0, candidate_values, candidate_probabilities),
        reference_entities,
        candidate_entities,
        words,
    )


def test_variable_task_labels_pass_and_report_is_deterministic_and_sanitized(
    tmp_path: Path,
) -> None:
    specifications = [
        _specification("task-a", ("A", "B"), (2.0, -1.0)),
        _specification("task-b", ("X", "Y", "Z"), (-1.0, 3.0, 0.0)),
    ]
    oracle, backend, tasks = _oracle_and_backend(specifications)
    report = run_piimb_validation(
        backend,
        oracle,
        expected_case_count=2,
        expected_tasks=tasks,
    )

    assert report["pass"] is True
    assert [call[0] for call in backend.calls] == [("A", "B"), ("X", "Y", "Z")]
    assert all(session.cleared for session in backend.sessions)
    assert report["metrics"]["candidate_top_label_agreement"] == 1.0
    assert set(report["cases"][0]) == {
        "selection_index",
        "benchmark_split",
        "uid",
        "source_row_index",
        "task_name",
        "pass",
        "candidate_count",
        "top_label_matches",
        "top_label_matches_adjusted",
        "decoded_identity_exact",
        "disagreement_categories",
    }
    validate_sanitized_piimb_report(report)
    first = deterministic_piimb_json_bytes(report)
    assert first == deterministic_piimb_json_bytes(report)
    artifact = write_piimb_validation_report(report, tmp_path / "report.json")
    assert (tmp_path / "report.json").read_bytes() == first
    assert artifact["sha256"] == hashlib.sha256(first).hexdigest()


def test_exact_structural_mismatch_is_a_mandatory_failure() -> None:
    specification = _specification("task-a", ("A", "B"), (2.0, -1.0), words=("Wrong",))
    oracle, backend, tasks = _oracle_and_backend([specification])
    report = run_piimb_validation(
        backend,
        oracle,
        expected_case_count=1,
        expected_tasks=tasks,
    )

    assert report["pass"] is False
    assert report["gates"]["exact_structure"] is False
    assert report["disagreement_categories"] == {"model_words_mismatch": 1}


def test_saturated_near_top_tie_is_counted_but_not_material() -> None:
    labels = ("A", "B")
    logits = (-100.0, -99.95)
    specification = _specification(
        "task-a",
        labels,
        logits,
        reference_probs=(0.0, 0.0),
    )
    oracle, backend, tasks = _oracle_and_backend([specification])
    report = run_piimb_validation(
        backend,
        oracle,
        expected_case_count=1,
        expected_tasks=tasks,
    )

    assert report["pass"] is True
    assert report["metrics"]["candidate_top_label_agreement"] == 0.0
    assert report["metrics"]["candidate_top_label_agreement_adjusted"] == 1.0
    assert report["disagreement_categories"] == {"saturated_near_top_tie": 1}
    assert report["material_categories"] == []


def test_material_top_label_disagreement_fails_even_above_aggregate_count_floor() -> None:
    specification = _specification("task-a", ("A", "B"), (2.0, -1.0), (-1.0, 2.0))
    oracle, backend, tasks = _oracle_and_backend([specification])
    report = run_piimb_validation(
        backend,
        oracle,
        expected_case_count=1,
        expected_tasks=tasks,
    )

    assert report["pass"] is False
    assert "material_top_label_mismatch" in report["material_categories"]
    assert report["gates"]["no_material_or_systematic_category"] is False


def test_near_threshold_entity_is_sanitized_but_raw_identity_gate_still_applies() -> None:
    entity = FakeEntity(0, 3, "A", "Ada", 0.501)
    specification = _specification(
        "task-a",
        ("A", "B"),
        (0.004, -2.0),
        reference_entities=(entity,),
    )
    oracle, backend, tasks = _oracle_and_backend([specification])
    report = run_piimb_validation(
        backend,
        oracle,
        expected_case_count=1,
        expected_tasks=tasks,
    )

    assert report["pass"] is False
    assert report["gates"]["decoded_identity_agreement"] is False
    assert report["disagreement_categories"] == {"near_threshold_decoded_identity_mismatch": 1}
    assert "Ada" not in deterministic_piimb_json_bytes(report).decode()


def test_invalid_selection_checksum_and_forbidden_report_content_are_rejected() -> None:
    specification = _specification("task-a", ("A", "B"), (2.0, -1.0))
    oracle, backend, tasks = _oracle_and_backend([specification])
    cases = cast(list[dict[str, Any]], oracle["cases"])
    selection = cast(dict[str, Any], cases[0]["selection"])
    selection["metadata_sha256"] = "0" * 64
    with pytest.raises(PIIMBValidationError, match="metadata checksum"):
        run_piimb_validation(
            backend,
            oracle,
            expected_case_count=1,
            expected_tasks=tasks,
        )
    with pytest.raises(PIIMBValidationError, match="forbidden fields"):
        deterministic_piimb_json_bytes({"pass": True, "text": "must not leak"})


@pytest.mark.skipif(not PIIMB_REPORT.is_file(), reason="checked-in PIIMB parity report is missing")
def test_checked_in_piimb_report_passes_phase_9_real_data_gate() -> None:
    data = PIIMB_REPORT.read_bytes()
    report = json.loads(data)

    assert hashlib.sha256(data).hexdigest() == (
        "c9459d66d8ae849c4a37b1ff9a68689d57e3ab3005b137b2ba095a6b58f5db4c"
    )
    assert report["pass"] is True
    assert report["model_id"] == REFERENCE_MODEL_ID
    assert report["model_revision_sha"] == REFERENCE_REVISION
    assert all(report["gates"].values())
    assert report["counts"] == {
        "candidate_top_label_matches": 38_106,
        "candidate_top_label_matches_adjusted": 38_106,
        "candidate_vectors": 38_106,
        "cases": 100,
        "decoded_identity_adjusted_cases": 100,
        "decoded_identity_exact_cases": 100,
        "scalar_scores": 1_001_529,
        "structural_exact_cases": 100,
    }
    assert report["metrics"]["candidate_top_label_agreement"] == 1.0
    assert report["metrics"]["decoded_identity_case_agreement"] == 1.0
    assert report["metrics"]["logit_cosine_similarity"] == pytest.approx(
        0.9999999999930581
    )
    assert report["metrics"]["probability_maximum_absolute_error"] == pytest.approx(
        5.150932503750205e-6
    )
    assert report["disagreement_categories"] == {}
    assert report["material_categories"] == []
    assert report["systematic_categories"] == []
    validate_sanitized_piimb_report(report)
