from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from streamner_commit.metrics.stability import (
    aggregate_premise_reports,
    analyze_premise_trace,
    boundary_extension_metrics,
    decoded_snapshot_churn,
    raw_revision_metrics,
    revision_horizon_metrics,
)


def update(
    *,
    step: int,
    boundary: tuple[int, int],
    kind: str,
    top_probability: float,
    previous: float | None,
    visible_word_count: int,
) -> dict[str, object]:
    start_word, end_word = boundary
    return {
        "step": step,
        "start_word": start_word,
        "end_word": end_word,
        "update_kind": kind,
        "top_probability": top_probability,
        "previous_top_probability": previous,
        "top_probability_delta": (None if previous is None else top_probability - previous),
        "visible_word_count": visible_word_count,
        "tail_distance_words": (visible_word_count - 1) - end_word,
    }


def entity(start: int, end: int, label: str, text: str) -> dict[str, object]:
    return {
        "start_char": start,
        "end_char": end,
        "label": label,
        "text": text,
        "score": 0.9,
    }


def snapshot(step: int, entities: list[dict[str, object]]) -> dict[str, object]:
    return {"step": step, "public_entities": entities}


def synthetic_trace(example_id: str = "example-1") -> dict[str, object]:
    updates = [
        update(
            step=1,
            boundary=(0, 0),
            kind="new",
            top_probability=0.6,
            previous=None,
            visible_word_count=1,
        ),
        update(
            step=2,
            boundary=(0, 0),
            kind="rescore",
            top_probability=0.8,
            previous=0.6,
            visible_word_count=3,
        ),
        update(
            step=2,
            boundary=(0, 1),
            kind="new",
            top_probability=0.7,
            previous=None,
            visible_word_count=3,
        ),
        update(
            step=3,
            boundary=(0, 0),
            kind="rescore",
            top_probability=0.75,
            previous=0.8,
            visible_word_count=4,
        ),
        update(
            step=3,
            boundary=(0, 1),
            kind="rescore",
            top_probability=0.9,
            previous=0.7,
            visible_word_count=4,
        ),
    ]
    steps = [
        snapshot(1, [entity(0, 5, "person", "Sarah")]),
        snapshot(
            2,
            [
                entity(0, 13, "person", "Sarah Johnson"),
                entity(20, 30, "email address", "a@fake.test"),
            ],
        ),
        snapshot(3, [entity(0, 13, "person", "Sarah Johnson")]),
    ]
    return {"example_id": example_id, "span_updates": updates, "steps": steps}


def test_raw_rescore_rates_and_probability_movements() -> None:
    metrics = raw_revision_metrics(synthetic_trace()["span_updates"])  # type: ignore[arg-type]

    assert metrics["span_update_count"] == 5
    assert metrics["rescore_update_count"] == 3
    assert metrics["rescore_update_rate"] == pytest.approx(0.6)
    assert metrics["observed_boundary_count"] == 2
    assert metrics["rescored_boundary_count"] == 2
    assert metrics["rescored_boundary_rate"] == 1.0
    assert metrics["probability_movement"] == {
        "count": 3,
        "mean": pytest.approx(0.15),
        "median": pytest.approx(0.2),
        "maximum": pytest.approx(0.2),
        "absolute_values": pytest.approx([0.2, 0.05, 0.2]),
    }


def test_decoded_snapshot_churn_uses_entity_identity_symmetric_difference() -> None:
    metrics = decoded_snapshot_churn(synthetic_trace()["steps"])  # type: ignore[arg-type]

    assert metrics["transition_count"] == 2
    assert metrics["total_symmetric_difference"] == 4
    assert metrics["mean_symmetric_difference"] == 2.0
    assert metrics["micro_normalized_churn"] == pytest.approx(0.8)
    assert metrics["mean_normalized_churn"] == pytest.approx(0.75)
    assert [row["symmetric_difference_count"] for row in metrics["transitions"]] == [3, 1]


def test_boundary_extensions_require_same_start_and_label() -> None:
    steps = synthetic_trace()["steps"]
    metrics = boundary_extension_metrics(steps)  # type: ignore[arg-type]

    assert metrics == {
        "count": 1,
        "cases": [
            {
                "from_step": 1,
                "to_step": 2,
                "start_char": 0,
                "from_end_char": 5,
                "to_end_char": 13,
                "label": "person",
                "from_text": "Sarah",
                "to_text": "Sarah Johnson",
            }
        ],
    }

    non_extensions = [
        snapshot(1, [entity(0, 5, "person", "Sarah")]),
        snapshot(
            2,
            [
                entity(1, 13, "person", "arah Johnson"),
                entity(0, 13, "organization", "Sarah Johnson"),
            ],
        ),
    ]
    assert boundary_extension_metrics(non_extensions)["count"] == 0


def test_revision_horizon_uses_last_raw_rescore_model_word_coordinates() -> None:
    metrics = revision_horizon_metrics(synthetic_trace()["span_updates"])  # type: ignore[arg-type]

    assert metrics["boundary_count"] == 2
    assert metrics["values_words"] == [3.0, 2.0]
    assert metrics["mean"] == 2.5
    assert metrics["maximum"] == 3.0
    assert metrics["per_boundary"] == [
        {
            "start_word": 0,
            "end_word": 0,
            "last_rescore_step": 3,
            "last_rescore_visible_word": 3,
            "revision_horizon_words": 3,
        },
        {
            "start_word": 0,
            "end_word": 1,
            "last_rescore_step": 3,
            "last_rescore_visible_word": 3,
            "revision_horizon_words": 2,
        },
    ]


def test_trace_analysis_and_aggregation_do_not_cross_example_boundaries() -> None:
    first = analyze_premise_trace(synthetic_trace("first"))
    second_trace = synthetic_trace("second")
    second_trace["steps"] = second_trace["steps"][:1]  # type: ignore[index]
    second = analyze_premise_trace(second_trace)
    summary = aggregate_premise_reports([first, second])

    assert summary["trace_count"] == 2
    assert summary["raw_revisions"]["span_update_count"] == 10
    assert summary["raw_revisions"]["rescore_update_rate"] == pytest.approx(0.6)
    assert summary["decoded_snapshot_churn"]["transition_count"] == 2
    assert summary["boundary_extensions"]["count"] == 1
    assert summary["boundary_extensions"]["cases"][0]["example_id"] == "first"
    assert summary["revision_horizon"]["boundary_count"] == 4


def test_empty_trace_has_explicit_zero_or_null_descriptives() -> None:
    report = analyze_premise_trace({"example_id": "empty", "span_updates": [], "steps": []})

    assert report["raw_revisions"]["rescore_update_rate"] == 0.0
    assert report["raw_revisions"]["probability_movement"]["mean"] is None
    assert report["decoded_snapshot_churn"]["transition_count"] == 0
    assert report["boundary_extensions"]["count"] == 0
    assert report["revision_horizon"]["boundary_count"] == 0


def write_payload(path: Path, *, run_id: str, chunk_words: int, traces: list[object]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "words_per_chunk": chunk_words,
                "traces": traces,
            }
        ),
        encoding="utf-8",
    )


def test_cli_accepts_multiple_trace_files_and_writes_json(tmp_path: Path) -> None:
    first_path = tmp_path / "one-word.json"
    second_path = tmp_path / "two-word.json"
    output = tmp_path / "report" / "premise.json"
    write_payload(first_path, run_id="run-1", chunk_words=1, traces=[synthetic_trace("a")])
    write_payload(second_path, run_id="run-2", chunk_words=2, traces=[synthetic_trace("b")])

    subprocess.run(
        [
            sys.executable,
            "scripts/premise_validation.py",
            "--trace",
            str(first_path),
            "--trace",
            str(second_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["summary"]["trace_count"] == 2
    assert [source["file"] for source in payload["sources"]] == [
        "one-word.json",
        "two-word.json",
    ]
    assert [row["words_per_chunk"] for row in payload["traces"]] == [1, 2]
    assert str(tmp_path) not in output.read_text(encoding="utf-8")


def test_cli_accepts_a_single_unwrapped_trace(tmp_path: Path) -> None:
    path = tmp_path / "single.json"
    output = tmp_path / "single-report.json"
    path.write_text(json.dumps(synthetic_trace()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/premise_validation.py",
            "--trace",
            str(path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["summary"]["trace_count"] == 1
    assert report["traces"][0]["example_id"] == "example-1"


def test_malformed_order_and_tail_distance_are_rejected() -> None:
    updates = synthetic_trace()["span_updates"]  # type: ignore[assignment]
    with pytest.raises(ValueError, match="nondecreasing"):
        raw_revision_metrics(list(reversed(updates)))  # type: ignore[arg-type]

    malformed = [dict(updates[1])]  # type: ignore[index]
    malformed[0]["tail_distance_words"] = 999
    with pytest.raises(ValueError, match="tail_distance_words"):
        revision_horizon_metrics(malformed)
