from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamner_commit.mlx.assets import REFERENCE_MODEL_ID, REFERENCE_REVISION

REPORT = Path(__file__).resolve().parents[1] / "results" / "parity" / "parity_report.json"


@pytest.mark.skipif(not REPORT.is_file(), reason="checked-in streaming parity report is missing")
def test_checked_in_streaming_parity_report_passes_phase_9_synthetic_gates() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["kind"] == "mlx_streaming_parity_report"
    assert report["model_id"] == REFERENCE_MODEL_ID
    assert report["model_revision_sha"] == REFERENCE_REVISION
    assert report["chunk_unit_schedules"] == [1, 2, 4]
    assert report["canonical_threshold"] == 0.5
    assert report["pass"] is True
    assert report["totals"] == {
        "case_count": 30,
        "condition_count": 3,
        "exact_step_count": 172,
        "material_reason_count": 0,
        "near_threshold_count": 0,
        "near_top_tie_count": 0,
        "step_count": 172,
    }

    numerical = report["numerical"]
    assert numerical["within_tolerances"] is True
    assert numerical["vector_count"] == 6_400
    assert numerical["scalar_count"] == 38_400
    assert numerical["logit_cosine_similarity"] == pytest.approx(0.9999999999963489)
    assert numerical["probability_mean_absolute_error"] == pytest.approx(9.021745796725868e-9)
    assert numerical["probability_maximum_absolute_error"] == pytest.approx(
        6.669524558966522e-6
    )
    assert numerical["top_label_matches"] == 6_400
    assert numerical["top_label_total"] == 6_400

    conditions = report["conditions"]
    assert [condition["chunk_units"] for condition in conditions] == [1, 2, 4]
    assert [condition["case_count"] for condition in conditions] == [10, 10, 10]
    assert [sum(case["step_count"] for case in condition["cases"]) for condition in conditions] == [
        94,
        50,
        28,
    ]
    assert all(condition["pass"] for condition in conditions)

    cases = [case for condition in conditions for case in condition["cases"]]
    assert all(case["pass"] and not case["material_reasons"] for case in cases)
    assert sum(len(case["categorized_disagreements"]) for case in cases) == 0
    assert {
        disagreement["category"]
        for case in cases
        for disagreement in case["categorized_disagreements"]
    } == set()
    for case in cases:
        assert case["final"]["pass"] is True
        assert all(case["final"]["exact"].values())
        assert case["final"]["decoded"]["identity_match"] is True
        for step in case["steps"]:
            assert step["pass"] is True
            assert all(step["exact"].values())
            assert step["reference_update_count"] == step["mlx_update_count"]
            assert step["decoded"]["identity_match"] is True
            assert not step["decoded"]["material"]
            assert not step["decoded"]["near_threshold"]
