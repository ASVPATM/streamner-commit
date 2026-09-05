from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamner_commit.mlx.assets import REFERENCE_REVISION

REPORT = Path("artifacts/reference") / REFERENCE_REVISION / "mlx_cold_parity_report.json"


@pytest.mark.skipif(not REPORT.is_file(), reason="ignored cold parity report not generated")
def test_generated_cold_parity_report_passes_every_phase_7_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["kind"] == "mlx_cold_parity_report"
    assert report["model_revision_sha"] == REFERENCE_REVISION
    assert report["pass"] is True
    assert report["case_count"] == 10
    assert len(report["cases"]) == 10
    assert all(case["pass"] for case in report["cases"])
    assert all(all(case["discrete_exact"].values()) for case in report["cases"])
    assert all(all(case["metadata_exact"].values()) for case in report["cases"])
    assert sum(case["candidate_count"] for case in report["cases"]) == 881
    assert sum(case["material_top_label_mismatches"] for case in report["cases"]) == 0
    for case in report["cases"]:
        assert all(
            diagnostic["shape_equal"]
            and diagnostic["cosine_similarity"] >= report["minimum_component_cosine"]
            for diagnostic in case["numerical"].values()
        )
        assert all(
            case["decoder"][str(threshold)]["boundary_label_match"] for threshold in (0.3, 0.5, 0.7)
        )
