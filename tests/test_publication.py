from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.text import Text

from streamner_commit.publication import (
    PILOT_LABEL,
    SYNTHETIC_TRAJECTORY_PREFIXES,
    generate_figures,
    generate_tables,
)


def test_required_publication_outputs_are_pilot_labeled_and_privacy_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    secret = "BENCHMARK_CONTENT_MUST_NOT_APPEAR"
    families = (
        ("fixed-threshold", "main"),
        ("fixed-lag", "main"),
        ("rescore-patience", "main"),
        ("stability-gate", "full"),
    )
    pareto_rows = [
        {
            "policy_id": f"{family}-{variant}",
            "policy_family": family,
            "policy_variant": variant,
            "analysis_only": False,
            "split": "dev",
            "chunk_words": 1,
            "aggregation": "overall",
            "task": secret,
            "pareto_objective": "delay_vs_strict_f1",
            "mean_commit_context_words": 0.5 + index,
            "strict_f1": 0.80 + (0.01 * index),
        }
        for index, (family, variant) in enumerate(families)
    ]
    pareto_path = analysis / "dev_pareto.parquet"
    pd.DataFrame(pareto_rows).to_parquet(pareto_path, index=False)

    benchmark_rows: list[dict[str, object]] = []
    for chunk in (1, 2, 4, 8):
        for index, (family, variant) in enumerate(families):
            benchmark_rows.append(
                {
                    "policy_id": f"{family}-{variant}",
                    "policy_family": family,
                    "policy_variant": variant,
                    "analysis_only": False,
                    "split": "test",
                    "chunk_words": chunk,
                    "aggregation": "overall",
                    "task": secret,
                    "selection_mode": "matched_quality",
                    "strict_precision": 0.9,
                    "strict_recall": 0.8,
                    "strict_f1": 0.84 + (0.01 * index),
                    "masking_f2": 0.88,
                    "mean_commit_context_words": float(chunk + index),
                    "median_commit_context_words": float(chunk),
                    "gold_premature_rate": 0.05,
                    "wrong_commitment_rate": 0.04,
                    "blocked_revision_count": index,
                    "blocked_revisions_per_100_commitments": float(index),
                }
            )
    for variant in ("minus_instability", "minus_label_margin", "minus_extension"):
        row = benchmark_rows[-1].copy()
        row.update(
            {
                "policy_id": f"stability-gate-{variant}",
                "policy_variant": variant,
                "chunk_words": 1,
            }
        )
        benchmark_rows.append(row)
    benchmark_path = analysis / "test_benchmark.parquet"
    pd.DataFrame(benchmark_rows).to_parquet(benchmark_path, index=False)

    horizons_path = analysis / "revision_horizons.parquet"
    pd.DataFrame(
        [
            {
                "split": "test",
                "chunk_words": 1,
                "task": secret,
                "example_id_sha256": "a" * 64,
                "start_word": 0,
                "end_word": 0,
                "last_rescore_step": 1,
                "last_rescore_visible_word": horizon,
                "revision_horizon_words": horizon,
                "run_id": "synthetic-run",
            }
            for horizon in (0, 1, 2)
        ]
    ).to_parquet(horizons_path, index=False)
    manifest_path = analysis / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "pilot",
                "final_reproducibility_gate": False,
                "trace_provenance": {},
            }
        ),
        encoding="utf-8",
    )

    streaming_path = tmp_path / "streaming.json"
    streaming_path.write_text(
        json.dumps(
            {
                "pass": True,
                "totals": {"step_count": 1},
                "numerical": {"top_label_matches": 2, "top_label_total": 2},
                "conditions": [
                    {
                        "cases": [
                            {
                                "steps": [
                                    {
                                        "exact": {"structure": True},
                                        "decoded": {"identity_match": True},
                                    }
                                ]
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    piimb_path = tmp_path / "piimb.json"
    piimb_path.write_text(
        json.dumps(
            {
                "pass": True,
                "counts": {"candidate_vectors": 3},
                "metrics": {
                    "structural_case_agreement": 1.0,
                    "candidate_top_label_agreement": 1.0,
                    "decoded_identity_case_agreement": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, set[str]] = {}
    original_savefig = Figure.savefig

    def capture_savefig(self: Figure, *args, **kwargs) -> None:
        captured.setdefault(self.get_label(), set()).update(
            item.get_text() for item in self.findobj(match=Text)
        )
        original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    figure_paths = generate_figures(
        pareto_path=pareto_path,
        revision_horizons_path=horizons_path,
        benchmark_manifest_path=manifest_path,
        streaming_parity_path=streaming_path,
        piimb_parity_path=piimb_path,
        output_dir=tmp_path / "figures",
    )
    table_paths = generate_tables(
        benchmark_path=benchmark_path,
        benchmark_manifest_path=manifest_path,
        output_dir=tmp_path / "tables",
    )

    assert len(figure_paths) == 8 and all(path.stat().st_size > 0 for path in figure_paths)
    assert len(table_paths) == 6 and all(path.stat().st_size > 0 for path in table_paths)
    assert all(PILOT_LABEL in labels for labels in captured.values())
    trajectory_text = captured["figure3_synthetic_trajectory"]
    assert set(SYNTHETIC_TRAJECTORY_PREFIXES).issubset(trajectory_text)
    assert secret not in trajectory_text
    for path in table_paths:
        if path.suffix == ".md":
            assert PILOT_LABEL in path.read_text(encoding="utf-8")
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                assert all(row["Status"] == PILOT_LABEL for row in csv.DictReader(handle))
