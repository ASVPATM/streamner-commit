from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamner_commit.experiments.config import ResearchConfigError, load_research_config
from streamner_commit.experiments.policies import pilot_policy_specs
from streamner_commit.experiments.sweep import (
    _combine_provenance,
    _read_policy_checkpoint,
    _write_policy_checkpoint,
    freeze_development_configs,
)
from streamner_commit.experiments.traces import (
    ExperimentTraceError,
    TraceCondition,
    TraceProvenance,
    _status_has_non_analysis_changes,
    trace_provenance,
)
from streamner_commit.serialization import (
    TraceFingerprint,
    TraceRun,
    TraceRunData,
    canonical_json_bytes,
    canonical_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_boundary_has_no_model_or_backend_imports() -> None:
    paths = [
        *sorted((PROJECT_ROOT / "src/streamner_commit/experiments").glob("*.py")),
        PROJECT_ROOT / "scripts/run_policy_sweep.py",
        PROJECT_ROOT / "scripts/run_benchmark.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "streamner_commit.mlx" not in source
    assert "streamner_commit.backends" not in source
    assert "expand_policy_grid" not in (
        PROJECT_ROOT / "src/streamner_commit/experiments/benchmark.py"
    ).read_text(encoding="utf-8")


def test_development_freezing_is_deterministic_and_rejects_test_rows() -> None:
    config = load_research_config(PROJECT_ROOT / "configs/research.yaml")
    specs = pilot_policy_specs(config)
    rows = [
        {
            "policy_id": spec.policy_id,
            "split": "dev",
            "aggregation": "overall",
            "strict_precision": 0.8,
            "strict_recall": 0.7,
            "strict_f1": 0.75,
            "mean_commit_context_words": 1.0,
            "selection_error_rate": 0.2,
            "gold_premature_rate": 0.1,
            "wrong_commitment_rate": 0.1,
            "missed_entity_rate": 0.1,
        }
        for spec in specs
    ]
    provenance = TraceProvenance(("trace-a",), ("f" * 64,), None, "e" * 64, False)

    first = freeze_development_configs(config, rows, specs, provenance, preset="smoke")
    second = freeze_development_configs(
        config, list(reversed(rows)), specs, provenance, preset="smoke"
    )
    assert first == second
    assert first["dev_only_selection"] is True
    assert first["trace_provenance"]["final_reproducibility_gate"] is False  # type: ignore[index]

    rows[0]["split"] = "test"
    with pytest.raises(ResearchConfigError, match="development"):
        freeze_development_configs(config, rows, specs, provenance, preset="smoke")


def test_final_trace_gate_rejects_uncommitted_runs_but_pilot_records_tree_digest() -> None:
    payload = {"project_git_commit": None, "source_tree_sha256": "a" * 64}
    digest = canonical_sha256(payload)
    fingerprint = TraceFingerprint(f"trace-{digest[:24]}", digest, payload)
    run = TraceRun(Path("unused"), {}, fingerprint, TraceRunData((), (), (), ()))
    condition = TraceCondition(run, "dev", "smoke", 1, ())

    pilot = trace_provenance((condition,), pilot=True, project_root=PROJECT_ROOT)
    assert pilot.final_reproducibility_gate is False
    assert pilot.source_tree_sha256 == "a" * 64
    with pytest.raises(ExperimentTraceError, match="project_git_commit"):
        trace_provenance((condition,), pilot=False, project_root=PROJECT_ROOT)

    dirty_payload = {"project_git_commit": "b" * 40, "source_tree_sha256": "a" * 64}
    dirty_digest = canonical_sha256(dirty_payload)
    dirty_run = TraceRun(
        Path("unused"),
        {},
        TraceFingerprint(f"trace-{dirty_digest[:24]}", dirty_digest, dirty_payload),
        TraceRunData((), (), (), ()),
    )
    dirty_condition = TraceCondition(dirty_run, "dev", "smoke", 1, ())
    with pytest.raises(ExperimentTraceError, match="dirty source tree"):
        trace_provenance((dirty_condition,), pilot=False, project_root=PROJECT_ROOT)


def test_clean_source_gate_allows_only_generated_analysis_outputs() -> None:
    assert not _status_has_non_analysis_changes(
        " M results/analysis/frozen_configs.json\n"
        "?? results/analysis/benchmark_manifest.json\n"
    )
    assert _status_has_non_analysis_changes(" M configs/research.yaml\n")
    assert _status_has_non_analysis_changes(
        "R  results/analysis/old.json -> src/streamner_commit/changed.py\n"
    )


def test_policy_checkpoint_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    config = load_research_config(PROJECT_ROOT / "configs/research.yaml")
    spec = pilot_policy_specs(config)[0]
    row = {
        "policy_id": spec.policy_id,
        "split": "dev",
        "chunk_words": 1,
        "aggregation": "overall",
        "strict_f1": 0.75,
    }
    path = tmp_path / "policy.json"
    condition_key = "a" * 64
    _write_policy_checkpoint(
        path,
        condition_key=condition_key,
        spec=spec,
        chunk_words=1,
        rows=(row,),
    )
    assert _read_policy_checkpoint(
        path,
        condition_key=condition_key,
        spec=spec,
        chunk_words=1,
    ) == (dict(sorted(row.items())),)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["aggregate_rows"][0]["strict_f1"] = 0.1
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ResearchConfigError, match="digest differs"):
        _read_policy_checkpoint(
            path,
            condition_key=condition_key,
            spec=spec,
            chunk_words=1,
        )


def test_checkpointed_sweep_rejects_mixed_trace_provenance() -> None:
    first = TraceProvenance(("trace-a",), ("a" * 64,), "1" * 40, None, True)
    second = TraceProvenance(("trace-b",), ("b" * 64,), "2" * 40, None, True)
    with pytest.raises(ResearchConfigError, match="mixed provenance"):
        _combine_provenance((first, second))
