"""Held-out replay of frozen policies with paired parent bootstrap intervals."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from streamner_commit.experiments.config import ResearchConfig, ResearchConfigError
from streamner_commit.experiments.evaluate import ConditionEvaluation, evaluate_policy_condition
from streamner_commit.experiments.policies import PolicySpec, policy_spec_from_mapping
from streamner_commit.experiments.sweep import _write_json_atomic, _write_parquet_atomic
from streamner_commit.experiments.traces import TraceCondition, TraceProvenance
from streamner_commit.metrics import paired_stratified_bootstrap, revision_horizon_metrics
from streamner_commit.serialization import canonical_sha256

BENCHMARK_FILENAME = "test_benchmark.parquet"
BOOTSTRAP_FILENAME = "test_bootstrap.parquet"
REVISION_HORIZONS_FILENAME = "revision_horizons.parquet"
BENCHMARK_MANIFEST_FILENAME = "benchmark_manifest.json"


def run_frozen_benchmark(
    config: ResearchConfig,
    conditions: Sequence[TraceCondition],
    provenance: TraceProvenance,
    frozen: Mapping[str, object],
    output_dir: str | Path,
    *,
    pilot: bool,
) -> Mapping[str, Path]:
    """Evaluate exactly frozen dev selections on separated held-out traces."""

    _validate_test_conditions(config, conditions, pilot=pilot)
    _validate_frozen_provenance(frozen, provenance)
    selected = _frozen_specs(frozen)
    evaluations: dict[tuple[str, str, int], ConditionEvaluation] = {}
    benchmark_rows: list[dict[str, object]] = []
    for mode, key, spec in selected:
        for condition in sorted(conditions, key=lambda item: item.chunk_words):
            evaluation = evaluate_policy_condition(condition, spec, config)
            evaluations[(mode, key, condition.chunk_words)] = evaluation
            benchmark_rows.extend(
                {**row, "selection_mode": mode, "selection_key": key}
                for row in evaluation.aggregate_rows
            )

    bootstrap_rows = _bootstrap_rows(config, conditions, evaluations, pilot=pilot)
    horizon_rows = _revision_horizon_rows(conditions)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    benchmark_path = destination / BENCHMARK_FILENAME
    bootstrap_path = destination / BOOTSTRAP_FILENAME
    horizon_path = destination / REVISION_HORIZONS_FILENAME
    manifest_path = destination / BENCHMARK_MANIFEST_FILENAME
    _write_parquet_atomic(benchmark_path, benchmark_rows)
    _write_parquet_atomic(bootstrap_path, bootstrap_rows)
    _write_parquet_atomic(horizon_path, horizon_rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "mode": "pilot" if pilot else "final",
        "final_reproducibility_gate": provenance.final_reproducibility_gate,
        "split": "test",
        "preset": frozen.get("preset"),
        "research_config_sha256": config.sha256,
        "frozen_payload_sha256": frozen.get("frozen_payload_sha256"),
        "trace_provenance": provenance.to_dict(),
        "bootstrap": {
            "replicates": min(_bootstrap_replicates(config), 50)
            if pilot
            else _bootstrap_replicates(config),
            "confidence": config.bootstrap["confidence"],
            "seed": config.bootstrap["seed"],
            "unit": "parent_id",
            "stratified_by": "task",
        },
        "files": {
            path.name: {"sha256": _sha256_file(path), "rows": rows}
            for path, rows in (
                (benchmark_path, len(benchmark_rows)),
                (bootstrap_path, len(bootstrap_rows)),
                (horizon_path, len(horizon_rows)),
            )
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    _write_json_atomic(manifest_path, manifest)
    return {
        "benchmark": benchmark_path,
        "bootstrap": bootstrap_path,
        "revision_horizons": horizon_path,
        "manifest": manifest_path,
    }


def _frozen_specs(frozen: Mapping[str, object]) -> tuple[tuple[str, str, PolicySpec], ...]:
    result: list[tuple[str, str, PolicySpec]] = []
    for mode in ("matched_quality", "matched_latency"):
        choices = frozen.get(mode)
        if not isinstance(choices, Mapping):
            raise ResearchConfigError(f"frozen config lacks {mode} selections")
        for key in sorted(choices):
            choice = choices[key]
            if not isinstance(key, str) or not isinstance(choice, Mapping):
                raise ResearchConfigError("frozen selection is malformed")
            policy = choice.get("policy")
            if not isinstance(policy, Mapping):
                raise ResearchConfigError("frozen selection lacks an exact policy spec")
            result.append((mode, key, policy_spec_from_mapping(policy)))
    return tuple(result)


def _bootstrap_rows(
    config: ResearchConfig,
    conditions: Sequence[TraceCondition],
    evaluations: Mapping[tuple[str, str, int], ConditionEvaluation],
    *,
    pilot: bool,
) -> tuple[dict[str, object], ...]:
    comparisons = ("fixed-threshold", "fixed-lag", "rescore-patience")
    result: list[dict[str, object]] = []
    replicates = min(_bootstrap_replicates(config), 50) if pilot else _bootstrap_replicates(config)
    confidence = _number(config.bootstrap.get("confidence"), "bootstrap confidence")
    seed = _integer(config.bootstrap.get("seed"), "bootstrap seed")
    for mode in ("matched_quality", "matched_latency"):
        for condition in sorted(conditions, key=lambda item: item.chunk_words):
            proposed = evaluations.get((mode, "stability-gate/full", condition.chunk_words))
            if proposed is None:
                raise ResearchConfigError("frozen selections lack full StabilityGate")
            for baseline_key in comparisons:
                baseline = evaluations.get((mode, baseline_key, condition.chunk_words))
                if baseline is None:
                    raise ResearchConfigError(f"frozen selections lack {baseline_key}")
                records = _paired_example_rows(baseline, proposed)
                for metric in ("selection_error_rate", "mean_commit_context_words"):
                    baseline_statistic, proposed_statistic = _bootstrap_statistics(metric)
                    interval = paired_stratified_bootstrap(
                        records,
                        baseline_statistic=baseline_statistic,
                        comparison_statistic=proposed_statistic,
                        replicates=replicates,
                        confidence=confidence,
                        seed=seed,
                    )
                    result.append(
                        {
                            "selection_mode": mode,
                            "chunk_words": condition.chunk_words,
                            "comparison": "stability-gate/full",
                            "baseline": baseline_key,
                            "metric": metric,
                            **interval.to_dict(),
                        }
                    )
    return tuple(result)


def _paired_example_rows(
    baseline: ConditionEvaluation,
    proposed: ConditionEvaluation,
) -> tuple[dict[str, object], ...]:
    if len(baseline.examples) != len(proposed.examples):
        raise ResearchConfigError("paired policies have different example counts")
    rows: list[dict[str, object]] = []
    for left, right in zip(baseline.examples, proposed.examples, strict=True):
        if (left.task, left.parent_id) != (right.task, right.parent_id):
            raise ResearchConfigError("paired policy examples are not aligned")
        rows.append(
            {
                "task": left.task,
                "parent_id": left.parent_id,
                "baseline_error_numerator": left.selection_error_numerator,
                "baseline_error_denominator": left.selection_error_denominator,
                "proposed_error_numerator": right.selection_error_numerator,
                "proposed_error_denominator": right.selection_error_denominator,
                "baseline_delay_sum": sum(left.context_delays),
                "baseline_delay_count": len(left.context_delays),
                "proposed_delay_sum": sum(right.context_delays),
                "proposed_delay_count": len(right.context_delays),
            }
        )
    return tuple(rows)


def _bootstrap_statistics(metric: str):
    if metric == "selection_error_rate":
        return (
            lambda rows: _record_ratio(
                rows, "baseline_error_numerator", "baseline_error_denominator"
            ),
            lambda rows: _record_ratio(
                rows, "proposed_error_numerator", "proposed_error_denominator"
            ),
        )
    return (
        lambda rows: _record_ratio(rows, "baseline_delay_sum", "baseline_delay_count"),
        lambda rows: _record_ratio(rows, "proposed_delay_sum", "proposed_delay_count"),
    )


def _record_ratio(rows, numerator_field: str, denominator_field: str) -> float:
    numerator = sum(_integer(row[numerator_field], numerator_field) for row in rows)
    denominator = sum(_integer(row[denominator_field], denominator_field) for row in rows)
    return numerator / denominator if denominator else 0.0


def _revision_horizon_rows(conditions: Sequence[TraceCondition]) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    for condition in sorted(conditions, key=lambda item: item.chunk_words):
        for replay in condition.examples:
            report = revision_horizon_metrics(
                tuple(update.to_dict() for update in replay.span_updates)
            )
            for record in report["per_boundary"]:
                result.append(
                    {
                        "split": condition.split,
                        "chunk_words": condition.chunk_words,
                        "task": replay.example.task_name,
                        "example_id_sha256": hashlib.sha256(
                            replay.example.example_id.encode("utf-8")
                        ).hexdigest(),
                        "start_word": record["start_word"],
                        "end_word": record["end_word"],
                        "last_rescore_step": record["last_rescore_step"],
                        "last_rescore_visible_word": record["last_rescore_visible_word"],
                        "revision_horizon_words": record["revision_horizon_words"],
                        "run_id": condition.run.fingerprint.run_id,
                    }
                )
    if not result:
        raise ResearchConfigError("held-out traces contain no revision-horizon records")
    return tuple(result)


def _validate_test_conditions(
    config: ResearchConfig,
    conditions: Sequence[TraceCondition],
    *,
    pilot: bool,
) -> None:
    if not conditions or any(condition.split != "test" for condition in conditions):
        raise ResearchConfigError("benchmark accepts held-out test traces only")
    chunks = {condition.chunk_words for condition in conditions}
    expected = {config.primary_chunk_words} if pilot else set(config.chunk_words)
    if not expected.issubset(chunks):
        raise ResearchConfigError("benchmark is missing required chunk conditions")
    for condition in conditions:
        if {example.example.task_name for example in condition.examples} != set(config.tasks):
            raise ResearchConfigError("held-out condition does not cover every configured task")


def _validate_frozen_provenance(frozen: Mapping[str, object], provenance: TraceProvenance) -> None:
    dev = frozen.get("trace_provenance")
    if not isinstance(dev, Mapping):
        raise ResearchConfigError("frozen development provenance is missing")
    if dev.get("project_git_commit") != provenance.project_git_commit:
        raise ResearchConfigError("development/test project commits differ")
    if dev.get("source_tree_sha256") != provenance.source_tree_sha256:
        raise ResearchConfigError("development/test source-tree provenance differs")


def _record_value(row: Mapping[str, object], field: str) -> int:
    return _integer(row.get(field), field)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchConfigError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResearchConfigError(f"{name} must be numeric")
    return float(value)


def _bootstrap_replicates(config: ResearchConfig) -> int:
    return _integer(config.bootstrap.get("replicates"), "bootstrap replicates")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BENCHMARK_FILENAME",
    "BENCHMARK_MANIFEST_FILENAME",
    "BOOTSTRAP_FILENAME",
    "REVISION_HORIZONS_FILENAME",
    "run_frozen_benchmark",
]
