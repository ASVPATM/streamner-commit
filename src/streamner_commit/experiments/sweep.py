"""Development-only policy sweep, Pareto extraction, and automatic freezing."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from streamner_commit.experiments.config import (
    ResearchConfig,
    ResearchConfigError,
    load_research_config,
)
from streamner_commit.experiments.evaluate import evaluate_policy_condition
from streamner_commit.experiments.policies import PolicySpec, expand_policy_grid
from streamner_commit.experiments.traces import (
    TraceCondition,
    TraceProvenance,
    load_trace_conditions,
    trace_provenance,
)
from streamner_commit.metrics import pareto_front
from streamner_commit.serialization import canonical_json_bytes, canonical_sha256, project_version

SWEEP_FILENAME = "dev_policy_sweep.parquet"
PARETO_FILENAME = "dev_pareto.parquet"
FROZEN_FILENAME = "frozen_configs.json"
SWEEP_MANIFEST_FILENAME = "dev_sweep_manifest.json"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ConditionSweepResult:
    chunk_words: int
    rows: tuple[dict[str, object], ...]
    provenance: TraceProvenance
    evaluated: int
    reused: int


def run_development_sweep(
    config: ResearchConfig,
    conditions: Sequence[TraceCondition],
    provenance: TraceProvenance,
    output_dir: str | Path,
    *,
    preset: str,
    specs: Sequence[PolicySpec] | None = None,
) -> Mapping[str, Path]:
    """Evaluate configured policies on development traces and freeze selections."""

    _validate_development_conditions(config, conditions)
    policy_specs = tuple(expand_policy_grid(config) if specs is None else specs)
    if not policy_specs:
        raise ResearchConfigError("policy sweep requires at least one configuration")
    rows: list[dict[str, object]] = []
    for condition in sorted(conditions, key=lambda item: item.chunk_words):
        for spec in sorted(policy_specs, key=lambda item: item.policy_id):
            evaluation = evaluate_policy_condition(condition, spec, config)
            rows.extend(evaluation.aggregate_rows)
            del evaluation
    return _finalize_development_sweep(
        config,
        rows,
        policy_specs,
        provenance,
        output_dir,
        preset=preset,
    )


def run_checkpointed_development_sweep(
    config: ResearchConfig,
    trace_root: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    *,
    preset: str,
    chunk_words: Sequence[int] | None = None,
    checkpoint_dir: str | Path | None = None,
    workers: int = 2,
    progress_seconds: float = 15.0,
) -> Mapping[str, Path]:
    """Run the full dev sweep in bounded, resumable chunk workers.

    Each worker holds one trace condition and one policy's example evaluations at
    a time. Only compact aggregate rows are checkpointed. Process scheduling does
    not change policy evaluation or metric aggregation.
    """

    chunks = tuple(config.chunk_words if chunk_words is None else chunk_words)
    if (
        not chunks
        or len(set(chunks)) != len(chunks)
        or any(chunk not in config.chunk_words for chunk in chunks)
        or config.primary_chunk_words not in chunks
    ):
        raise ResearchConfigError(
            "checkpointed development sweep requires unique configured chunks including primary"
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ResearchConfigError("workers must be a positive integer")
    if progress_seconds <= 0:
        raise ResearchConfigError("progress_seconds must be positive")

    policy_specs = tuple(sorted(expand_policy_grid(config), key=lambda item: item.policy_id))
    if not policy_specs:
        raise ResearchConfigError("policy sweep requires at least one configuration")
    root = Path(project_root).resolve()
    current_commit, _current_tree = project_version(root)
    checkpoints = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else Path(output_dir) / ".sweep_checkpoints"
    )
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_root = checkpoints / "_progress" / f"run-{os.getpid()}"
    progress_root.mkdir(parents=True, exist_ok=True)
    progress_paths = {chunk: progress_root / f"chunk-{chunk}.json" for chunk in chunks}
    for chunk, path in progress_paths.items():
        _write_progress(path, chunk=chunk, completed=0, evaluated=0, reused=0, stage="queued")

    maximum_workers = min(workers, len(chunks))
    started = time.monotonic()
    executor = ProcessPoolExecutor(
        max_workers=maximum_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )
    futures: dict[Future[_ConditionSweepResult], int] = {}
    completed_results: list[_ConditionSweepResult] = []
    try:
        for chunk in chunks:
            future = executor.submit(
                _evaluate_condition_checkpointed,
                config.path,
                Path(trace_root),
                root,
                preset,
                chunk,
                checkpoints,
                progress_paths[chunk],
                current_commit,
            )
            futures[future] = chunk
        pending = set(futures)
        _print_sweep_progress(progress_paths, len(policy_specs), started)
        while pending:
            done, pending = wait(pending, timeout=progress_seconds)
            for future in done:
                completed_results.append(future.result())
            _print_sweep_progress(progress_paths, len(policy_specs), started)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        for path in progress_paths.values():
            path.unlink(missing_ok=True)
        try:
            progress_root.rmdir()
            progress_root.parent.rmdir()
        except OSError:
            pass

    ordered = tuple(sorted(completed_results, key=lambda item: item.chunk_words))
    rows = tuple(row for result in ordered for row in result.rows)
    provenance = _combine_provenance(tuple(result.provenance for result in ordered))
    print(
        "sweep complete "
        f"evaluated={sum(result.evaluated for result in ordered)} "
        f"reused={sum(result.reused for result in ordered)} "
        f"elapsed={_format_duration(time.monotonic() - started)}",
        file=sys.stderr,
        flush=True,
    )
    return _finalize_development_sweep(
        config,
        rows,
        policy_specs,
        provenance,
        output_dir,
        preset=preset,
    )


def _evaluate_condition_checkpointed(
    config_path: Path,
    trace_root: Path,
    project_root: Path,
    preset: str,
    chunk_words: int,
    checkpoint_root: Path,
    progress_path: Path,
    required_project_commit: str | None,
) -> _ConditionSweepResult:
    started = time.monotonic()
    _write_progress(
        progress_path,
        chunk=chunk_words,
        completed=0,
        evaluated=0,
        reused=0,
        stage="loading-trace",
    )
    config = load_research_config(config_path)
    specs = tuple(sorted(expand_policy_grid(config), key=lambda item: item.policy_id))
    conditions = load_trace_conditions(
        trace_root,
        config,
        preset=preset,
        split="dev",
        chunk_words=(chunk_words,),
        required_project_commit=required_project_commit,
    )
    provenance = trace_provenance(conditions, pilot=False, project_root=project_root)
    condition = conditions[0]
    condition_key = canonical_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "research_config_sha256": config.sha256,
            "preset": preset,
            "split": "dev",
            "chunk_words": chunk_words,
            "trace_run_id": condition.run.fingerprint.run_id,
            "trace_fingerprint_sha256": condition.run.fingerprint.sha256,
            "trace_provenance": provenance.to_dict(),
            "policy_grid_sha256": canonical_sha256([spec.to_dict() for spec in specs]),
        }
    )
    condition_dir = checkpoint_root / f"chunk-{chunk_words}-{condition_key[:16]}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    evaluated = 0
    reused = 0
    last_progress = 0.0
    for completed, spec in enumerate(specs, start=1):
        checkpoint_path = condition_dir / f"policy-{canonical_sha256(spec.to_dict())}.json"
        aggregate_rows = _read_policy_checkpoint(
            checkpoint_path,
            condition_key=condition_key,
            spec=spec,
            chunk_words=chunk_words,
        )
        if aggregate_rows is None:
            evaluation = evaluate_policy_condition(condition, spec, config)
            aggregate_rows = _normalized_rows(evaluation.aggregate_rows)
            _write_policy_checkpoint(
                checkpoint_path,
                condition_key=condition_key,
                spec=spec,
                chunk_words=chunk_words,
                rows=aggregate_rows,
            )
            del evaluation
            evaluated += 1
        else:
            reused += 1
        rows.extend(aggregate_rows)
        now = time.monotonic()
        if now - last_progress >= 1.0 or completed == len(specs):
            _write_progress(
                progress_path,
                chunk=chunk_words,
                completed=completed,
                evaluated=evaluated,
                reused=reused,
                stage="evaluating" if completed < len(specs) else "complete",
                policy_id=spec.policy_id,
                elapsed_seconds=now - started,
            )
            last_progress = now
    return _ConditionSweepResult(
        chunk_words=chunk_words,
        rows=tuple(rows),
        provenance=provenance,
        evaluated=evaluated,
        reused=reused,
    )


def _finalize_development_sweep(
    config: ResearchConfig,
    rows: Sequence[Mapping[str, object]],
    policy_specs: Sequence[PolicySpec],
    provenance: TraceProvenance,
    output_dir: str | Path,
    *,
    preset: str,
) -> Mapping[str, Path]:
    primary = tuple(
        dict(row)
        for row in rows
        if row.get("chunk_words") == config.primary_chunk_words
        and row.get("aggregation") == "overall"
    )
    frozen = freeze_development_configs(
        config,
        primary,
        policy_specs,
        provenance,
        preset=preset,
    )
    pareto_rows = _pareto_rows(primary)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sweep_path = destination / SWEEP_FILENAME
    pareto_path = destination / PARETO_FILENAME
    frozen_path = destination / FROZEN_FILENAME
    manifest_path = destination / SWEEP_MANIFEST_FILENAME
    _write_parquet_atomic(sweep_path, rows)
    _write_parquet_atomic(pareto_path, pareto_rows)
    _write_json_atomic(frozen_path, frozen)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "mode": "final" if provenance.final_reproducibility_gate else "pilot",
        "final_reproducibility_gate": provenance.final_reproducibility_gate,
        "split": "dev",
        "preset": preset,
        "research_config_sha256": config.sha256,
        "trace_provenance": provenance.to_dict(),
        "files": {
            sweep_path.name: {"sha256": _sha256_file(sweep_path), "rows": len(rows)},
            pareto_path.name: {"sha256": _sha256_file(pareto_path), "rows": len(pareto_rows)},
            frozen_path.name: {"sha256": _sha256_file(frozen_path)},
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    _write_json_atomic(manifest_path, manifest)
    return {
        "sweep": sweep_path,
        "pareto": pareto_path,
        "frozen": frozen_path,
        "manifest": manifest_path,
    }


def _write_policy_checkpoint(
    path: Path,
    *,
    condition_key: str,
    spec: PolicySpec,
    chunk_words: int,
    rows: Sequence[Mapping[str, object]],
) -> None:
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition_key": condition_key,
        "chunk_words": chunk_words,
        "policy_spec": spec.to_dict(),
        "aggregate_rows": [dict(row) for row in rows],
    }
    payload["checkpoint_payload_sha256"] = canonical_sha256(payload)
    _write_json_atomic(path, payload)


def _read_policy_checkpoint(
    path: Path,
    *,
    condition_key: str,
    spec: PolicySpec,
    chunk_words: int,
) -> tuple[dict[str, object], ...] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        value: Any = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchConfigError(f"policy checkpoint is unreadable: {path.name}") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ResearchConfigError(f"policy checkpoint is not canonical: {path.name}")
    digest = value.get("checkpoint_payload_sha256")
    unsigned = {key: item for key, item in value.items() if key != "checkpoint_payload_sha256"}
    if digest != canonical_sha256(unsigned):
        raise ResearchConfigError(f"policy checkpoint digest differs: {path.name}")
    if (
        value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or value.get("condition_key") != condition_key
        or value.get("chunk_words") != chunk_words
        or value.get("policy_spec") != spec.to_dict()
    ):
        raise ResearchConfigError(f"policy checkpoint identity differs: {path.name}")
    raw_rows = value.get("aggregate_rows")
    if isinstance(raw_rows, str | bytes) or not isinstance(raw_rows, Sequence):
        raise ResearchConfigError(f"policy checkpoint rows are malformed: {path.name}")
    rows: list[Mapping[str, object]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise ResearchConfigError(f"policy checkpoint row is malformed: {path.name}")
        if (
            row.get("split") != "dev"
            or row.get("chunk_words") != chunk_words
            or row.get("policy_id") != spec.policy_id
        ):
            raise ResearchConfigError(f"policy checkpoint row identity differs: {path.name}")
        rows.append(row)
    if not rows or sum(row.get("aggregation") == "overall" for row in rows) != 1:
        raise ResearchConfigError(f"policy checkpoint aggregate rows are incomplete: {path.name}")
    return _normalized_rows(rows)


def _normalized_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return tuple({key: row[key] for key in sorted(row)} for row in rows)


def _combine_provenance(provenances: Sequence[TraceProvenance]) -> TraceProvenance:
    if not provenances:
        raise ResearchConfigError("checkpointed sweep produced no trace provenance")
    first = provenances[0]
    for provenance in provenances[1:]:
        if (
            provenance.project_git_commit != first.project_git_commit
            or provenance.source_tree_sha256 != first.source_tree_sha256
            or provenance.final_reproducibility_gate != first.final_reproducibility_gate
        ):
            raise ResearchConfigError("checkpointed sweep conditions have mixed provenance")
    return TraceProvenance(
        run_ids=tuple(run_id for item in provenances for run_id in item.run_ids),
        fingerprint_sha256s=tuple(
            digest for item in provenances for digest in item.fingerprint_sha256s
        ),
        project_git_commit=first.project_git_commit,
        source_tree_sha256=first.source_tree_sha256,
        final_reproducibility_gate=first.final_reproducibility_gate,
    )


def _write_progress(
    path: Path,
    *,
    chunk: int,
    completed: int,
    evaluated: int,
    reused: int,
    stage: str,
    policy_id: str | None = None,
    elapsed_seconds: float = 0.0,
) -> None:
    _write_json_atomic(
        path,
        {
            "chunk_words": chunk,
            "completed": completed,
            "evaluated": evaluated,
            "reused": reused,
            "stage": stage,
            "policy_id": policy_id,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def _print_sweep_progress(
    paths: Mapping[int, Path],
    policies_per_chunk: int,
    started: float,
) -> None:
    states: list[Mapping[str, object]] = []
    for chunk in sorted(paths):
        try:
            value = json.loads(paths[chunk].read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            value = {"chunk_words": chunk, "completed": 0, "stage": "starting"}
        states.append(value if isinstance(value, Mapping) else {})
    completed = sum(_progress_integer(state.get("completed")) for state in states)
    evaluated = sum(_progress_integer(state.get("evaluated")) for state in states)
    total = policies_per_chunk * len(paths)
    elapsed = max(0.0, time.monotonic() - started)
    remaining = max(0, total - completed)
    eta = remaining * elapsed / evaluated if evaluated else None
    active = ",".join(
        f"{state.get('chunk_words', '?')}:{state.get('completed', 0)}/{policies_per_chunk}"
        f"({state.get('stage', 'starting')})"
        for state in states
    )
    eta_text = _format_duration(eta) if eta is not None else "pending"
    print(
        f"progress={completed}/{total} ({100.0 * completed / total:.1f}%) "
        f"elapsed={_format_duration(elapsed)} eta~={eta_text} chunks={active}",
        file=sys.stderr,
        flush=True,
    )


def _progress_integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:d}m{secs:02d}s"


def freeze_development_configs(
    config: ResearchConfig,
    overall_rows: Sequence[Mapping[str, object]],
    specs: Sequence[PolicySpec],
    provenance: TraceProvenance,
    *,
    preset: str,
) -> dict[str, object]:
    """Select matched-quality and matched-latency configurations using dev only."""

    rows = tuple(overall_rows)
    by_id = {spec.policy_id: spec for spec in specs}
    if len(by_id) != len(tuple(specs)):
        raise ResearchConfigError("policy specs must have unique IDs")
    for row in rows:
        if row.get("split") != "dev" or row.get("aggregation") != "overall":
            raise ResearchConfigError("freezing accepts only overall development rows")
        if row.get("policy_id") not in by_id:
            raise ResearchConfigError("development row refers to an unknown policy spec")
    reference_config = _mapping(config.selection, "matched_quality_reference")
    reference = _reference_row(rows, reference_config, by_id)
    tolerance = _number(config.selection.get("precision_tolerance"), "precision_tolerance")
    baseline_precision = _row_number(reference, "strict_precision")
    baseline_delay = _row_number(reference, "mean_commit_context_words")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        spec = by_id[str(row["policy_id"])]
        grouped.setdefault(_selection_key(spec), []).append(row)

    matched_quality: dict[str, object] = {}
    matched_latency: dict[str, object] = {}
    reference_spec = by_id[str(reference["policy_id"])]
    for key in sorted(grouped):
        candidates = grouped[key]
        if key == _selection_key(reference_spec):
            quality_choice = latency_choice = reference
        else:
            quality_eligible = [
                row
                for row in candidates
                if _row_number(row, "strict_precision") + tolerance >= baseline_precision
            ]
            quality_pool = quality_eligible or candidates
            quality_choice = min(
                quality_pool,
                key=lambda row: (
                    0 if row in quality_eligible else 1,
                    _row_number(row, "mean_commit_context_words"),
                    _row_number(row, "selection_error_rate"),
                    -_row_number(row, "strict_f1"),
                    str(row["policy_id"]),
                ),
            )
            latency_eligible = [
                row
                for row in candidates
                if _row_number(row, "mean_commit_context_words") <= baseline_delay + tolerance
            ]
            latency_pool = latency_eligible or candidates
            latency_choice = min(
                latency_pool,
                key=lambda row: (
                    0 if row in latency_eligible else 1,
                    _row_number(row, "selection_error_rate"),
                    _row_number(row, "gold_premature_rate"),
                    -_row_number(row, "strict_precision"),
                    _row_number(row, "mean_commit_context_words"),
                    str(row["policy_id"]),
                ),
            )
        matched_quality[key] = _frozen_choice(
            by_id[str(quality_choice["policy_id"])], quality_choice
        )
        matched_latency[key] = _frozen_choice(
            by_id[str(latency_choice["policy_id"])], latency_choice
        )

    lock = config.manifest(preset)
    payload: dict[str, object] = {
        "schema_version": 1,
        "research_config_sha256": config.sha256,
        "preset": preset,
        "development_manifest": {
            "name": lock.name,
            "manifest_sha256": lock.manifest_sha256,
            "file_sha256": lock.file_sha256,
        },
        "selection_split": "dev",
        "dev_only_selection": True,
        "primary_chunk_words": config.primary_chunk_words,
        "trace_provenance": provenance.to_dict(),
        "selection_rules": {
            "matched_quality": "precision>=fixed-threshold-0.5;min_context_delay",
            "matched_latency": "delay<=fixed-threshold-0.5;min_selection_error",
            "precision_tolerance": tolerance,
            "deterministic_ties": True,
        },
        "baseline_reference": _frozen_choice(reference_spec, reference),
        "matched_quality": matched_quality,
        "matched_latency": matched_latency,
    }
    payload["frozen_payload_sha256"] = canonical_sha256(payload)
    return payload


def read_frozen_configs(
    path: str | Path,
    config: ResearchConfig,
    *,
    require_final_gate: bool,
) -> Mapping[str, object]:
    """Read canonical frozen selections without recomputing or retuning them."""

    frozen_path = Path(path)
    try:
        raw = frozen_path.read_bytes()
        value: Any = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchConfigError("frozen config is unreadable") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ResearchConfigError("frozen config must be a canonical JSON object")
    digest = value.pop("frozen_payload_sha256", None)
    if digest != canonical_sha256(value):
        raise ResearchConfigError("frozen config digest differs")
    value["frozen_payload_sha256"] = digest
    if value.get("research_config_sha256") != config.sha256:
        raise ResearchConfigError("frozen config was selected under a different research config")
    if value.get("selection_split") != "dev" or value.get("dev_only_selection") is not True:
        raise ResearchConfigError("frozen config does not prove development-only selection")
    provenance = value.get("trace_provenance")
    if not isinstance(provenance, Mapping):
        raise ResearchConfigError("frozen trace provenance is missing")
    if require_final_gate and provenance.get("final_reproducibility_gate") is not True:
        raise ResearchConfigError("final benchmark requires a commit-gated frozen config")
    return value


def _pareto_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    deployable = [dict(row) for row in rows if row.get("analysis_only") is False]
    error_front = pareto_front(
        deployable,
        minimize=("mean_commit_context_words", "selection_error_rate"),
    )
    f1_front = pareto_front(
        deployable,
        minimize=("mean_commit_context_words",),
        maximize=("strict_f1",),
    )
    result = [dict(row, pareto_objective="delay_vs_error") for row in error_front]
    result.extend(dict(row, pareto_objective="delay_vs_strict_f1") for row in f1_front)
    return tuple(
        sorted(result, key=lambda row: (str(row["pareto_objective"]), str(row["policy_id"])))
    )


def _reference_row(
    rows: Sequence[Mapping[str, object]],
    reference: Mapping[str, object],
    by_id: Mapping[str, PolicySpec],
) -> Mapping[str, object]:
    family = reference.get("policy")
    threshold = _number(reference.get("threshold"), "reference threshold")
    matches = []
    for row in rows:
        spec = by_id[str(row["policy_id"])]
        value = spec.parameters.get("threshold")
        if spec.family == family and isinstance(value, int | float) and float(value) == threshold:
            matches.append(row)
    if len(matches) != 1:
        raise ResearchConfigError("fixed-threshold development reference is missing or ambiguous")
    return matches[0]


def _frozen_choice(spec: PolicySpec, row: Mapping[str, object]) -> dict[str, object]:
    metric_fields = (
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "mean_commit_context_words",
        "selection_error_rate",
        "gold_premature_rate",
        "wrong_commitment_rate",
        "missed_entity_rate",
    )
    return {
        "policy": spec.to_dict(),
        "development_metrics": {field: row[field] for field in metric_fields},
    }


def _selection_key(spec: PolicySpec) -> str:
    return (
        spec.family
        if spec.variant in {"main", "analysis-only"}
        else f"{spec.family}/{spec.variant}"
    )


def _validate_development_conditions(
    config: ResearchConfig,
    conditions: Sequence[TraceCondition],
) -> None:
    if not conditions or any(condition.split != "dev" for condition in conditions):
        raise ResearchConfigError("policy sweep accepts development traces only")
    chunks = {condition.chunk_words for condition in conditions}
    if config.primary_chunk_words not in chunks or not chunks.issubset(set(config.chunk_words)):
        raise ResearchConfigError(
            "development sweep requires the primary configured chunk condition"
        )
    for condition in conditions:
        if {example.example.task_name for example in condition.examples} != set(config.tasks):
            raise ResearchConfigError("development condition does not cover every configured task")


def _write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ResearchConfigError("cannot write an empty Parquet result")
    first_fields = tuple(rows[0])
    extra_fields = tuple(sorted(set().union(*(set(row) for row in rows)) - set(first_fields)))
    fields = (*first_fields, *extra_fields)
    table = pa.Table.from_pylist([{field: row.get(field) for field in fields} for row in rows])
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd", version="2.6", use_dictionary=True)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(canonical_json_bytes(value) + b"\n")
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(row: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ResearchConfigError(f"{field} must be a mapping")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResearchConfigError(f"{name} must be numeric")
    return float(value)


def _row_number(row: Mapping[str, object], field: str) -> float:
    return _number(row.get(field), field)


__all__ = [
    "FROZEN_FILENAME",
    "PARETO_FILENAME",
    "SWEEP_FILENAME",
    "SWEEP_MANIFEST_FILENAME",
    "freeze_development_configs",
    "read_frozen_configs",
    "run_checkpointed_development_sweep",
    "run_development_sweep",
]
