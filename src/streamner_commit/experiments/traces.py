"""Verified trace discovery and reconstruction for inference-free experiments."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from streamner_commit.experiments.config import ResearchConfig
from streamner_commit.serialization import TraceExample, TraceRun, project_version, read_trace_run
from streamner_commit.streaming.tracker import StreamingObservation, build_streaming_observations
from streamner_commit.types import ColdFullResult, GoldEntity, SnapshotStep, SpanScoreUpdate


class ExperimentTraceError(ValueError):
    """Cached traces are absent, mixed, or inconsistent with locked research inputs."""


@dataclass(frozen=True, slots=True)
class ExampleReplay:
    example: TraceExample
    observations: tuple[StreamingObservation, ...]
    snapshots: tuple[SnapshotStep, ...]
    span_updates: tuple[SpanScoreUpdate, ...]
    gold_entities: tuple[GoldEntity, ...]
    cold_full: ColdFullResult


@dataclass(frozen=True, slots=True)
class TraceCondition:
    run: TraceRun
    split: str
    preset: str
    chunk_words: int
    examples: tuple[ExampleReplay, ...]


@dataclass(frozen=True, slots=True)
class TraceProvenance:
    run_ids: tuple[str, ...]
    fingerprint_sha256s: tuple[str, ...]
    project_git_commit: str | None
    source_tree_sha256: str | None
    final_reproducibility_gate: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "run_ids": list(self.run_ids),
            "fingerprint_sha256s": list(self.fingerprint_sha256s),
            "project_git_commit": self.project_git_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "final_reproducibility_gate": self.final_reproducibility_gate,
        }


def load_trace_conditions(
    trace_root: str | Path,
    config: ResearchConfig,
    *,
    preset: str,
    split: str,
    chunk_words: Sequence[int] | None = None,
    rolling_window: int | None = None,
    required_project_commit: str | None = None,
) -> tuple[TraceCondition, ...]:
    """Read fully verified TraceRun directories and select one per condition."""

    if split not in {"dev", "test"}:
        raise ExperimentTraceError("experiment split must be dev or test")
    lock = config.manifest(preset)
    chunks = config.chunk_words if chunk_words is None else tuple(chunk_words)
    if (
        not chunks
        or len(set(chunks)) != len(chunks)
        or any(chunk not in config.chunk_words for chunk in chunks)
    ):
        raise ExperimentTraceError("requested chunk conditions are invalid")
    root = Path(trace_root)
    if not root.is_dir():
        raise ExperimentTraceError("trace root does not exist")
    horizon = rolling_window or _maximum_instability_horizon(config)

    selected: dict[int, TraceCondition] = {}
    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("trace-")
    )
    for path in candidates:
        if not _manifest_may_match(
            path,
            preset=preset,
            split=split,
            chunks=chunks,
            required_project_commit=required_project_commit,
        ):
            continue
        run = read_trace_run(path)
        selection = _selection(run)
        run_preset = _string(selection, "preset")
        run_split = _string(selection, "split")
        chunk = _integer(run.fingerprint.payload.get("chunk_words"), "chunk_words")
        if run_preset != preset or run_split != split or chunk not in chunks:
            continue
        if (
            required_project_commit is not None
            and run.fingerprint.payload.get("project_git_commit") != required_project_commit
        ):
            continue
        _validate_lock(run, config, lock.manifest_sha256, split=split, preset=preset)
        if chunk in selected:
            raise ExperimentTraceError(
                f"multiple verified trace runs exist for chunk_words={chunk}"
            )
        selected[chunk] = TraceCondition(
            run=run,
            split=split,
            preset=preset,
            chunk_words=chunk,
            examples=_reconstruct_examples(run, config, split=split, rolling_window=horizon),
        )
    missing = sorted(set(chunks) - set(selected))
    if missing:
        raise ExperimentTraceError(f"verified trace conditions are missing chunk sizes: {missing}")
    return tuple(selected[chunk] for chunk in sorted(selected))


def _manifest_may_match(
    path: Path,
    *,
    preset: str,
    split: str,
    chunks: Sequence[int],
    required_project_commit: str | None,
) -> bool:
    """Cheaply reject unrelated runs before expensive full trace verification."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    if not isinstance(manifest, Mapping):
        return True
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, Mapping):
        return True
    extra = fingerprint.get("extra_inputs")
    if not isinstance(extra, Mapping) or not isinstance(extra.get("selection"), Mapping):
        return True
    selection = extra["selection"]
    assert isinstance(selection, Mapping)
    selectors = (
        selection.get("preset"),
        selection.get("split"),
        fingerprint.get("chunk_words"),
    )
    if not isinstance(selectors[0], str) or not isinstance(selectors[1], str):
        return True
    if isinstance(selectors[2], bool) or not isinstance(selectors[2], int):
        return True
    if selectors[0] != preset or selectors[1] != split or selectors[2] not in chunks:
        return False
    return (
        required_project_commit is None
        or fingerprint.get("project_git_commit") == required_project_commit
    )


def trace_provenance(
    conditions: Sequence[TraceCondition],
    *,
    pilot: bool,
    project_root: str | Path,
) -> TraceProvenance:
    """Enforce the project-commit gate and return sanitized trace identity."""

    if not conditions:
        raise ExperimentTraceError("at least one trace condition is required")
    commits = {
        condition.run.fingerprint.payload.get("project_git_commit") for condition in conditions
    }
    trees = {
        condition.run.fingerprint.payload.get("source_tree_sha256") for condition in conditions
    }
    if len(commits) != 1 or len(trees) != 1:
        raise ExperimentTraceError("trace conditions have mixed project provenance")
    commit = next(iter(commits))
    tree = next(iter(trees))
    if commit is not None and not isinstance(commit, str):
        raise ExperimentTraceError("trace project commit is malformed")
    if tree is not None and not isinstance(tree, str):
        raise ExperimentTraceError("trace source-tree digest is malformed")
    final_gate = not pilot
    if final_gate:
        if commit is None:
            raise ExperimentTraceError(
                "final held-out evaluation requires traces with project_git_commit; "
                "use --pilot only for nonclaim validation"
            )
        if tree is not None:
            raise ExperimentTraceError(
                "final held-out evaluation rejects traces generated from a dirty source tree"
            )
        current_commit, _tree = project_version(project_root)
        if current_commit != commit:
            raise ExperimentTraceError("current project commit differs from trace project commit")
        if _working_tree_is_dirty(project_root):
            raise ExperimentTraceError("final evaluation requires a clean project working tree")
    elif commit is None and tree is None:
        raise ExperimentTraceError(
            "pilot traces require a source-tree digest when no commit exists"
        )
    return TraceProvenance(
        run_ids=tuple(condition.run.fingerprint.run_id for condition in conditions),
        fingerprint_sha256s=tuple(condition.run.fingerprint.sha256 for condition in conditions),
        project_git_commit=commit,
        source_tree_sha256=tree,
        final_reproducibility_gate=final_gate,
    )


def _reconstruct_examples(
    run: TraceRun,
    config: ResearchConfig,
    *,
    split: str,
    rolling_window: int,
) -> tuple[ExampleReplay, ...]:
    examples = _typed_tuple(run.data.examples, TraceExample, "examples")
    snapshots = _typed_tuple(run.data.steps, SnapshotStep, "steps")
    updates = _typed_tuple(run.data.span_updates, SpanScoreUpdate, "span_updates")
    cold = _typed_tuple(run.data.cold_full, ColdFullResult, "cold_full")
    if run.data.gold_entities is None:
        raise ExperimentTraceError("research traces must contain gold entities")
    gold = _typed_tuple(run.data.gold_entities, GoldEntity, "gold_entities")

    by_example_steps = _group_by_example(snapshots)
    by_example_updates = _group_by_example(updates)
    by_example_gold = _group_by_example(gold)
    cold_by_id = _unique_by_example(cold, "cold_full")
    result: list[ExampleReplay] = []
    for example in sorted(examples, key=lambda item: item.example_id):
        if example.split != split:
            raise ExperimentTraceError("trace example split differs from requested split")
        if example.task_name not in config.tasks or example.parent_id is None:
            raise ExperimentTraceError("trace example lacks a locked task or parent_id")
        example_steps = tuple(
            sorted(by_example_steps.pop(example.example_id, ()), key=lambda item: item.step)
        )
        example_updates = tuple(
            sorted(
                by_example_updates.pop(example.example_id, ()),
                key=lambda item: (item.step, item.start_word, item.end_word),
            )
        )
        example_gold = tuple(
            sorted(
                by_example_gold.pop(example.example_id, ()),
                key=lambda item: (item.start_char, item.end_char, item.label),
            )
        )
        cold_result = cold_by_id.pop(example.example_id, None)
        if not example_steps or cold_result is None or cold_result.full_text != example.text:
            raise ExperimentTraceError("trace example is incomplete or cold text differs")
        if example_steps[-1].accumulated_text != example.text:
            raise ExperimentTraceError("streaming trace does not reconstruct the full text")
        result.append(
            ExampleReplay(
                example=example,
                observations=build_streaming_observations(
                    example_steps,
                    example_updates,
                    example.labels,
                    rolling_window=rolling_window,
                ),
                snapshots=example_steps,
                span_updates=example_updates,
                gold_entities=example_gold,
                cold_full=cold_result,
            )
        )
    if by_example_steps or by_example_updates or by_example_gold or cold_by_id:
        raise ExperimentTraceError("trace tables contain records for unknown examples")
    return tuple(result)


def _validate_lock(
    run: TraceRun,
    config: ResearchConfig,
    manifest_sha256: str,
    *,
    split: str,
    preset: str,
) -> None:
    payload = run.fingerprint.payload
    expected = {
        "model_sha": config.model["revision"],
        "dataset_id": config.dataset["id"],
        "dataset_revision": config.dataset["revision"],
        "dataset_subset": config.dataset["subset"],
        "sample_manifest_sha256": manifest_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ExperimentTraceError(f"trace fingerprint differs from research lock: {field}")
    dataset_tasks = payload.get("dataset_tasks")
    if not isinstance(dataset_tasks, Sequence) or isinstance(dataset_tasks, str | bytes):
        raise ExperimentTraceError("trace task set is malformed")
    if tuple(dataset_tasks) != tuple(sorted(config.tasks)):
        raise ExperimentTraceError("trace task set differs from research lock")
    selection = _selection(run)
    if selection.get("split") != split or selection.get("preset") != preset:
        raise ExperimentTraceError("trace selection split/preset differs")
    if selection.get("manifest_sha256") != manifest_sha256:
        raise ExperimentTraceError("trace selection manifest differs")
    extra = payload.get("extra_inputs")
    if not isinstance(extra, Mapping):
        raise ExperimentTraceError("trace extra_inputs are malformed")
    assets = extra.get("asset_bundle")
    source = extra.get("dataset_source")
    if not isinstance(assets, Mapping) or not isinstance(source, Mapping):
        raise ExperimentTraceError("trace asset/source provenance is missing")
    asset_fields = (
        "weights_sha256",
        "export_manifest_sha256",
        "tensor_manifest_sha256",
    )
    for field in asset_fields:
        if assets.get(field) != config.model[field]:
            raise ExperimentTraceError(f"trace asset digest differs: {field}")
    if (
        source.get("sha256") != config.dataset["source_sha256"]
        or source.get("rows") != config.dataset["source_rows"]
    ):
        raise ExperimentTraceError("trace dataset source lock differs")
    run_config = run.manifest.get("run_config")
    if not isinstance(run_config, Mapping):
        raise ExperimentTraceError("trace run_config is missing")
    if (
        run_config.get("model_id") != config.model["id"]
        or run_config.get("model_revision_sha") != config.model["revision"]
    ):
        raise ExperimentTraceError("trace model identity differs")
    for field in ("checkpoint_config_sha256", "model_config_sha256"):
        if run_config.get(field) != config.model[field]:
            raise ExperimentTraceError(f"trace model digest differs: {field}")


def _selection(run: TraceRun) -> Mapping[str, object]:
    extra = run.fingerprint.payload.get("extra_inputs")
    if not isinstance(extra, Mapping) or not isinstance(extra.get("selection"), Mapping):
        raise ExperimentTraceError("trace selection provenance is missing")
    selection = extra["selection"]
    assert isinstance(selection, Mapping)
    return selection


def _maximum_instability_horizon(config: ResearchConfig) -> int:
    stability = config.policy_grids.get("stability_gate")
    if not isinstance(stability, Mapping):
        raise ExperimentTraceError("stability_gate grid is malformed")
    values = stability.get("instability_horizon")
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ExperimentTraceError("instability_horizon grid is malformed")
    horizons = tuple(_integer(value, "instability_horizon") for value in values)
    return max(horizons, default=3)


def _group_by_example[T](values: Sequence[T]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = {}
    for value in values:
        example_id = getattr(value, "example_id", None)
        if not isinstance(example_id, str):
            raise ExperimentTraceError("trace record lacks example_id")
        grouped.setdefault(example_id, []).append(value)
    return grouped


def _unique_by_example(values: Sequence[ColdFullResult], name: str) -> dict[str, ColdFullResult]:
    result: dict[str, ColdFullResult] = {}
    for value in values:
        if value.example_id in result:
            raise ExperimentTraceError(f"{name} contains duplicate examples")
        result[value.example_id] = value
    return result


def _typed_tuple[T](values: Sequence[object], expected: type[T], name: str) -> tuple[T, ...]:
    rows = tuple(values)
    if not all(isinstance(row, expected) for row in rows):
        raise ExperimentTraceError(f"{name} contains unexpected record types")
    return tuple(row for row in rows if isinstance(row, expected))


def _string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ExperimentTraceError(f"trace selection {field} is malformed")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentTraceError(f"{name} must be a nonnegative integer")
    return value


def _working_tree_is_dirty(project_root: str | Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(project_root).resolve()), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ExperimentTraceError("cannot verify project working-tree state") from error
    return _status_has_non_analysis_changes(result.stdout)


def _status_has_non_analysis_changes(status: str) -> bool:
    for line in status.splitlines():
        # Development selection necessarily rewrites these generated artifacts
        # before the frozen held-out run. They are checksum-verified inputs to
        # that run, not implementation changes. Every other tracked/untracked
        # path still fails the final clean-source gate.
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", maxsplit=1)[1]
        if not path.startswith("results/analysis/"):
            return True
    return False


__all__ = [
    "ExampleReplay",
    "ExperimentTraceError",
    "TraceCondition",
    "TraceProvenance",
    "load_trace_conditions",
    "trace_provenance",
]
