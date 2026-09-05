"""Content-safe orchestration for one-pass PIIMB MLX trace generation.

This module is the narrow boundary between the validated PIIMB source loader,
the live trace generator, and immutable Parquet persistence.  Commitment
policies intentionally do not belong here: completed traces are replay inputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from streamner_commit.datasets.piimb import (
    PIIMB_DATASET_ID,
    PIIMB_REVISION,
    PIIMB_SOURCE_FILE,
    PIIMB_SOURCE_ROW_COUNT,
    PIIMB_SOURCE_SHA256,
    PIIMB_SOURCE_SIZE_BYTES,
    PIIMB_SUBSET,
    PRIMARY_TASKS,
)
from streamner_commit.datasets.piimb_trace import ReconstructedPIIMBTrace
from streamner_commit.serialization import (
    TraceFingerprint,
    TraceRun,
    TraceRunData,
    build_trace_fingerprint,
    canonical_sha256,
    read_trace_run,
    write_trace_run,
)
from streamner_commit.streaming.trace_generation import (
    TRACE_CHUNK_WORDS,
    TraceInputExample,
    generate_condition_traces,
)

TraceSplit = Literal["dev", "test", "both"]

PRESET_MANIFEST_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "smoke": Path("experiments/manifests/piimb_smoke.json"),
        "research-small": Path("experiments/manifests/piimb_research_small.json"),
        "research-full": Path("experiments/manifests/piimb_research_full.json"),
    }
)

_TRACE_METADATA_FIELDS = frozenset(
    {
        "selection_index",
        "benchmark_split",
        "uid",
        "source_row_index",
        "task_name",
        "source_dataset",
        "source_uid",
        "parent_id",
        "sentence_index",
        "language",
        "metadata_sha256",
        "manifest_sha256",
        "preset",
    }
)
_PERSISTED_METADATA_FIELDS = (
    "task_name",
    "uid",
    "source_row_index",
    "parent_id",
    "source_dataset",
    "source_uid",
    "sentence_index",
    "language",
    "metadata_sha256",
)
_RUNTIME_VERSION_DISTRIBUTIONS = MappingProxyType(
    {
        "mlx_version": "mlx",
        "mlx_lm_version": "mlx-lm",
        "transformers_version": "transformers",
        "torch_version": "torch",
        "gliner_version": "gliner",
    }
)


class TracePipelineError(ValueError):
    """Trace inputs or provenance are unsafe or internally inconsistent."""


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TracePipelineError(f"{name} must be a nonblank string")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _identifier(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise TracePipelineError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TracePipelineError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TracePipelineError(f"{name} must be a nonnegative integer")
    return value


def _json_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TracePipelineError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result: Any = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TracePipelineError(f"{name} must contain finite JSON data") from error
    if not isinstance(result, dict):
        raise TracePipelineError(f"{name} must contain a JSON object")
    return MappingProxyType(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_runtime_versions() -> Mapping[str, str | None]:
    versions: dict[str, str | None] = {}
    for field_name, distribution_name in _RUNTIME_VERSION_DISTRIBUTIONS.items():
        try:
            versions[field_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[field_name] = None
    return MappingProxyType(versions)


@dataclass(frozen=True, slots=True)
class TraceModelProvenance:
    """All immutable model/runtime inputs needed for a trace fingerprint."""

    model_id: str
    model_revision_sha: str
    weights_sha256: str
    checkpoint_config_sha256: str
    export_manifest_sha256: str
    tensor_manifest_sha256: str
    tensor_count: int
    parameter_count: int
    model_config: Mapping[str, object] = field(hash=False)
    context_limit: int = 40_960
    right_context_width: int = 12
    backend_name: str = "mlx-streaming"
    device: str = "gpu"
    dtype: str = "float32"
    gliner_reference_tag: str = "0.2.28"
    runtime_versions: Mapping[str, str | None] = field(
        default_factory=_installed_runtime_versions,
        hash=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision_sha",
            "backend_name",
            "device",
            "dtype",
            "gliner_reference_tag",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        for name in (
            "weights_sha256",
            "checkpoint_config_sha256",
            "export_manifest_sha256",
            "tensor_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "tensor_count",
            _positive_int(self.tensor_count, name="tensor_count"),
        )
        object.__setattr__(
            self,
            "parameter_count",
            _positive_int(self.parameter_count, name="parameter_count"),
        )
        object.__setattr__(
            self,
            "context_limit",
            _positive_int(self.context_limit, name="context_limit"),
        )
        object.__setattr__(
            self,
            "right_context_width",
            _nonnegative_int(self.right_context_width, name="right_context_width"),
        )
        object.__setattr__(
            self,
            "model_config",
            _json_mapping(self.model_config, name="model_config"),
        )
        if not isinstance(self.runtime_versions, Mapping):
            raise TracePipelineError("runtime_versions must be a mapping")
        if set(self.runtime_versions) != set(_RUNTIME_VERSION_DISTRIBUTIONS):
            raise TracePipelineError("runtime_versions must contain the complete locked field set")
        normalized_versions: dict[str, str | None] = {}
        for name in _RUNTIME_VERSION_DISTRIBUTIONS:
            value = self.runtime_versions[name]
            normalized_versions[name] = (
                None if value is None else _identifier(value, name=f"runtime_versions.{name}")
            )
        object.__setattr__(self, "runtime_versions", MappingProxyType(normalized_versions))
        if self.dtype != "float32":
            raise TracePipelineError("Phase 11 trace generation requires unquantized float32")

    @classmethod
    def from_asset_bundle(
        cls,
        bundle: Any,
        backend: Any,
        *,
        runtime_versions: Mapping[str, str | None] | None = None,
    ) -> TraceModelProvenance:
        """Derive hash-only provenance from a validated asset bundle and backend."""

        tensors = tuple(getattr(bundle, "tensors", ()))
        tensor_dtypes = {getattr(tensor, "dtype", None) for tensor in tensors}
        if not tensors or tensor_dtypes != {"float32"}:
            raise TracePipelineError("validated asset tensors must all be float32")
        config_path = Path(bundle.config_path)
        export_root = Path(bundle.root)
        tensor_manifest_path = Path(bundle.tensor_manifest_path)
        export_manifest_path = export_root / "export_manifest.json"
        for name, path in (
            ("config", config_path),
            ("export manifest", export_manifest_path),
            ("tensor manifest", tensor_manifest_path),
        ):
            if not path.is_file():
                raise TracePipelineError(f"validated asset {name} is missing")
        checkpoint_config = bundle.config
        effective_config = {
            "checkpoint": checkpoint_config,
            "runtime": {
                "context_limit": backend.context_limit,
                "right_context_width": backend.right_context_width,
                "public_threshold": 0.5,
                "flat_ner": True,
                "multi_label": False,
            },
        }
        return cls(
            model_id=bundle.model_id,
            model_revision_sha=bundle.revision,
            weights_sha256=bundle.weights_sha256,
            checkpoint_config_sha256=_sha256_file(config_path),
            export_manifest_sha256=_sha256_file(export_manifest_path),
            tensor_manifest_sha256=_sha256_file(tensor_manifest_path),
            tensor_count=bundle.tensor_count,
            parameter_count=bundle.parameter_count,
            model_config=effective_config,
            context_limit=backend.context_limit,
            right_context_width=backend.right_context_width,
            runtime_versions=(
                _installed_runtime_versions() if runtime_versions is None else runtime_versions
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceConditionResult:
    """One completed or exactly reused chunk-condition trace run."""

    chunk_words: int
    run: TraceRun
    reused: bool

    @property
    def example_count(self) -> int:
        return len(self.run.data.examples)

    @property
    def step_count(self) -> int:
        return len(self.run.data.steps)

    @property
    def span_update_count(self) -> int:
        return len(self.run.data.span_updates)


@dataclass(frozen=True, slots=True)
class TracePipelineResult:
    """Sanitizable result for one preset/split across requested conditions."""

    preset: str
    split: TraceSplit
    conditions: tuple[TraceConditionResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "conditions", tuple(self.conditions))


def manifest_path_for_preset(project_root: str | Path, preset: str) -> Path:
    """Resolve one checked-in preset manifest without accepting an unknown alias."""

    try:
        relative = PRESET_MANIFEST_PATHS[preset]
    except KeyError as error:
        raise TracePipelineError(
            f"preset must be one of {tuple(PRESET_MANIFEST_PATHS)}"
        ) from error
    return Path(project_root).resolve() / relative


def _normalize_split(value: object) -> TraceSplit:
    if value not in {"dev", "test", "both"}:
        raise TracePipelineError("split must be one of ('dev', 'test', 'both')")
    return value  # type: ignore[return-value]


def _normalize_chunk_conditions(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TracePipelineError("chunk_words must be a sequence")
    conditions = tuple(values)
    if not conditions:
        raise TracePipelineError("at least one chunk condition is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in conditions):
        raise TracePipelineError("chunk conditions must be integers")
    invalid = tuple(value for value in conditions if value not in TRACE_CHUNK_WORDS)
    if invalid:
        raise TracePipelineError(f"chunk conditions must be drawn from {TRACE_CHUNK_WORDS}")
    if len(conditions) != len(set(conditions)):
        raise TracePipelineError("duplicate chunk conditions would rerun the same model condition")
    return conditions


def _expected_metadata(
    reconstructed: ReconstructedPIIMBTrace,
    *,
    selection_index: int,
) -> Mapping[str, object]:
    record = reconstructed.manifest.records[selection_index]
    return {
        "selection_index": selection_index,
        "benchmark_split": record.split,
        **record.source_metadata(),
        "metadata_sha256": record.metadata_sha256,
        "manifest_sha256": reconstructed.manifest.manifest_sha256,
        "preset": reconstructed.manifest.preset.name,
    }


def trace_inputs_from_piimb(
    reconstructed: ReconstructedPIIMBTrace,
    *,
    preset: str,
    split: TraceSplit,
) -> tuple[TraceInputExample, ...]:
    """Validate and convert selected PIIMB rows without changing their gold spans."""

    if not isinstance(reconstructed, ReconstructedPIIMBTrace):
        raise TypeError("reconstructed must be a ReconstructedPIIMBTrace")
    requested_split = _normalize_split(split)
    expected_preset = _identifier(preset, name="preset")
    if reconstructed.manifest.preset.name != expected_preset:
        raise TracePipelineError(
            "loaded manifest preset differs from the explicitly requested preset"
        )
    if len(reconstructed.examples) != len(reconstructed.manifest.records):
        raise TracePipelineError("reconstructed examples and manifest records differ")

    labels_by_task: dict[str, tuple[str, ...]] = {}
    converted: list[TraceInputExample] = []
    for selection_index, (record, example) in enumerate(
        zip(reconstructed.manifest.records, reconstructed.examples, strict=True)
    ):
        if set(example.metadata) != _TRACE_METADATA_FIELDS:
            raise TracePipelineError("PIIMB trace example metadata fields are mixed or incomplete")
        expected_metadata = _expected_metadata(
            reconstructed,
            selection_index=selection_index,
        )
        if dict(example.metadata) != dict(expected_metadata):
            raise TracePipelineError("PIIMB trace example metadata differs from its manifest row")
        if example.split != record.split:
            raise TracePipelineError("PIIMB trace example split differs from its manifest row")
        expected_example_id = (
            f"piimb:{record.split}:{record.task_name}:{record.source_row_index}:{record.uid}"
        )
        if example.example_id != expected_example_id:
            raise TracePipelineError("PIIMB trace example identity differs from its manifest row")
        task_labels = tuple(reconstructed.manifest.task_labels.get(record.task_name, ()))
        if not task_labels or example.labels != task_labels:
            raise TracePipelineError("PIIMB trace example labels differ from its task vocabulary")
        prior_labels = labels_by_task.setdefault(record.task_name, example.labels)
        if prior_labels != example.labels:
            raise TracePipelineError("PIIMB task examples use mixed label orders")
        if requested_split != "both" and record.split != requested_split:
            continue

        persisted_metadata = {
            field_name: expected_metadata[field_name]
            for field_name in _PERSISTED_METADATA_FIELDS
        }
        persisted_metadata["split"] = record.split
        trace_input = TraceInputExample(
            example_id=example.example_id,
            text=example.text,
            labels=example.labels,
            gold_entities=example.gold_entities,
            metadata=persisted_metadata,
        )
        if trace_input.gold_entities != example.gold_entities:
            raise TracePipelineError("PIIMB gold annotations changed during trace conversion")
        converted.append(trace_input)

    if not converted:
        raise TracePipelineError("the requested PIIMB split contains no examples")
    return tuple(converted)


def _safe_output_root(
    project_root: Path,
    *,
    output_root: str | Path | None,
    test_output_root: str | Path | None,
) -> Path:
    if output_root is not None and test_output_root is not None:
        raise TracePipelineError("output_root and test_output_root are mutually exclusive")
    if test_output_root is not None:
        return Path(test_output_root).resolve()

    safe_root = (project_root / "results" / "traces").resolve()
    selected = safe_root if output_root is None else Path(output_root).resolve()
    if not selected.is_relative_to(safe_root):
        raise TracePipelineError("raw traces must remain under the ignored results/traces tree")
    gitignore_path = project_root / ".gitignore"
    try:
        ignored_lines = {
            line.strip().lstrip("/")
            for line in gitignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as error:
        raise TracePipelineError("project .gitignore cannot be verified") from error
    if "results/traces/" not in ignored_lines:
        raise TracePipelineError("project .gitignore must exclude results/traces/")
    return selected


def _validate_backend_provenance(backend: object, provenance: TraceModelProvenance) -> None:
    context_limit = getattr(backend, "context_limit", provenance.context_limit)
    right_context_width = getattr(
        backend,
        "right_context_width",
        provenance.right_context_width,
    )
    if context_limit != provenance.context_limit:
        raise TracePipelineError("backend context limit differs from trace provenance")
    if right_context_width != provenance.right_context_width:
        raise TracePipelineError("backend right-context width differs from trace provenance")


def _fingerprint(
    *,
    examples: tuple[TraceInputExample, ...],
    reconstructed: ReconstructedPIIMBTrace,
    project_root: Path,
    split: TraceSplit,
    chunk_words: int,
    provenance: TraceModelProvenance,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
) -> TraceFingerprint:
    selected_identities = [
        {
            "example_id": example.example_id,
            "uid": example.metadata["uid"],
            "source_row_index": example.metadata["source_row_index"],
            "task_name": example.metadata["task_name"],
            "split": example.metadata["split"],
            "metadata_sha256": example.metadata["metadata_sha256"],
        }
        for example in examples
    ]
    extra_inputs = {
        "asset_bundle": {
            "weights_sha256": provenance.weights_sha256,
            "export_manifest_sha256": provenance.export_manifest_sha256,
            "tensor_manifest_sha256": provenance.tensor_manifest_sha256,
            "tensor_count": provenance.tensor_count,
            "parameter_count": provenance.parameter_count,
        },
        "dataset_source": {
            "logical_path": PIIMB_SOURCE_FILE,
            "size_bytes": PIIMB_SOURCE_SIZE_BYTES,
            "sha256": PIIMB_SOURCE_SHA256,
            "rows": PIIMB_SOURCE_ROW_COUNT,
        },
        "selection": {
            "preset": reconstructed.manifest.preset.name,
            "split": split,
            "manifest_sha256": reconstructed.manifest.manifest_sha256,
            "task_labels_sha256": reconstructed.manifest.task_labels_sha256,
            "selected_identity_sha256": canonical_sha256(selected_identities),
            "example_count": len(examples),
        },
        "inference": {
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32"),
            "threshold": threshold,
            "flat_ner": flat_ner,
            "multi_label": multi_label,
            "context_limit": provenance.context_limit,
            "right_context_width": provenance.right_context_width,
            "decoder_tie_break": "first-label-index",
            "probability_mapping": "sigmoid",
        },
    }
    checkpoint = provenance.model_config.get("checkpoint")
    max_width = checkpoint.get("max_width") if isinstance(checkpoint, Mapping) else None
    return build_trace_fingerprint(
        examples=examples,
        project_root=project_root,
        model_sha=provenance.model_revision_sha,
        backend=provenance.backend_name,
        dtype=provenance.dtype,
        chunk_strategy="maximal-nonwhitespace-units",
        chunk_words=chunk_words,
        model_config=provenance.model_config,
        device=provenance.device,
        runtime_versions=provenance.runtime_versions,
        model_id=provenance.model_id,
        checkpoint_config_sha256=provenance.checkpoint_config_sha256,
        public_threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
        max_width=max_width if isinstance(max_width, int) else None,
        right_context_width=provenance.right_context_width,
        gliner_reference_tag=provenance.gliner_reference_tag,
        dataset_id=PIIMB_DATASET_ID,
        dataset_revision=PIIMB_REVISION,
        dataset_subset=PIIMB_SUBSET,
        dataset_tasks=PRIMARY_TASKS,
        sample_manifest_sha256=reconstructed.manifest.manifest_sha256,
        random_seed=0,
        extra_inputs=extra_inputs,
    )


def run_piimb_trace_pipeline(
    backend: object,
    reconstructed: ReconstructedPIIMBTrace,
    provenance: TraceModelProvenance,
    *,
    project_root: str | Path,
    preset: str,
    split: TraceSplit = "both",
    chunk_words: Sequence[int] = (1,),
    output_root: str | Path | None = None,
    test_output_root: str | Path | None = None,
    threshold: float = 0.5,
    flat_ner: bool = True,
    multi_label: bool = False,
) -> TracePipelineResult:
    """Generate each missing condition once and exactly reuse verified runs."""

    # Import lazily so this policy-free coordinator still preserves the MLX
    # package's process-wide configuration-before-first-matmul invariant.
    from streamner_commit.mlx.precision import require_mlx_full_precision

    require_mlx_full_precision()
    if not isinstance(provenance, TraceModelProvenance):
        raise TypeError("provenance must be a TraceModelProvenance")
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise TracePipelineError("project_root must be an existing directory")
    requested_split = _normalize_split(split)
    conditions = _normalize_chunk_conditions(chunk_words)
    destination_root = _safe_output_root(
        root,
        output_root=output_root,
        test_output_root=test_output_root,
    )
    _validate_backend_provenance(backend, provenance)
    examples = trace_inputs_from_piimb(
        reconstructed,
        preset=preset,
        split=requested_split,
    )

    completed: list[TraceConditionResult] = []
    for condition in conditions:
        fingerprint = _fingerprint(
            examples=examples,
            reconstructed=reconstructed,
            project_root=root,
            split=requested_split,
            chunk_words=condition,
            provenance=provenance,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        existing_path = destination_root / fingerprint.run_id
        if existing_path.exists():
            existing = read_trace_run(
                existing_path,
                expected_fingerprint=fingerprint,
            )
            completed.append(
                TraceConditionResult(chunk_words=condition, run=existing, reused=True)
            )
            continue

        generated = generate_condition_traces(
            backend,  # type: ignore[arg-type]
            examples,
            run_id=fingerprint.run_id,
            chunk_words=condition,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        run_data = TraceRunData.from_generated(generated, include_gold=True)
        persisted = write_trace_run(destination_root, fingerprint, run_data)
        completed.append(
            TraceConditionResult(chunk_words=condition, run=persisted, reused=False)
        )
    return TracePipelineResult(
        preset=reconstructed.manifest.preset.name,
        split=requested_split,
        conditions=tuple(completed),
    )


__all__ = [
    "PRESET_MANIFEST_PATHS",
    "TraceConditionResult",
    "TraceModelProvenance",
    "TracePipelineError",
    "TracePipelineResult",
    "TraceSplit",
    "manifest_path_for_preset",
    "run_piimb_trace_pipeline",
    "trace_inputs_from_piimb",
]
