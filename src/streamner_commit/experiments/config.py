"""Strict loading for the checked-in offline research configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]

from streamner_commit.serialization import canonical_sha256


class ResearchConfigError(ValueError):
    """A research configuration is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ManifestLock:
    name: str
    path: str
    manifest_sha256: str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Validated JSON-compatible configuration and commonly used fields."""

    path: Path
    payload: Mapping[str, object]
    sha256: str
    model: Mapping[str, object]
    dataset: Mapping[str, object]
    metrics: Mapping[str, object]
    policy_grids: Mapping[str, object]
    selection: Mapping[str, object]
    bootstrap: Mapping[str, object]
    tasks: tuple[str, ...]
    chunk_words: tuple[int, ...]
    primary_chunk_words: int
    experiment_seed: int
    development_manifest: str
    held_out_manifest: str
    manifests: Mapping[str, ManifestLock]

    def manifest(self, name: str) -> ManifestLock:
        try:
            return self.manifests[name]
        except KeyError as error:
            raise ResearchConfigError(f"unknown manifest preset: {name}") from error


def load_research_config(path: str | Path) -> ResearchConfig:
    """Load finite YAML, reject ambiguous fields, and expose a canonical digest."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ResearchConfigError("research config is unreadable") from error
    payload = _json_mapping(raw, name="research config")
    if payload.get("schema_version") != 1:
        raise ResearchConfigError("research config schema_version must be 1")

    model = _mapping(payload, "model")
    for field in (
        "id",
        "revision",
        "weights_sha256",
        "checkpoint_config_sha256",
        "export_manifest_sha256",
        "tensor_manifest_sha256",
        "model_config_sha256",
    ):
        value = _string(model, field)
        if field.endswith("sha256"):
            _sha256(value, name=f"model.{field}")

    dataset = _mapping(payload, "dataset")
    for field in ("id", "subset", "revision", "source_sha256"):
        value = _string(dataset, field)
        if field == "source_sha256":
            _sha256(value, name="dataset.source_sha256")
    _positive_int(dataset.get("source_rows"), name="dataset.source_rows")
    tasks = _string_tuple(dataset.get("tasks"), name="dataset.tasks")
    if len(set(tasks)) != len(tasks):
        raise ResearchConfigError("dataset.tasks must be unique")

    raw_manifests = _mapping(dataset, "manifests")
    manifests: dict[str, ManifestLock] = {}
    for name in sorted(raw_manifests):
        if not isinstance(name, str) or not name.strip():
            raise ResearchConfigError("manifest names must be nonblank strings")
        entry = raw_manifests[name]
        if not isinstance(entry, Mapping):
            raise ResearchConfigError(f"dataset.manifests.{name} must be a mapping")
        manifests[name] = ManifestLock(
            name=name,
            path=_string(entry, "path"),
            manifest_sha256=_sha256(
                _string(entry, "manifest_sha256"), name=f"manifest {name} canonical digest"
            ),
            file_sha256=_sha256(_string(entry, "file_sha256"), name=f"manifest {name} file digest"),
        )
    project_root = config_path.resolve().parent.parent
    for lock in manifests.values():
        manifest_path = (project_root / lock.path).resolve()
        if not manifest_path.is_relative_to(project_root):
            raise ResearchConfigError("manifest paths must remain inside the project")
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest_payload = json.loads(manifest_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResearchConfigError(f"locked manifest is unreadable: {lock.name}") from error
        if hashlib.sha256(manifest_bytes).hexdigest() != lock.file_sha256:
            raise ResearchConfigError(f"locked manifest file digest differs: {lock.name}")
        if not isinstance(manifest_payload, Mapping):
            raise ResearchConfigError(f"locked manifest is not an object: {lock.name}")
        checksums = manifest_payload.get("checksums")
        if (
            not isinstance(checksums, Mapping)
            or checksums.get("manifest_sha256") != lock.manifest_sha256
        ):
            raise ResearchConfigError(f"locked manifest canonical digest differs: {lock.name}")
    development_manifest = _string(dataset, "development_manifest")
    held_out_manifest = _string(dataset, "held_out_manifest")
    if development_manifest not in manifests or held_out_manifest not in manifests:
        raise ResearchConfigError("development/held-out manifest names must be locked")

    conditions = _mapping(payload, "conditions")
    chunk_words = _int_tuple(conditions.get("chunk_words"), name="conditions.chunk_words")
    if len(set(chunk_words)) != len(chunk_words) or any(
        value not in {1, 2, 4, 8} for value in chunk_words
    ):
        raise ResearchConfigError("conditions.chunk_words must be unique values from 1,2,4,8")
    primary = _positive_int(
        conditions.get("primary_chunk_words"), name="conditions.primary_chunk_words"
    )
    if primary not in chunk_words:
        raise ResearchConfigError("primary chunk size must appear in chunk_words")

    metrics = _mapping(payload, "metrics")
    policy_grids = _mapping(payload, "policy_grids")
    required_policies = {
        "fixed_threshold",
        "fixed_lag",
        "snapshot_patience",
        "rescore_patience",
        "ema",
        "stability_gate",
        "oracle_stable",
    }
    if set(policy_grids) != required_policies:
        raise ResearchConfigError("policy_grids must contain every required policy exactly once")
    selection = _mapping(payload, "selection")
    bootstrap = _mapping(payload, "bootstrap")
    _positive_int(bootstrap.get("replicates"), name="bootstrap.replicates")
    confidence = _finite_float(bootstrap.get("confidence"), name="bootstrap.confidence")
    if not 0.0 < confidence < 1.0:
        raise ResearchConfigError("bootstrap.confidence must be in (0,1)")
    _nonnegative_int(bootstrap.get("seed"), name="bootstrap.seed")
    experiment_seed = _nonnegative_int(payload.get("experiment_seed"), name="experiment_seed")

    return ResearchConfig(
        path=config_path.resolve(),
        payload=payload,
        sha256=canonical_sha256(payload),
        model=model,
        dataset=dataset,
        metrics=metrics,
        policy_grids=policy_grids,
        selection=selection,
        bootstrap=bootstrap,
        tasks=tasks,
        chunk_words=chunk_words,
        primary_chunk_words=primary,
        experiment_seed=experiment_seed,
        development_manifest=development_manifest,
        held_out_manifest=held_out_manifest,
        manifests=MappingProxyType(manifests),
    )


def _json_mapping(value: object, *, name: str) -> Mapping[str, object]:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        normalized: Any = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ResearchConfigError(f"{name} must contain finite JSON-compatible values") from error
    if not isinstance(normalized, dict):
        raise ResearchConfigError(f"{name} must be a mapping")
    return MappingProxyType(normalized)


def _mapping(row: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise ResearchConfigError(f"{field} must be a mapping")
    return value


def _string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResearchConfigError(f"{field} must be a nonblank string")
    return value


def _sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ResearchConfigError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ResearchConfigError(f"{name} must be a sequence")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise ResearchConfigError(f"{name} must contain nonblank strings")
    return result


def _int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ResearchConfigError(f"{name} must be a sequence")
    result = tuple(_positive_int(item, name=name) for item in value)
    if not result:
        raise ResearchConfigError(f"{name} must not be empty")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result < 1:
        raise ResearchConfigError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchConfigError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResearchConfigError(f"{name} must be numeric")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ResearchConfigError(f"{name} must be finite")
    return result


__all__ = [
    "ManifestLock",
    "ResearchConfig",
    "ResearchConfigError",
    "load_research_config",
]
