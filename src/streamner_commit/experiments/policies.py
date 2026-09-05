"""Deterministic expansion and construction of configured policy grids."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from streamner_commit.experiments.config import ResearchConfig, ResearchConfigError
from streamner_commit.policies import (
    EMA,
    FixedLag,
    FixedThreshold,
    OracleStable,
    RescorePatience,
    SnapshotPatience,
    StabilityGate,
    StabilityGateConfig,
)
from streamner_commit.policies.base import CommitmentPolicy
from streamner_commit.serialization import canonical_sha256
from streamner_commit.streaming.tracker import StreamingObservation


@dataclass(frozen=True, slots=True)
class PolicySpec:
    policy_id: str
    family: str
    variant: str
    parameters: Mapping[str, object]
    analysis_only: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "family": self.family,
            "variant": self.variant,
            "parameters": dict(self.parameters),
            "analysis_only": self.analysis_only,
        }


def expand_policy_grid(config: ResearchConfig) -> tuple[PolicySpec, ...]:
    """Expand every required baseline, oracle, StabilityGate, and ablation."""

    specs: list[PolicySpec] = []
    grids = config.policy_grids
    specs.extend(
        _spec("fixed-threshold", "main", {"threshold": threshold})
        for threshold in _numbers(_grid(grids, "fixed_threshold"), "threshold")
    )
    specs.extend(
        _spec("fixed-lag", "main", {"threshold": threshold, "lag_words": lag})
        for threshold, lag in itertools.product(
            _numbers(_grid(grids, "fixed_lag"), "threshold"),
            _integers(_grid(grids, "fixed_lag"), "lag_words"),
        )
    )
    for family, grid_name in (
        ("snapshot-patience", "snapshot_patience"),
        ("rescore-patience", "rescore_patience"),
    ):
        grid = _grid(grids, grid_name)
        specs.extend(
            _spec(family, "main", {"threshold": threshold, "patience": patience})
            for threshold, patience in itertools.product(
                _numbers(grid, "threshold"), _integers(grid, "patience")
            )
        )
    ema_grid = _grid(grids, "ema")
    specs.extend(
        _spec("ema", "main", {"threshold": threshold, "alpha": alpha})
        for threshold, alpha in itertools.product(
            _numbers(ema_grid, "threshold"), _numbers(ema_grid, "alpha")
        )
    )
    specs.extend(_stability_specs(_grid(grids, "stability_gate")))
    specs.extend(
        _spec("oracle-stable", "analysis-only", {"threshold": threshold}, analysis_only=True)
        for threshold in _numbers(_grid(grids, "oracle_stable"), "threshold")
    )
    ordered = tuple(sorted(specs, key=lambda item: item.policy_id))
    if len({spec.policy_id for spec in ordered}) != len(ordered):
        raise ResearchConfigError("expanded policy grid contains duplicate configurations")
    return ordered


def build_policy(
    spec: PolicySpec,
    observations: Sequence[StreamingObservation],
) -> CommitmentPolicy:
    """Construct one fresh policy; model/backend objects are intentionally absent."""

    parameters = dict(spec.parameters)
    if spec.family == "fixed-threshold":
        return FixedThreshold(**parameters)  # type: ignore[arg-type]
    if spec.family == "fixed-lag":
        return FixedLag(**parameters)  # type: ignore[arg-type]
    if spec.family == "snapshot-patience":
        return SnapshotPatience(**parameters)  # type: ignore[arg-type]
    if spec.family == "rescore-patience":
        return RescorePatience(**parameters)  # type: ignore[arg-type]
    if spec.family == "ema":
        return EMA(**parameters)  # type: ignore[arg-type]
    if spec.family == "stability-gate":
        return StabilityGate(StabilityGateConfig.from_mapping(parameters))
    if spec.family == "oracle-stable":
        threshold = parameters.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ResearchConfigError("oracle threshold is malformed")
        return OracleStable(observations, float(threshold))
    raise ResearchConfigError(f"unknown policy family: {spec.family}")


def pilot_policy_specs(config: ResearchConfig) -> tuple[PolicySpec, ...]:
    """Choose one deterministic configuration per family/ablation for smoke runs."""

    expanded = expand_policy_grid(config)
    grouped: dict[str, list[PolicySpec]] = {}
    for spec in expanded:
        key = (
            spec.family
            if spec.variant in {"main", "analysis-only"}
            else f"{spec.family}/{spec.variant}"
        )
        grouped.setdefault(key, []).append(spec)
    result: list[PolicySpec] = []
    for key in sorted(grouped):
        choices = grouped[key]
        if key == "fixed-threshold":
            reference = [spec for spec in choices if spec.parameters.get("threshold") == 0.5]
            if len(reference) != 1:
                raise ResearchConfigError("pilot grid requires fixed-threshold 0.5")
            result.append(reference[0])
        else:
            result.append(min(choices, key=lambda item: item.policy_id))
    return tuple(result)


def policy_spec_from_mapping(value: Mapping[str, object]) -> PolicySpec:
    """Reconstruct an exactly frozen spec without consulting a search grid."""

    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ResearchConfigError("frozen policy parameters must be a mapping")
    spec = _spec(
        _required_string(value, "family"),
        _required_string(value, "variant"),
        dict(parameters),
        analysis_only=bool(value.get("analysis_only", False)),
    )
    if value.get("policy_id") != spec.policy_id:
        raise ResearchConfigError("frozen policy ID does not match its parameters")
    return spec


def _stability_specs(grid: Mapping[str, object]) -> list[PolicySpec]:
    axes = {
        "tau": _numbers(grid, "tau"),
        "min_rescores": _integers(grid, "min_rescores"),
        "instability_horizon": _integers(grid, "instability_horizon"),
        "epsilon": _numbers(grid, "epsilon"),
        "min_label_margin": _numbers(grid, "min_label_margin"),
        "max_extension_advantage": _numbers(grid, "max_extension_advantage"),
    }
    ablations = _strings(grid, "ablations")
    allowed = {"full", "minus_instability", "minus_label_margin", "minus_extension"}
    if set(ablations) != allowed:
        raise ResearchConfigError("StabilityGate must configure all required ablations")
    result: list[PolicySpec] = []
    for variant in ablations:
        epsilon_values = axes["epsilon"] if variant != "minus_instability" else axes["epsilon"][:1]
        margin_values = (
            axes["min_label_margin"]
            if variant != "minus_label_margin"
            else axes["min_label_margin"][:1]
        )
        extension_values = (
            axes["max_extension_advantage"]
            if variant != "minus_extension"
            else axes["max_extension_advantage"][:1]
        )
        for values in itertools.product(
            axes["tau"],
            axes["min_rescores"],
            axes["instability_horizon"],
            epsilon_values,
            margin_values,
            extension_values,
        ):
            tau, min_rescores, horizon, epsilon, margin, extension = values
            parameters: dict[str, object] = {
                "tau": tau,
                "min_rescores": min_rescores,
                "instability_horizon": horizon,
                "epsilon": epsilon,
                "min_label_margin": margin,
                "max_extension_advantage": extension,
                "use_instability": variant != "minus_instability",
                "use_label_margin": variant != "minus_label_margin",
                "use_extension": variant != "minus_extension",
                "min_tail_distance_words": None,
            }
            result.append(_spec("stability-gate", variant, parameters))
    return result


def _spec(
    family: str,
    variant: str,
    parameters: Mapping[str, object],
    *,
    analysis_only: bool = False,
) -> PolicySpec:
    identity = {
        "family": family,
        "variant": variant,
        "parameters": dict(parameters),
        "analysis_only": analysis_only,
    }
    digest = canonical_sha256(identity)
    return PolicySpec(
        policy_id=f"{family}:{variant}:{digest[:16]}",
        family=family,
        variant=variant,
        parameters=dict(parameters),
        analysis_only=analysis_only,
    )


def _grid(grids: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = grids.get(name)
    if not isinstance(value, Mapping):
        raise ResearchConfigError(f"policy grid is malformed: {name}")
    return value


def _sequence(grid: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = grid.get(name)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ResearchConfigError(f"policy grid axis must be a sequence: {name}")
    result = tuple(value)
    if not result:
        raise ResearchConfigError(f"policy grid axis must not be empty: {name}")
    return result


def _numbers(grid: Mapping[str, object], name: str) -> tuple[float, ...]:
    result: list[float] = []
    for value in _sequence(grid, name):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ResearchConfigError(f"policy grid axis must be numeric: {name}")
        result.append(float(value))
    return tuple(result)


def _integers(grid: Mapping[str, object], name: str) -> tuple[int, ...]:
    result = _sequence(grid, name)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise ResearchConfigError(f"policy grid axis must contain integers: {name}")
    return result  # type: ignore[return-value]


def _strings(grid: Mapping[str, object], name: str) -> tuple[str, ...]:
    result = _sequence(grid, name)
    if any(not isinstance(value, str) or not value for value in result):
        raise ResearchConfigError(f"policy grid axis must contain strings: {name}")
    return result  # type: ignore[return-value]


def _required_string(value: Mapping[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ResearchConfigError(f"frozen policy {field} is malformed")
    return item


__all__ = [
    "PolicySpec",
    "build_policy",
    "expand_policy_grid",
    "pilot_policy_specs",
    "policy_spec_from_mapping",
]
