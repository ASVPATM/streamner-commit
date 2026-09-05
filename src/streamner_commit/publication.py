"""Deterministic Phase 18 figures and tables from sanitized aggregate artifacts."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PILOT_LABEL = "PILOT — NOT A RESEARCH RESULT"
SYNTHETIC_TRAJECTORY_PREFIXES = (
    "Sarah",
    "Sarah Johnson",
    "Sarah Johnson arrived",
    "Sarah Johnson arrived today",
)


class PublicationError(ValueError):
    """A required aggregate artifact is absent, malformed, or ambiguous."""


@dataclass(frozen=True, slots=True)
class PublicationStatus:
    status: str
    mode: str
    final_reproducibility_gate: bool

    @property
    def pilot(self) -> bool:
        return not self.final_reproducibility_gate

    @property
    def table_value(self) -> str:
        return PILOT_LABEL if self.pilot else "FINAL"


def _json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise PublicationError(f"required {name} is absent: {source}")
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"{name} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise PublicationError(f"{name} must contain a JSON object")
    return value


def load_publication_status(path: str | Path) -> PublicationStatus:
    value = _json_object(path, name="benchmark manifest")
    status = value.get("status")
    mode = value.get("mode")
    gate = value.get("final_reproducibility_gate")
    if not isinstance(status, str) or not status.strip():
        raise PublicationError("benchmark manifest status must be nonblank")
    if not isinstance(mode, str) or not mode.strip():
        raise PublicationError("benchmark manifest mode must be nonblank")
    if not isinstance(gate, bool):
        raise PublicationError("benchmark manifest final_reproducibility_gate must be boolean")
    return PublicationStatus(status, mode, gate)


def _parquet(path: str | Path, *, name: str, columns: Sequence[str]) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise PublicationError(f"required {name} is absent: {source}")
    try:
        frame = pd.read_parquet(source)
    except Exception as error:
        raise PublicationError(f"{name} is unreadable Parquet") from error
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PublicationError(f"{name} is missing columns: {missing}")
    if frame.empty:
        raise PublicationError(f"{name} contains no rows")
    return frame


def _finite(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise PublicationError(f"{name}.{column} must be numeric") from error
        if not np.isfinite(values).all():
            raise PublicationError(f"{name}.{column} must be finite")


def _rows(
    frame: pd.DataFrame,
    *,
    split: str,
    chunk_words: int | None = None,
    selection_mode: str | None = None,
) -> pd.DataFrame:
    selected = frame[(frame["split"] == split) & (frame["aggregation"] == "overall")]
    if chunk_words is not None:
        selected = selected[selected["chunk_words"] == chunk_words]
    if selection_mode is not None:
        selected = selected[selected["selection_mode"] == selection_mode]
    if "analysis_only" in selected:
        selected = selected[selected["analysis_only"] == False]  # noqa: E712
    return selected.copy()


_POLICY_NAMES = {
    "ema": "EMA",
    "fixed-lag": "Fixed lag",
    "fixed-threshold": "Fixed threshold",
    "rescore-patience": "Rescore patience",
    "snapshot-patience": "Snapshot patience",
    "stability-gate": "StabilityGate",
}


def _policy_name(family: object) -> str:
    if not isinstance(family, str) or family not in _POLICY_NAMES:
        raise PublicationError(f"unknown deployable policy family: {family!r}")
    return _POLICY_NAMES[family]


def _require_policy_coverage(frame: pd.DataFrame, *, name: str) -> None:
    families = set(frame["policy_family"])
    missing = {"fixed-threshold", "fixed-lag", "stability-gate"} - families
    if not families.intersection({"snapshot-patience", "rescore-patience"}):
        missing.add("patience")
    if missing:
        raise PublicationError(f"{name} lacks required policy families: {sorted(missing)}")


def _main_policy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["policy_variant"] == "main")
        | ((frame["policy_family"] == "stability-gate") & (frame["policy_variant"] == "full"))
    ].copy()


def _label_figure(figure: Figure, status: PublicationStatus) -> None:
    if status.pilot:
        figure.text(
            0.5,
            0.995,
            PILOT_LABEL,
            ha="center",
            va="top",
            color="#a00000",
            fontsize=10,
            fontweight="bold",
        )


def _save_figure(figure: Figure, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("png", "pdf"):
        target = destination / f"{figure.get_label()}.{suffix}"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=f".{suffix}", dir=destination
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            metadata: dict[str, object] = {"Creator": "StreamNER-Commit"}
            if suffix == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            figure.savefig(
                temporary,
                format=suffix,
                dpi=160,
                bbox_inches="tight",
                metadata=metadata,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(target)
    plt.close(figure)
    return outputs[0], outputs[1]


def _figure1(pareto: pd.DataFrame, status: PublicationStatus, primary_chunk: int) -> Figure:
    required = (
        "split",
        "chunk_words",
        "aggregation",
        "analysis_only",
        "policy_family",
        "pareto_objective",
        "mean_commit_context_words",
        "strict_f1",
    )
    missing = sorted(set(required) - set(pareto.columns))
    if missing:
        raise PublicationError(f"development Pareto input is missing columns: {missing}")
    selected = _rows(pareto, split="dev", chunk_words=primary_chunk)
    selected = selected[selected["pareto_objective"] == "delay_vs_strict_f1"]
    if selected.empty:
        raise PublicationError("development Pareto input has no delay_vs_strict_f1 rows")
    _finite(selected, ("mean_commit_context_words", "strict_f1"), name="dev Pareto")
    _require_policy_coverage(selected, name="development Pareto input")

    figure, axis = plt.subplots(figsize=(7.2, 4.8), num="figure1_pareto")
    markers = ("o", "s", "^", "D", "P", "X")
    for marker, family in zip(markers, sorted(set(selected["policy_family"])), strict=False):
        rows = selected[selected["policy_family"] == family].sort_values(
            ["mean_commit_context_words", "strict_f1"]
        )
        axis.plot(
            rows["mean_commit_context_words"],
            rows["strict_f1"],
            marker=marker,
            linewidth=1.2,
            label=_policy_name(family),
        )
    axis.set_title("Development Pareto frontier: strict F1 vs commitment delay")
    axis.set_xlabel("Mean right-context words at commitment (lower is earlier)")
    axis.set_ylabel("Strict committed F1")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    _label_figure(figure, status)
    return figure


def _figure2(horizons: pd.DataFrame, status: PublicationStatus, primary_chunk: int) -> Figure:
    selected = horizons[(horizons["split"] == "test") & (horizons["chunk_words"] == primary_chunk)]
    if selected.empty:
        raise PublicationError("revision horizon input has no primary held-out rows")
    _finite(selected, ("revision_horizon_words",), name="revision horizons")
    values = selected["revision_horizon_words"].to_numpy(dtype=float)
    lower, upper = math.floor(float(values.min())), math.ceil(float(values.max()))
    bins = np.arange(lower - 0.5, upper + 1.5, 1.0)
    figure, axis = plt.subplots(figsize=(7.2, 4.5), num="figure2_revision_horizon")
    axis.hist(values, bins=bins.tolist(), color="#4c78a8", edgecolor="white")
    axis.set_title("Held-out revision-horizon distribution")
    axis.set_xlabel("Words of visible context after span end at last rescore")
    axis.set_ylabel("Span count")
    axis.grid(axis="y", alpha=0.25)
    _label_figure(figure, status)
    return figure


def _figure3(status: PublicationStatus) -> Figure:
    steps = np.arange(1, 5)
    short_probability = (0.72, 0.60, 0.39, 0.22)
    long_probability = (np.nan, 0.66, 0.84, 0.88)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), num="figure3_synthetic_trajectory")
    axis.plot(steps, short_probability, marker="o", label="Sarah")
    axis.plot(steps, long_probability, marker="s", label="Sarah Johnson")
    axis.axhline(0.5, color="0.45", linestyle=":", linewidth=1, label="0.5 threshold")
    axis.axvline(1, color="#e45756", linestyle="--", linewidth=1.2, label="Fixed threshold commits")
    axis.axvline(3, color="#54a24b", linestyle="--", linewidth=1.2, label="StabilityGate commits")
    axis.set_xticks(steps, SYNTHETIC_TRAJECTORY_PREFIXES, rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Illustrative person probability")
    axis.set_title("Fully synthetic trajectory: Sarah → Sarah Johnson")
    axis.text(
        0.01,
        0.02,
        "Fictional illustration; probabilities are not benchmark measurements.",
        transform=axis.transAxes,
        fontsize=8,
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    _label_figure(figure, status)
    return figure


def _agreement_fraction(numerator: object, denominator: object, *, name: str) -> float:
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int | float)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int | float)
        or float(denominator) <= 0.0
    ):
        raise PublicationError(f"{name} counts must be positive numeric values")
    value = float(numerator) / float(denominator)
    if not 0.0 <= value <= 1.0:
        raise PublicationError(f"{name} agreement must be in [0,1]")
    return value


def _parity_matrix(stream_path: Path, piimb_path: Path) -> tuple[np.ndarray, tuple[str, str]]:
    stream = _json_object(stream_path, name="streaming parity report")
    piimb = _json_object(piimb_path, name="PIIMB parity report")
    conditions = stream.get("conditions")
    numerical = stream.get("numerical")
    totals = stream.get("totals")
    if (
        not isinstance(conditions, list)
        or not isinstance(numerical, Mapping)
        or not isinstance(totals, Mapping)
    ):
        raise PublicationError("streaming parity report has an invalid aggregate schema")
    steps: list[Mapping[str, Any]] = []
    for condition in conditions:
        if not isinstance(condition, Mapping) or not isinstance(condition.get("cases"), list):
            raise PublicationError("streaming parity conditions are malformed")
        for case in condition["cases"]:
            if not isinstance(case, Mapping) or not isinstance(case.get("steps"), list):
                raise PublicationError("streaming parity cases are malformed")
            steps.extend(step for step in case["steps"] if isinstance(step, Mapping))
    if not steps or len(steps) != totals.get("step_count"):
        raise PublicationError("streaming parity step total is inconsistent")
    structure = sum(
        bool(step.get("exact"))
        and isinstance(step["exact"], Mapping)
        and all(value is True for value in step["exact"].values())
        for step in steps
    ) / len(steps)
    decoded = sum(
        isinstance(step.get("decoded"), Mapping) and step["decoded"].get("identity_match") is True
        for step in steps
    ) / len(steps)
    top = _agreement_fraction(
        numerical.get("top_label_matches"),
        numerical.get("top_label_total"),
        name="streaming top-label",
    )

    counts = piimb.get("counts")
    metrics = piimb.get("metrics")
    if not isinstance(counts, Mapping) or not isinstance(metrics, Mapping):
        raise PublicationError("PIIMB parity report has an invalid aggregate schema")
    piimb_values = tuple(
        float(metrics.get(field, -1.0))
        for field in (
            "structural_case_agreement",
            "candidate_top_label_agreement",
            "decoded_identity_case_agreement",
        )
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in piimb_values):
        raise PublicationError("PIIMB parity agreement metrics must be finite fractions")
    labels = (
        f"Synthetic streaming\n{int(numerical['top_label_total']):,} vectors",
        f"PIIMB smoke\n{int(counts.get('candidate_vectors', 0)):,} vectors",
    )
    return np.array(((structure, top, decoded), piimb_values), dtype=float), labels


def _figure4(
    stream_path: Path,
    piimb_path: Path,
    status: PublicationStatus,
) -> Figure:
    values, labels = _parity_matrix(stream_path, piimb_path)
    figure, axis = plt.subplots(figsize=(7.2, 3.8), num="figure4_mlx_parity")
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    axis.set_xticks(range(3), ("Structure", "Top label", "Decoded identity"))
    axis.set_yticks(range(2), labels)
    for row in range(2):
        for column in range(3):
            axis.text(column, row, f"{100.0 * values[row, column]:.2f}%", ha="center", va="center")
    axis.set_title("PyTorch reference vs native MLX agreement")
    figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04, label="Agreement")
    _label_figure(figure, status)
    return figure


def generate_figures(
    *,
    pareto_path: str | Path,
    revision_horizons_path: str | Path,
    benchmark_manifest_path: str | Path,
    streaming_parity_path: str | Path,
    piimb_parity_path: str | Path,
    output_dir: str | Path,
    primary_chunk: int = 1,
) -> tuple[Path, ...]:
    status = load_publication_status(benchmark_manifest_path)
    pareto = _parquet(pareto_path, name="development Pareto result", columns=("split",))
    horizons = _parquet(
        revision_horizons_path,
        name="revision horizon result",
        columns=("split", "chunk_words", "revision_horizon_words"),
    )
    destination = Path(output_dir)
    figures = (
        _figure1(pareto, status, primary_chunk),
        _figure2(horizons, status, primary_chunk),
        _figure3(status),
        _figure4(Path(streaming_parity_path), Path(piimb_parity_path), status),
    )
    return tuple(path for figure in figures for path in _save_figure(figure, destination))


def _format(value: object, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicationError("table metric must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationError("table metric must be finite")
    return f"{result:.{digits}f}"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_table(
    destination: Path,
    stem: str,
    title: str,
    rows: Sequence[Mapping[str, str]],
    status: PublicationStatus,
) -> tuple[Path, Path]:
    if not rows:
        raise PublicationError(f"{title} has no rows")
    columns = tuple(rows[0])
    if any(tuple(row) != columns for row in rows):
        raise PublicationError(f"{title} rows have inconsistent columns")
    csv_rows = tuple({"Status": status.table_value, **row} for row in rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=tuple(csv_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_path = destination / f"{stem}.csv"
    _atomic_text(csv_path, buffer.getvalue())

    lines = [f"# {title}", ""]
    if status.pilot:
        lines.extend((f"> **{PILOT_LABEL}**", ""))
    lines.extend(
        (
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        )
    )
    lines.extend("| " + " | ".join(row[column] for column in columns) + " |" for row in rows)
    markdown_path = destination / f"{stem}.md"
    _atomic_text(markdown_path, "\n".join(lines) + "\n")
    return markdown_path, csv_path


def _unique_by_family(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    counts = frame.groupby("policy_family", dropna=False).size()
    ambiguous = sorted(str(family) for family, count in counts.items() if count != 1)
    if ambiguous:
        raise PublicationError(f"{name} has ambiguous policy rows: {ambiguous}")
    return frame.sort_values("policy_family")


def _table1(frame: pd.DataFrame, status: PublicationStatus) -> tuple[Mapping[str, str], ...]:
    frame = _unique_by_family(frame, name="main result")
    return tuple(
        {
            "Policy": _policy_name(row.policy_family),
            "Strict P": _format(row.strict_precision),
            "Strict R": _format(row.strict_recall),
            "Strict F1": _format(row.strict_f1),
            "Masking F2": _format(row.masking_f2),
            "Mean delay words": _format(row.mean_commit_context_words, 2),
            "Median delay words": _format(row.median_commit_context_words, 2),
            "Gold premature rate": _format(row.gold_premature_rate),
            "Wrong commit rate": _format(row.wrong_commitment_rate),
            "Blocked revisions": str(int(row.blocked_revision_count)),
        }
        for row in frame.itertuples(index=False)
    )


def _table2(frame: pd.DataFrame) -> tuple[Mapping[str, str], ...]:
    families = tuple(sorted(set(frame["policy_family"])))
    if set(frame["chunk_words"]) != {1, 2, 4, 8}:
        raise PublicationError("generalization result must contain chunk sizes 1,2,4,8")
    result: list[Mapping[str, str]] = []
    for chunk in (1, 2, 4, 8):
        chunk_rows = _unique_by_family(frame[frame["chunk_words"] == chunk], name=f"chunk {chunk}")
        if tuple(chunk_rows["policy_family"]) != families:
            raise PublicationError("generalization policy coverage differs across chunk sizes")
        row: dict[str, str] = {"Chunk words": str(chunk)}
        for item in chunk_rows.itertuples(index=False):
            label = _policy_name(item.policy_family)
            row[f"{label} F1"] = _format(item.strict_f1)
            row[f"{label} delay"] = _format(item.mean_commit_context_words, 2)
        result.append(row)
    return tuple(result)


def _table3(frame: pd.DataFrame) -> tuple[Mapping[str, str], ...]:
    required = ("full", "minus_instability", "minus_label_margin", "minus_extension")
    if set(frame["policy_variant"]) != set(required):
        raise PublicationError("StabilityGate ablation result must contain all required variants")
    labels = {
        "full": "Full StabilityGate",
        "minus_instability": "Minus instability",
        "minus_label_margin": "Minus label margin",
        "minus_extension": "Minus extension",
    }
    result: list[Mapping[str, str]] = []
    for variant in required:
        rows = frame[frame["policy_variant"] == variant]
        if len(rows) != 1:
            raise PublicationError(f"StabilityGate ablation {variant} is ambiguous")
        row = next(rows.itertuples(index=False))
        result.append(
            {
                "Variant": labels[variant],
                "Strict F1": _format(row.strict_f1),
                "Mean delay words": _format(row.mean_commit_context_words, 2),
                "Gold premature rate": _format(row.gold_premature_rate),
                "Wrong commit rate": _format(row.wrong_commitment_rate),
                "Blocked revisions / 100": _format(row.blocked_revisions_per_100_commitments, 2),
            }
        )
    return tuple(result)


def generate_tables(
    *,
    benchmark_path: str | Path,
    benchmark_manifest_path: str | Path,
    output_dir: str | Path,
    selection_mode: str = "matched_quality",
    primary_chunk: int = 1,
) -> tuple[Path, ...]:
    status = load_publication_status(benchmark_manifest_path)
    columns = (
        "policy_family",
        "policy_variant",
        "analysis_only",
        "split",
        "chunk_words",
        "aggregation",
        "selection_mode",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "masking_f2",
        "mean_commit_context_words",
        "median_commit_context_words",
        "gold_premature_rate",
        "wrong_commitment_rate",
        "blocked_revision_count",
        "blocked_revisions_per_100_commitments",
    )
    benchmark = _parquet(benchmark_path, name="held-out benchmark", columns=columns)
    selected = _rows(benchmark, split="test", selection_mode=selection_mode)
    if selected.empty:
        raise PublicationError(f"held-out benchmark has no {selection_mode!r} rows")
    primary = selected[selected["chunk_words"] == primary_chunk]
    main_primary = _main_policy_rows(primary)
    main_all = _main_policy_rows(selected)
    _require_policy_coverage(main_primary, name="main held-out result")
    _finite(
        selected,
        (
            "strict_precision",
            "strict_recall",
            "strict_f1",
            "masking_f2",
            "mean_commit_context_words",
            "median_commit_context_words",
            "gold_premature_rate",
            "wrong_commitment_rate",
            "blocked_revision_count",
            "blocked_revisions_per_100_commitments",
        ),
        name="held-out benchmark",
    )
    ablations = primary[primary["policy_family"] == "stability-gate"]
    destination = Path(output_dir)
    outputs = (
        _write_table(
            destination,
            "table1_main_results",
            "Table 1 — Main held-out result",
            _table1(main_primary, status),
            status,
        ),
        _write_table(
            destination,
            "table2_chunk_generalization",
            "Table 2 — Generalization across chunk sizes",
            _table2(main_all),
            status,
        ),
        _write_table(
            destination,
            "table3_stabilitygate_ablation",
            "Table 3 — StabilityGate ablation",
            _table3(ablations),
            status,
        ),
    )
    return tuple(path for pair in outputs for path in pair)


__all__ = [
    "PILOT_LABEL",
    "PublicationError",
    "PublicationStatus",
    "SYNTHETIC_TRAJECTORY_PREFIXES",
    "generate_figures",
    "generate_tables",
    "load_publication_status",
]
