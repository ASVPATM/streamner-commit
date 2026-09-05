"""Compute offline revision-premise metrics from raw trace JSON files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from streamner_commit.metrics.stability import (
    aggregate_premise_reports,
    analyze_premise_trace,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        dest="trace_paths",
        action="append",
        required=True,
        type=Path,
        help="Raw trace JSON path; repeat for multiple files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit replayable per-event arrays from the written report.",
    )
    return parser.parse_args(argv)


def read_trace_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def trace_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    traces = payload.get("traces")
    if traces is None and "span_updates" in payload:
        return (payload,)
    if isinstance(traces, str | bytes) or not isinstance(traces, Sequence):
        raise TypeError("trace payload must contain a traces sequence or one raw trace")
    rows = tuple(traces)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("traces must contain only JSON objects")
    return rows


def build_report(paths: Sequence[Path]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        payload = read_trace_payload(path)
        source = {
            "file": path.name,
            "run_id": payload.get("run_id"),
            # This is the benchmark chunk condition, not the observer's model
            # word-coordinate convention.
            "words_per_chunk": payload.get("words_per_chunk"),
        }
        sources.append(source)
        for trace in trace_rows(payload):
            report = analyze_premise_trace(trace)
            report["source_file"] = path.name
            report["run_id"] = payload.get("run_id")
            report["words_per_chunk"] = payload.get("words_per_chunk")
            reports.append(report)

    return {
        "schema_version": 1,
        "sources": sources,
        "summary": aggregate_premise_reports(reports),
        "traces": reports,
    }


def compact_report(value: Any) -> Any:
    """Remove bulky values that are deterministically recoverable from raw traces."""
    if isinstance(value, Mapping):
        omitted = {"absolute_values", "transitions", "per_boundary", "values_words"}
        return {key: compact_report(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [compact_report(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args.trace_paths)
    output_report = compact_report(report) if args.compact else report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote premise metrics for {report['summary']['trace_count']} traces to {args.output}")


if __name__ == "__main__":
    main()
