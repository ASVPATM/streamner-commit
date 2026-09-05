from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import streamner_commit.streaming.trace_pipeline as pipeline
from streamner_commit.datasets.piimb import PRIMARY_TASKS, resolve_preset
from streamner_commit.datasets.piimb_trace import (
    PIIMBTraceExample,
    PIIMBTraceManifest,
    PIIMBTraceManifestRecord,
    ReconstructedPIIMBTrace,
)
from streamner_commit.mlx.precision import MLXPrecisionError
from streamner_commit.streaming.trace_generation import (
    GeneratedExampleTrace,
    TraceInputExample,
    span_state_sha256,
)
from streamner_commit.streaming.trace_pipeline import (
    TraceModelProvenance,
    TracePipelineError,
    run_piimb_trace_pipeline,
)
from streamner_commit.types import (
    ColdFullResult,
    GoldEntity,
    SnapshotStep,
    SpanBoundary,
    SpanScoreUpdate,
)


class FakeBackend:
    context_limit = 512
    right_context_width = 12


def _source() -> ReconstructedPIIMBTrace:
    record = PIIMBTraceManifestRecord(
        split="dev",
        uid="fixture-uid",
        source_row_index=7,
        task_name=PRIMARY_TASKS[0],
        source_dataset="fixture/source",
        source_uid="source-7",
        parent_id="fixture:parent-7",
        sentence_index=0,
        language="en",
        metadata_sha256="c" * 64,
    )
    manifest = PIIMBTraceManifest(
        preset=resolve_preset("smoke"),
        task_labels={task: ("email", "person") for task in PRIMARY_TASKS},
        records=(record,),
        manifest_sha256="a" * 64,
        task_labels_sha256="b" * 64,
    )
    example_id = f"piimb:dev:{record.task_name}:7:fixture-uid"
    metadata = {
        "selection_index": 0,
        "benchmark_split": "dev",
        **record.source_metadata(),
        "metadata_sha256": record.metadata_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "preset": "smoke",
    }
    return ReconstructedPIIMBTrace(
        manifest=manifest,
        examples=(
            PIIMBTraceExample(
                split="dev",
                example_id=example_id,
                text="Ada",
                labels=("email", "person"),
                gold_entities=(GoldEntity(example_id, 0, 3, "person", "Ada"),),
                metadata=metadata,
            ),
        ),
    )


def _provenance() -> TraceModelProvenance:
    return TraceModelProvenance(
        model_id="fixture/model",
        model_revision_sha="revision-1",
        weights_sha256="1" * 64,
        checkpoint_config_sha256="2" * 64,
        export_manifest_sha256="3" * 64,
        tensor_manifest_sha256="4" * 64,
        tensor_count=1,
        parameter_count=1,
        model_config={
            "checkpoint": {"max_width": 12, "right_context_width": 12},
            "runtime": {"context_limit": 512, "right_context_width": 12},
        },
        context_limit=512,
        right_context_width=12,
        runtime_versions={
            "mlx_version": "0.32.2",
            "mlx_lm_version": "0.31.3",
            "transformers_version": "5.12.1",
            "torch_version": None,
            "gliner_version": None,
        },
    )


def _generated(example: TraceInputExample, run_id: str, chunk_words: int) -> GeneratedExampleTrace:
    probability = 1.0 / (1.0 + math.exp(-1.0))
    update = SpanScoreUpdate(
        run_id=run_id,
        example_id=example.example_id,
        step=1,
        chunk=example.text,
        visible_char_count=len(example.text),
        visible_word_count=1,
        start_word=0,
        end_word=0,
        start_char=0,
        end_char=len(example.text),
        span_text=example.text,
        logits=(1.0, -1.0),
        probs=(probability, 1.0 - probability),
        top_label_index=0,
        top_label=example.labels[0],
        top_probability=probability,
        second_probability=1.0 - probability,
        label_margin=(2.0 * probability) - 1.0,
        previous_top_probability=None,
        top_probability_delta=None,
        update_kind="new",
        tail_distance_words=0,
    )
    return GeneratedExampleTrace(
        run_id=run_id,
        chunk_words=chunk_words,
        example=example,
        snapshots=(
            SnapshotStep(
                run_id=run_id,
                example_id=example.example_id,
                step=1,
                chunk=example.text,
                accumulated_text=example.text,
                visible_char_count=len(example.text),
                visible_word_count=1,
                elapsed_ms=1.0,
                public_entities=(),
            ),
        ),
        span_updates=(update,),
        cold_full=ColdFullResult(
            example_id=example.example_id,
            full_text=example.text,
            public_entities=(),
            raw_final_span_state={SpanBoundary(0, 0): (1.0, -1.0)},
        ),
        final_state_sha256=span_state_sha256({SpanBoundary(0, 0): (1.0, -1.0)}),
    )


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "source.txt").write_text("fixed source\n", encoding="utf-8")
    return root


def test_conditions_persist_once_then_reuse_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def generate(
        _backend: object,
        examples: tuple[TraceInputExample, ...],
        *,
        run_id: str,
        chunk_words: int,
        **_kwargs: Any,
    ) -> tuple[GeneratedExampleTrace, ...]:
        calls.append((run_id, chunk_words))
        return tuple(_generated(example, run_id, chunk_words) for example in examples)

    monkeypatch.setattr(pipeline, "generate_condition_traces", generate)
    project = _project(tmp_path)
    output = tmp_path / "traces"
    first = run_piimb_trace_pipeline(
        FakeBackend(),
        _source(),
        _provenance(),
        project_root=project,
        preset="smoke",
        split="dev",
        chunk_words=(1, 2),
        test_output_root=output,
    )

    assert [condition.chunk_words for condition in first.conditions] == [1, 2]
    assert len({condition.run.fingerprint.run_id for condition in first.conditions}) == 2
    assert calls == [
        (first.conditions[0].run.fingerprint.run_id, 1),
        (first.conditions[1].run.fingerprint.run_id, 2),
    ]
    assert all(not condition.reused for condition in first.conditions)
    assert first.conditions[0].run.data.gold_entities == _source().examples[0].gold_entities
    assert "streamner_commit.policies" not in Path(pipeline.__file__).read_text(encoding="utf-8")

    calls.clear()
    second = run_piimb_trace_pipeline(
        FakeBackend(),
        _source(),
        _provenance(),
        project_root=project,
        preset="smoke",
        split="dev",
        chunk_words=(1, 2),
        test_output_root=output,
    )
    assert calls == []
    assert all(condition.reused for condition in second.conditions)


def test_refuses_mismatched_inputs_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def unexpected(*_args: object, **_kwargs: object) -> tuple[()]:
        nonlocal calls
        calls += 1
        return ()

    monkeypatch.setattr(pipeline, "generate_condition_traces", unexpected)
    source = _source()
    bad_metadata = dict(source.examples[0].metadata)
    bad_metadata["preset"] = "research-small"
    mixed = ReconstructedPIIMBTrace(
        manifest=source.manifest,
        examples=(replace(source.examples[0], metadata=bad_metadata),),
    )
    project = _project(tmp_path)

    with pytest.raises(TracePipelineError, match="metadata differs"):
        run_piimb_trace_pipeline(
            FakeBackend(),
            mixed,
            _provenance(),
            project_root=project,
            preset="smoke",
            split="dev",
            test_output_root=tmp_path / "traces",
        )
    with pytest.raises(TracePipelineError, match="requested preset"):
        run_piimb_trace_pipeline(
            FakeBackend(),
            source,
            _provenance(),
            project_root=project,
            preset="research-small",
            split="dev",
            test_output_root=tmp_path / "traces",
        )
    with pytest.raises(TracePipelineError, match="duplicate chunk"):
        run_piimb_trace_pipeline(
            FakeBackend(),
            source,
            _provenance(),
            project_root=project,
            preset="smoke",
            split="dev",
            chunk_words=(1, 1),
            test_output_root=tmp_path / "traces",
        )
    assert calls == 0


def test_generation_failure_leaves_no_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> tuple[()]:
        raise RuntimeError("injected model failure")

    monkeypatch.setattr(pipeline, "generate_condition_traces", fail)
    output = tmp_path / "traces"
    with pytest.raises(RuntimeError, match="injected model failure"):
        run_piimb_trace_pipeline(
            FakeBackend(),
            _source(),
            _provenance(),
            project_root=_project(tmp_path),
            preset="smoke",
            split="dev",
            test_output_root=output,
        )
    assert not output.exists() or not list(output.glob("*/manifest.json"))


def test_refuses_incompatible_tf32_and_external_raw_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(MLXPrecisionError):
        run_piimb_trace_pipeline(
            FakeBackend(),
            _source(),
            _provenance(),
            project_root=project,
            preset="smoke",
            split="dev",
            test_output_root=tmp_path / "traces",
        )
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    with pytest.raises(TracePipelineError, match="results/traces"):
        run_piimb_trace_pipeline(
            FakeBackend(),
            _source(),
            _provenance(),
            project_root=project,
            preset="smoke",
            split="dev",
            output_root=tmp_path / "unsafe",
        )
