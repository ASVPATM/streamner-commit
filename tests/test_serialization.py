from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

import streamner_commit.serialization as serialization
from streamner_commit.serialization import (
    COLD_FULL_SCHEMA,
    EXAMPLES_SCHEMA,
    GOLD_ENTITIES_SCHEMA,
    SNAPSHOTS_SCHEMA,
    SPAN_UPDATES_SCHEMA,
    STEPS_SCHEMA,
    IncompleteTraceRunError,
    TraceExample,
    TraceIntegrityError,
    TraceRunData,
    TraceSerializationError,
    build_trace_fingerprint,
    canonical_json_bytes,
    read_trace_run,
    source_tree_sha256,
    warm_span_state_sha256,
    write_trace_run,
)
from streamner_commit.streaming.replay import replay_span_updates
from streamner_commit.types import (
    ColdFullResult,
    GoldEntity,
    PublicEntity,
    SnapshotStep,
    SpanBoundary,
    SpanScoreUpdate,
)


def example() -> TraceExample:
    return TraceExample(
        example_id="example-1",
        text="Ada works.",
        labels=("person", "email"),
        task_name="fixture-task",
        uid="uid-1",
        source_row_index=7,
        parent_id="parent-1",
        split="dev",
        source_dataset="fixture",
        source_uid="source-1",
        sentence_index=2,
        language="en",
    )


def fingerprint(project_root: Path, trace_example: TraceExample | None = None):
    project_root.mkdir(parents=True, exist_ok=True)
    source = project_root / "source.txt"
    if not source.exists():
        source.write_text("source-v1\n", encoding="utf-8")
    return build_trace_fingerprint(
        examples=[trace_example or example()],
        project_root=project_root,
        model_sha="a" * 40,
        backend="mlx",
        dtype="float32",
        chunk_strategy="whitespace-units",
        chunk_words=1,
        model_config={
            "flat_ner": True,
            "multi_label": False,
            "public_threshold": 0.5,
            "max_width": 12,
            "right_context_width": 8,
        },
        device="gpu",
        runtime_versions={
            "mlx_version": "0.32.2",
            "mlx_lm_version": "0.31.3",
            "transformers_version": "5.12.1",
            "torch_version": None,
            "gliner_version": None,
        },
        model_id="fixture/model",
        checkpoint_config_sha256="c" * 64,
        public_threshold=0.5,
        flat_ner=True,
        multi_label=False,
        max_width=12,
        right_context_width=8,
        dataset_id="fixture/data",
        dataset_revision="dataset-sha-1",
        dataset_subset="sentences",
        dataset_tasks=("fixture-task",),
        sample_manifest_sha256="d" * 64,
        random_seed=17,
        extra_inputs={"device": "gpu", "decoder_tie_break": "first"},
    )


def first_update(run_id: str) -> SpanScoreUpdate:
    return SpanScoreUpdate(
        run_id=run_id,
        example_id="example-1",
        step=1,
        chunk="Ada ",
        visible_char_count=4,
        visible_word_count=1,
        start_word=0,
        end_word=0,
        start_char=0,
        end_char=3,
        span_text="Ada",
        logits=(1.25, -1.25),
        probs=(0.8, 0.2),
        top_label_index=0,
        top_label="person",
        top_probability=0.8,
        second_probability=0.2,
        label_margin=0.6,
        previous_top_probability=None,
        top_probability_delta=None,
        update_kind="new",
        tail_distance_words=0,
    )


def second_update(run_id: str) -> SpanScoreUpdate:
    return SpanScoreUpdate(
        run_id=run_id,
        example_id="example-1",
        step=2,
        chunk="works.",
        visible_char_count=10,
        visible_word_count=3,
        start_word=0,
        end_word=0,
        start_char=0,
        end_char=3,
        span_text="Ada",
        logits=(0.75, -0.75),
        probs=(0.7, 0.3),
        top_label_index=0,
        top_label="person",
        top_probability=0.7,
        second_probability=0.3,
        label_margin=0.4,
        previous_top_probability=0.8,
        top_probability_delta=-0.1,
        update_kind="rescore",
        tail_distance_words=2,
    )


def run_data(run_id: str, *, include_gold: bool = True) -> TraceRunData:
    entity = PublicEntity(
        start_char=0,
        end_char=3,
        label="person",
        text="Ada",
        score=0.7,
    )
    steps = (
        SnapshotStep(
            run_id=run_id,
            example_id="example-1",
            step=1,
            chunk="Ada ",
            accumulated_text="Ada ",
            visible_char_count=4,
            visible_word_count=1,
            elapsed_ms=1.125,
            public_entities=(),
        ),
        SnapshotStep(
            run_id=run_id,
            example_id="example-1",
            step=2,
            chunk="works.",
            accumulated_text="Ada works.",
            visible_char_count=10,
            visible_word_count=3,
            elapsed_ms=2.25,
            public_entities=(entity,),
        ),
    )
    cold = ColdFullResult(
        example_id="example-1",
        full_text="Ada works.",
        public_entities=(replace(entity, score=0.72),),
        raw_final_span_state={
            SpanBoundary(0, 0): (0.875, -0.625),
            SpanBoundary(1, 2): (-1.0, 0.25),
        },
    )
    gold = (
        GoldEntity(
            example_id="example-1",
            start_char=0,
            end_char=3,
            label="person",
            text="Ada",
        ),
    )
    updates = (first_update(run_id), second_update(run_id))
    final_state_sha256 = warm_span_state_sha256(replay_span_updates(updates))
    return TraceRunData(
        examples=(replace(example(), final_state_sha256=final_state_sha256),),
        steps=steps,
        span_updates=updates,
        cold_full=(cold,),
        gold_entities=gold if include_gold else None,
    )


def test_round_trip_preserves_empty_snapshot_gold_and_cold_raw_state(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    completed = write_trace_run(
        project_root / "results" / "traces",
        run_fingerprint,
        run_data(run_fingerprint.run_id),
        created_utc="2026-09-01T12:00:00Z",
    )

    assert completed.path.name == run_fingerprint.run_id
    assert completed.manifest["complete"] is True
    assert completed.manifest["labels_by_example"] == {
        "example-1": ["person", "email"]
    }
    assert completed.manifest["task_labels"] == {
        "fixture-task": ["person", "email"]
    }
    assert completed.manifest["run_config"] == completed.fingerprint.payload["run_config"]
    run_config = completed.manifest["run_config"]
    assert isinstance(run_config, Mapping)
    assert run_config["backend"] == "mlx"
    assert run_config["device"] == "gpu"
    assert run_config["mlx_version"] == "0.32.2"
    assert run_config["torch_version"] is None
    assert run_config["model_id"] == "fixture/model"
    assert run_config["model_revision_sha"] == "a" * 40
    assert run_config["public_threshold"] == 0.5
    assert run_config["chunk_words"] == 1
    assert run_config["dataset_revision"] == "dataset-sha-1"
    persisted_example = completed.data.examples[0]
    assert isinstance(persisted_example, TraceExample)
    assert replace(persisted_example, final_state_sha256=None) == example()
    assert len(completed.data.steps) == 2
    first_step = completed.data.steps[0]
    assert isinstance(first_step, SnapshotStep)
    assert first_step.public_entities == ()
    second_step = completed.data.steps[1]
    assert isinstance(second_step, SnapshotStep)
    assert second_step.public_entities[0].text == "Ada"
    assert completed.data.span_updates == (
        first_update(run_fingerprint.run_id),
        second_update(run_fingerprint.run_id),
    )
    replayed = replay_span_updates(completed.data.span_updates)  # type: ignore[arg-type]
    assert warm_span_state_sha256(replayed) == persisted_example.final_state_sha256
    cold = completed.data.cold_full[0]
    assert isinstance(cold, ColdFullResult)
    assert dict(cold.raw_final_span_state) == {
        SpanBoundary(0, 0): (0.875, -0.625),
        SpanBoundary(1, 2): (-1.0, 0.25),
    }
    assert cold.public_entities[0].start_char == 0
    assert cold.public_entities[0].end_char == 3
    assert completed.data.gold_entities == (
        GoldEntity("example-1", 0, 3, "person", "Ada"),
    )

    manifest = json.loads((completed.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["fingerprint"]["project_git_commit"] is None
    assert len(manifest["fingerprint"]["source_tree_sha256"]) == 64
    assert set(manifest["files"]) == {
        "examples.parquet",
        "steps.parquet",
        "span_updates.parquet",
        "snapshots.parquet",
        "cold_full.parquet",
        "gold_entities.parquet",
    }
    assert manifest["files"]["steps.parquet"]["row_count"] == 2
    assert manifest["files"]["snapshots.parquet"]["row_count"] == 1
    assert read_trace_run(completed.path).data == completed.data


def test_optional_gold_file_is_absent_for_nonbenchmark_run(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    completed = write_trace_run(
        project_root / "results" / "traces",
        run_fingerprint,
        run_data(run_fingerprint.run_id, include_gold=False),
    )
    assert not (completed.path / "gold_entities.parquet").exists()
    assert completed.data.gold_entities is None


def test_identical_verified_run_is_reused_without_any_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    output_root = project_root / "results" / "traces"
    original = write_trace_run(
        output_root,
        run_fingerprint,
        run_data(run_fingerprint.run_id),
        created_utc="2026-09-01T12:00:00Z",
    )
    before = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in original.path.iterdir()
    }

    def unexpected_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact reuse must not invoke the Parquet writer")

    monkeypatch.setattr(serialization, "_write_parquet", unexpected_write)
    reused = write_trace_run(
        output_root,
        run_fingerprint,
        run_data(run_fingerprint.run_id),
        created_utc="2099-01-01T00:00:00Z",
    )
    after = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in reused.path.iterdir()
    }
    assert reused.path == original.path
    assert after == before


def test_failure_cleans_staging_and_never_exposes_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    output_root = project_root / "results" / "traces"
    output_root.mkdir(parents=True)
    stale = output_root / f".{run_fingerprint.run_id}.staging-stale"
    stale.mkdir()
    (stale / "partial").write_text("partial", encoding="utf-8")
    calls = 0
    real_write = serialization._write_parquet

    def fail_after_one(table: pa.Table, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected write failure")
        real_write(table, path)

    monkeypatch.setattr(serialization, "_write_parquet", fail_after_one)
    with pytest.raises(RuntimeError, match="injected write failure"):
        write_trace_run(output_root, run_fingerprint, run_data(run_fingerprint.run_id))

    assert not (output_root / run_fingerprint.run_id).exists()
    assert not list(output_root.glob(f".{run_fingerprint.run_id}.staging-*"))
    assert not list(output_root.rglob("manifest.json"))


def test_from_generated_flattens_typed_core_contract_and_copies_final_hash() -> None:
    run_id = "trace-generated"
    data = run_data(run_id)
    persisted_example = data.examples[0]
    assert isinstance(persisted_example, TraceExample)
    generated_gold = cast(tuple[GoldEntity, ...], data.gold_entities)
    source = SimpleNamespace(
        gold_entities=data.gold_entities,
        to_dict=lambda: {
            "example_id": persisted_example.example_id,
            "text": persisted_example.text,
            "labels": list(persisted_example.labels),
            "gold_entities": [entity.to_dict() for entity in generated_gold],
            "metadata": {"task_name": "fixture-task", "uid": "uid-1"},
        },
    )
    generated = SimpleNamespace(
        example=source,
        final_state_sha256=persisted_example.final_state_sha256,
        snapshots=data.steps,
        span_updates=data.span_updates,
        cold_full=data.cold_full[0],
    )
    flattened = TraceRunData.from_generated((generated,))
    flattened_example = flattened.examples[0]
    assert isinstance(flattened_example, dict)
    assert flattened_example["final_state_sha256"] == persisted_example.final_state_sha256
    assert flattened.steps == data.steps
    assert flattened.span_updates == data.span_updates
    assert flattened.cold_full == data.cold_full
    assert flattened.gold_entities == data.gold_entities


def test_writer_rejects_a_generated_final_state_hash_that_replay_cannot_reproduce(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    data = run_data(run_fingerprint.run_id)
    persisted_example = data.examples[0]
    assert isinstance(persisted_example, TraceExample)
    bad_data = replace(
        data,
        examples=(replace(persisted_example, final_state_sha256="f" * 64),),
    )
    with pytest.raises(TraceSerializationError, match="replayed warm state"):
        write_trace_run(project_root / "results" / "traces", run_fingerprint, bad_data)


def test_every_fingerprint_input_component_changes_the_run_id(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    baseline = fingerprint(project_root)
    baseline_kwargs: dict[str, Any] = {
        "examples": [example()],
        "project_root": project_root,
        "model_sha": "a" * 40,
        "backend": "mlx",
        "dtype": "float32",
        "chunk_strategy": "whitespace-units",
        "chunk_words": 1,
        "model_config": {
            "flat_ner": True,
            "multi_label": False,
            "public_threshold": 0.5,
            "max_width": 12,
            "right_context_width": 8,
        },
        "device": "gpu",
        "runtime_versions": {
            "mlx_version": "0.32.2",
            "mlx_lm_version": "0.31.3",
            "transformers_version": "5.12.1",
            "torch_version": None,
            "gliner_version": None,
        },
        "model_id": "fixture/model",
        "checkpoint_config_sha256": "c" * 64,
        "public_threshold": 0.5,
        "flat_ner": True,
        "multi_label": False,
        "max_width": 12,
        "right_context_width": 8,
        "gliner_reference_tag": None,
        "gliner_reference_commit": None,
        "platform_name": "Darwin",
        "machine_architecture": "arm64",
        "python_version": "3.12.11",
        "dataset_id": "fixture/data",
        "dataset_revision": "dataset-sha-1",
        "dataset_subset": "sentences",
        "dataset_tasks": ("fixture-task",),
        "sample_manifest_sha256": "d" * 64,
        "random_seed": 17,
        "extra_inputs": {"device": "gpu", "decoder_tie_break": "first"},
    }
    variants = (
        {"model_sha": "b" * 40},
        {"backend": "reference"},
        {"dtype": "float16"},
        {"chunk_strategy": "characters"},
        {"chunk_words": 2},
        {
            "model_config": {
                "flat_ner": True,
                "multi_label": False,
                "public_threshold": 0.5,
                "max_width": 13,
                "right_context_width": 8,
            }
        },
        {"device": "cpu"},
        {
            "runtime_versions": {
                "mlx_version": "0.32.3",
                "mlx_lm_version": "0.31.3",
                "transformers_version": "5.12.1",
                "torch_version": None,
                "gliner_version": None,
            }
        },
        {"model_id": "fixture/other-model"},
        {"checkpoint_config_sha256": "e" * 64},
        {"public_threshold": 0.6},
        {"flat_ner": False},
        {"multi_label": True},
        {"max_width": 13},
        {"right_context_width": 9},
        {"gliner_reference_tag": "v0.2.28"},
        {"gliner_reference_commit": "reference-commit"},
        {"platform_name": "Linux"},
        {"machine_architecture": "x86_64"},
        {"python_version": "3.12.12"},
        {"dataset_id": "other/data"},
        {"dataset_revision": "dataset-sha-2"},
        {"dataset_subset": "documents"},
        {"dataset_tasks": ("other-task",)},
        {"sample_manifest_sha256": "e" * 64},
        {"random_seed": 18},
        {"extra_inputs": {"device": "cpu", "decoder_tie_break": "first"}},
        {"examples": [replace(example(), uid="uid-2")]},
        {"examples": [replace(example(), labels=("email", "person"))]},
        {"examples": [replace(example(), text="Ada rests.")]},
        {"examples": [replace(example(), task_name="other-task")]},
    )
    changed_ids = {
        build_trace_fingerprint(**{**baseline_kwargs, **variant}).run_id
        for variant in variants
    }
    assert baseline.run_id not in changed_ids
    assert len(changed_ids) == len(variants)

    (project_root / "source.txt").write_text("source-v2\n", encoding="utf-8")
    source_changed = build_trace_fingerprint(**baseline_kwargs)
    assert source_changed.run_id != baseline.run_id


def test_source_tree_digest_ignores_trace_outputs_and_is_order_stable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src").mkdir(parents=True)
    (project_root / "src" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (project_root / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    first = source_tree_sha256(project_root)
    (project_root / "results" / "traces" / "ignored").mkdir(parents=True)
    (project_root / "results" / "traces" / "ignored" / "x").write_text(
        "generated", encoding="utf-8"
    )
    assert source_tree_sha256(project_root) == first
    (project_root / "src" / "a.py").write_text("a = 3\n", encoding="utf-8")
    assert source_tree_sha256(project_root) != first


def test_project_version_binds_dirty_committed_source_to_tree_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        (
            SimpleNamespace(stdout=f"{'a' * 40}\n"),
            SimpleNamespace(stdout=" M src/example.py\n"),
        )
    )
    monkeypatch.setattr(serialization.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(serialization, "source_tree_sha256", lambda root: "b" * 64)

    assert serialization.project_version(tmp_path) == ("a" * 40, "b" * 64)


def test_tampered_file_and_manifest_row_count_are_rejected(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    completed = write_trace_run(
        project_root / "results" / "traces",
        run_fingerprint,
        run_data(run_fingerprint.run_id),
    )
    examples_path = completed.path / "examples.parquet"
    examples_path.write_bytes(examples_path.read_bytes() + b"tamper")
    with pytest.raises(TraceIntegrityError, match="checksum"):
        read_trace_run(completed.path)

    other_root = tmp_path / "other-project"
    other_fingerprint = fingerprint(other_root)
    other = write_trace_run(
        other_root / "results" / "traces",
        other_fingerprint,
        run_data(other_fingerprint.run_id),
    )
    manifest_path = other.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["steps.parquet"]["row_count"] += 1
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(TraceIntegrityError, match="row count"):
        read_trace_run(other.path)


def test_incomplete_run_and_wrong_expected_fingerprint_are_rejected(tmp_path: Path) -> None:
    incomplete = tmp_path / "trace-incomplete"
    incomplete.mkdir()
    (incomplete / "steps.parquet").write_bytes(b"partial")
    with pytest.raises(IncompleteTraceRunError, match="no manifest"):
        read_trace_run(incomplete)

    project_root = tmp_path / "project"
    run_fingerprint = fingerprint(project_root)
    completed = write_trace_run(
        project_root / "results" / "traces",
        run_fingerprint,
        run_data(run_fingerprint.run_id),
    )
    different = fingerprint(
        project_root,
        trace_example=replace(example(), uid="uid-different"),
    )
    with pytest.raises(TraceIntegrityError, match="fingerprint differs"):
        read_trace_run(completed.path, expected_fingerprint=different)


def test_schemas_are_explicit_and_never_hide_nested_records_in_json() -> None:
    assert EXAMPLES_SCHEMA.field("labels").type == pa.list_(
        pa.field("element", pa.string(), nullable=False)
    )
    assert STEPS_SCHEMA.get_field_index("public_entities") == -1
    assert SPAN_UPDATES_SCHEMA.field("logits").type.value_type == pa.float64()
    assert SPAN_UPDATES_SCHEMA.field("probs").type.value_type == pa.float64()
    assert SNAPSHOTS_SCHEMA.field("entity_index").type == pa.int64()
    assert pa.types.is_list(COLD_FULL_SCHEMA.field("public_entities").type)
    assert pa.types.is_struct(COLD_FULL_SCHEMA.field("public_entities").type.value_type)
    assert pa.types.is_list(COLD_FULL_SCHEMA.field("raw_final_span_state").type)
    assert pa.types.is_struct(COLD_FULL_SCHEMA.field("raw_final_span_state").type.value_type)
    raw_struct = COLD_FULL_SCHEMA.field("raw_final_span_state").type.value_type
    assert pa.types.is_list(raw_struct.field("logits").type)
    assert GOLD_ENTITIES_SCHEMA.get_field_index("entity_index") >= 0
    for schema in (
        EXAMPLES_SCHEMA,
        STEPS_SCHEMA,
        SPAN_UPDATES_SCHEMA,
        SNAPSHOTS_SCHEMA,
        COLD_FULL_SCHEMA,
        GOLD_ENTITIES_SCHEMA,
    ):
        assert all("json" not in field.name.lower() for field in schema)
