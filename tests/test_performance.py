from __future__ import annotations

from dataclasses import dataclass

import pytest

from streamner_commit.performance import (
    PerformanceConfiguration,
    RuntimeMetadata,
    benchmark_performance,
)


@dataclass
class _FakeBackend:
    cold_calls: int = 0
    synchronized_results: int = 0
    warm_runs: int = 0

    def cold_full(self) -> object:
        self.cold_calls += 1
        return {"call": self.cold_calls}

    def synchronize(self, _result: object) -> None:
        self.synchronized_results += 1

    def warm_append_run(self) -> tuple[float, ...]:
        self.warm_runs += 1
        return {
            1: (90.0, 91.0),  # warmup: deliberately absent from samples
            2: (1.0, 2.0),
            3: (3.0, 4.0),
        }[self.warm_runs]


def test_fake_benchmark_excludes_warmups_and_keeps_replay_model_free() -> None:
    configuration = PerformanceConfiguration.from_text(
        "alpha beta gamma",
        ("person", "place"),
        warmup_count=1,
        repetition_count=2,
        chunk_words=1,
    )
    runtime = RuntimeMetadata(
        model_revision="revision-123",
        dtype="float32",
        model_device="fake accelerator",
        replay_device="fake CPU",
        precision_mode="fake full precision",
        machine="fake-arm64",
        platform="fakeOS 1",
        python_version="3.12.fake",
        runtime_versions={"fake-runtime": "1.0"},
    )
    backend = _FakeBackend()
    replay_calls = 0

    def policy_replay() -> object:
        nonlocal replay_calls
        # The replay timer receives no backend/model object and must not alter either
        # count after the cold/warm categories have completed.
        assert (backend.cold_calls, backend.warm_runs) == (3, 3)
        replay_calls += 1
        return ("offline", replay_calls)

    clock_values = iter((0.0, 0.001, 1.0, 1.002, 2.0, 2.003, 3.0, 3.004))
    report = benchmark_performance(
        configuration,
        runtime,
        cold_full_operation=backend.cold_full,
        synchronize_cold=backend.synchronize,
        warm_append_run=backend.warm_append_run,
        policy_replay_operation=policy_replay,
        clock=lambda: next(clock_values),
    )

    assert backend.cold_calls == 3
    assert backend.synchronized_results == 3
    assert backend.warm_runs == 3
    assert replay_calls == 3

    assert report.cold_full.samples_ms == pytest.approx((1.0, 2.0))
    assert report.warm_append.samples_ms == (1.0, 2.0, 3.0, 4.0)
    assert report.policy_replay.samples_ms == pytest.approx((3.0, 4.0))
    assert report.cold_full.warmup_operations == 1
    assert report.warm_append.warmup_operations == 2
    assert report.policy_replay.warmup_operations == 1
    assert report.cold_full.timed_operations == 2
    assert report.warm_append.timed_operations == 4
    assert report.policy_replay.timed_operations == 2

    serialized = report.to_dict()
    assert serialized == report.to_dict()
    assert serialized["configuration"] == configuration.to_dict()
    assert serialized["runtime"] == runtime.to_dict()
    assert serialized["timings"]["cold_full"]["device"] == "fake accelerator"  # type: ignore[index]
    assert serialized["timings"]["policy_replay"]["device"] == "fake CPU"  # type: ignore[index]
