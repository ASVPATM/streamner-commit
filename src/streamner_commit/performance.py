"""Small, backend-agnostic timing harness for Phase 19 characterization.

The harness deliberately does not import MLX.  Callers timing a lazy model must
provide an explicit synchronizer for cold results, while warm sessions provide
their already-synchronized per-append durations.  Policy replay is a separate
host-only callable, so the timing path has no backend or model reference.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from streamner_commit.chunking import chunk_text_by_words, count_words

TimingOperation = Literal["cold_full", "warm_append", "policy_replay"]


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _nonblank(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _duration(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class PerformanceConfiguration:
    """Stable input dimensions shared by all three timing categories."""

    warmup_count: int
    repetition_count: int
    chunk_words: int
    text_length_chars: int
    text_update_units: int
    chunk_count: int
    labels: tuple[str, ...]
    text_sha256: str

    def __post_init__(self) -> None:
        _integer(self.warmup_count, name="warmup_count", minimum=0)
        _integer(self.repetition_count, name="repetition_count", minimum=1)
        _integer(self.chunk_words, name="chunk_words", minimum=1)
        _integer(self.text_length_chars, name="text_length_chars", minimum=1)
        _integer(self.text_update_units, name="text_update_units", minimum=1)
        _integer(self.chunk_count, name="chunk_count", minimum=2)
        labels = tuple(self.labels)
        if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("labels must contain nonblank strings")
        if len(labels) != len(set(labels)):
            raise ValueError("labels must be unique")
        if (
            not isinstance(self.text_sha256, str)
            or len(self.text_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.text_sha256)
        ):
            raise ValueError("text_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "labels", labels)

    @classmethod
    def from_text(
        cls,
        text: str,
        labels: Sequence[str],
        *,
        warmup_count: int,
        repetition_count: int,
        chunk_words: int,
    ) -> PerformanceConfiguration:
        """Construct dimensions from exact text without retaining its content."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("benchmark text must contain a non-whitespace character")
        chunks = tuple(chunk_text_by_words(text, chunk_words))
        if len(chunks) < 2:
            raise ValueError("benchmark text must produce at least one warm append")
        return cls(
            warmup_count=warmup_count,
            repetition_count=repetition_count,
            chunk_words=chunk_words,
            text_length_chars=len(text),
            text_update_units=count_words(text),
            chunk_count=len(chunks),
            labels=tuple(labels),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @property
    def warm_appends_per_run(self) -> int:
        """Number of timed appends after the untimed first-session append."""

        return self.chunk_count - 1

    def to_dict(self) -> dict[str, object]:
        return {
            "warmup_count": self.warmup_count,
            "repetition_count": self.repetition_count,
            "chunk_words": self.chunk_words,
            "text_length_chars": self.text_length_chars,
            "text_update_units": self.text_update_units,
            "chunk_count": self.chunk_count,
            "warm_appends_per_run": self.warm_appends_per_run,
            "labels": list(self.labels),
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Device, precision, platform, and package identity for one local run."""

    model_revision: str
    dtype: str
    model_device: str
    replay_device: str
    precision_mode: str
    machine: str
    platform: str
    python_version: str
    runtime_versions: Mapping[str, str] = field(hash=False)

    def __post_init__(self) -> None:
        for name in (
            "model_revision",
            "dtype",
            "model_device",
            "replay_device",
            "precision_mode",
            "machine",
            "platform",
            "python_version",
        ):
            _nonblank(getattr(self, name), name=name)
        if not isinstance(self.runtime_versions, Mapping):
            raise TypeError("runtime_versions must be a mapping")
        versions: dict[str, str] = {}
        for raw_name, raw_version in self.runtime_versions.items():
            name = _nonblank(raw_name, name="runtime name")
            version = _nonblank(raw_version, name=f"runtime version for {name}")
            versions[name] = version
        object.__setattr__(
            self,
            "runtime_versions",
            MappingProxyType(dict(sorted(versions.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_revision": self.model_revision,
            "dtype": self.dtype,
            "model_device": self.model_device,
            "replay_device": self.replay_device,
            "precision_mode": self.precision_mode,
            "machine": self.machine,
            "platform": self.platform,
            "python_version": self.python_version,
            "runtime_versions": dict(self.runtime_versions),
        }


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Raw synchronized samples plus transparent operation counts."""

    operation: TimingOperation
    device: str
    samples_ms: tuple[float, ...]
    warmup_operations: int
    timed_operations: int

    def __post_init__(self) -> None:
        if self.operation not in {"cold_full", "warm_append", "policy_replay"}:
            raise ValueError("unknown timing operation")
        _nonblank(self.device, name="device")
        samples = tuple(
            _duration(value, name=f"{self.operation} sample") for value in self.samples_ms
        )
        if not samples:
            raise ValueError("timing summary must contain a sample")
        _integer(self.warmup_operations, name="warmup_operations", minimum=0)
        timed = _integer(self.timed_operations, name="timed_operations", minimum=1)
        if timed != len(samples):
            raise ValueError("timed_operations must equal the number of samples")
        object.__setattr__(self, "samples_ms", samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "device": self.device,
            "warmup_operations": self.warmup_operations,
            "timed_operations": self.timed_operations,
            "samples_ms": list(self.samples_ms),
            "mean_ms": math.fsum(self.samples_ms) / len(self.samples_ms),
            "median_ms": statistics.median(self.samples_ms),
            "minimum_ms": min(self.samples_ms),
            "maximum_ms": max(self.samples_ms),
        }


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """One comparable cold, warm, and host-only policy timing report."""

    configuration: PerformanceConfiguration
    runtime: RuntimeMetadata
    cold_full: TimingSummary
    warm_append: TimingSummary
    policy_replay: TimingSummary

    def __post_init__(self) -> None:
        expected = (
            (self.cold_full, "cold_full"),
            (self.warm_append, "warm_append"),
            (self.policy_replay, "policy_replay"),
        )
        if any(summary.operation != name for summary, name in expected):
            raise ValueError("performance report timing categories are misassigned")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "configuration": self.configuration.to_dict(),
            "runtime": self.runtime.to_dict(),
            "timings": {
                "cold_full": self.cold_full.to_dict(),
                "warm_append": self.warm_append.to_dict(),
                "policy_replay": self.policy_replay.to_dict(),
            },
        }


def _elapsed_ms[T](clock: Callable[[], float], operation: Callable[[], T]) -> tuple[float, T]:
    started = clock()
    result = operation()
    elapsed = (clock() - started) * 1000.0
    return _duration(elapsed, name="elapsed milliseconds"), result


def benchmark_cold_full[T](
    operation: Callable[[], T],
    synchronize: Callable[[T], None],
    *,
    warmup_count: int,
    repetition_count: int,
    device: str,
    clock: Callable[[], float] = time.perf_counter,
) -> TimingSummary:
    """Time stateless cold calls, synchronizing each lazy result before stop."""

    warmups = _integer(warmup_count, name="warmup_count", minimum=0)
    repetitions = _integer(repetition_count, name="repetition_count", minimum=1)
    if not callable(operation) or not callable(synchronize) or not callable(clock):
        raise TypeError("cold operation, synchronizer, and clock must be callable")
    for _ in range(warmups):
        synchronize(operation())
    samples: list[float] = []
    for _ in range(repetitions):
        started = clock()
        result = operation()
        synchronize(result)
        samples.append(_duration((clock() - started) * 1000.0, name="cold duration"))
    return TimingSummary("cold_full", device, tuple(samples), warmups, repetitions)


def benchmark_warm_appends(
    run: Callable[[], Sequence[float]],
    *,
    warmup_count: int,
    repetition_count: int,
    appends_per_run: int,
    device: str,
) -> TimingSummary:
    """Collect backend-provided synchronized elapsed times for warm appends."""

    warmups = _integer(warmup_count, name="warmup_count", minimum=0)
    repetitions = _integer(repetition_count, name="repetition_count", minimum=1)
    expected = _integer(appends_per_run, name="appends_per_run", minimum=1)
    if not callable(run):
        raise TypeError("warm append run must be callable")

    def one_run() -> tuple[float, ...]:
        values = run()
        if isinstance(values, str | bytes) or not isinstance(values, Sequence):
            raise TypeError("warm append run must return a duration sequence")
        samples = tuple(_duration(value, name="warm append duration") for value in values)
        if len(samples) != expected:
            raise ValueError(
                f"warm append run returned {len(samples)} samples; expected {expected}"
            )
        return samples

    for _ in range(warmups):
        one_run()
    timed = tuple(sample for _ in range(repetitions) for sample in one_run())
    return TimingSummary(
        "warm_append",
        device,
        timed,
        warmups * expected,
        repetitions * expected,
    )


def benchmark_policy_replay(
    operation: Callable[[], object],
    *,
    warmup_count: int,
    repetition_count: int,
    device: str,
    clock: Callable[[], float] = time.perf_counter,
) -> TimingSummary:
    """Time a pure host-only replay callable with no backend/model parameter."""

    warmups = _integer(warmup_count, name="warmup_count", minimum=0)
    repetitions = _integer(repetition_count, name="repetition_count", minimum=1)
    if not callable(operation) or not callable(clock):
        raise TypeError("policy replay operation and clock must be callable")
    for _ in range(warmups):
        operation()
    samples = tuple(_elapsed_ms(clock, operation)[0] for _ in range(repetitions))
    return TimingSummary("policy_replay", device, samples, warmups, repetitions)


def benchmark_performance[T](
    configuration: PerformanceConfiguration,
    runtime: RuntimeMetadata,
    *,
    cold_full_operation: Callable[[], T],
    synchronize_cold: Callable[[T], None],
    warm_append_run: Callable[[], Sequence[float]],
    policy_replay_operation: Callable[[], object],
    clock: Callable[[], float] = time.perf_counter,
) -> PerformanceReport:
    """Run all categories after their own warmups and return a stable report."""

    if not isinstance(configuration, PerformanceConfiguration):
        raise TypeError("configuration must be PerformanceConfiguration")
    if not isinstance(runtime, RuntimeMetadata):
        raise TypeError("runtime must be RuntimeMetadata")
    cold = benchmark_cold_full(
        cold_full_operation,
        synchronize_cold,
        warmup_count=configuration.warmup_count,
        repetition_count=configuration.repetition_count,
        device=runtime.model_device,
        clock=clock,
    )
    warm = benchmark_warm_appends(
        warm_append_run,
        warmup_count=configuration.warmup_count,
        repetition_count=configuration.repetition_count,
        appends_per_run=configuration.warm_appends_per_run,
        device=runtime.model_device,
    )
    replay = benchmark_policy_replay(
        policy_replay_operation,
        warmup_count=configuration.warmup_count,
        repetition_count=configuration.repetition_count,
        device=runtime.replay_device,
        clock=clock,
    )
    return PerformanceReport(configuration, runtime, cold, warm, replay)


__all__ = [
    "PerformanceConfiguration",
    "PerformanceReport",
    "RuntimeMetadata",
    "TimingSummary",
    "benchmark_cold_full",
    "benchmark_performance",
    "benchmark_policy_replay",
    "benchmark_warm_appends",
]
