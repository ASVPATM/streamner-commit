"""Inference-free policy replay and aggregate benchmark metrics."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from streamner_commit.experiments.config import ResearchConfig
from streamner_commit.experiments.policies import PolicySpec, build_policy
from streamner_commit.experiments.traces import ExampleReplay, TraceCondition
from streamner_commit.metrics import (
    MaskingMetrics,
    StrictNERMetrics,
    boundary_revision_metrics,
    commitment_delay_record,
    commitment_outcomes,
    decoded_snapshot_churn,
    first_exact_detection_step,
    gold_visibility_point,
    piimb_masking_metrics,
    premature_prefix_rates,
    strict_ner_metrics,
)
from streamner_commit.policies import simulate_commitments


@dataclass(frozen=True, slots=True)
class ExampleEvaluation:
    task: str
    parent_id: str
    strict: StrictNERMetrics
    masking: MaskingMetrics
    commitment_count: int
    correct_commitments: int
    wrong_commitments: int
    gold_count: int
    missed_entities: int
    gold_premature: int
    silver_premature: int
    blocked_revisions: int
    context_delays: tuple[int, ...]
    update_delays: tuple[int, ...]
    model_delays: tuple[int, ...]
    policy_delays: tuple[int, ...]
    churn_difference: int
    churn_union: int
    boundary_revisions: int
    boundary_comparisons: int

    @property
    def selection_error_numerator(self) -> int:
        return self.wrong_commitments + self.gold_premature + self.missed_entities

    @property
    def selection_error_denominator(self) -> int:
        return self.commitment_count + self.gold_count


@dataclass(frozen=True, slots=True)
class ConditionEvaluation:
    spec: PolicySpec
    split: str
    chunk_words: int
    aggregate_rows: tuple[dict[str, object], ...]
    examples: tuple[ExampleEvaluation, ...]

    @property
    def overall(self) -> Mapping[str, object]:
        return next(row for row in self.aggregate_rows if row["aggregation"] == "overall")


def evaluate_policy_condition(
    condition: TraceCondition,
    spec: PolicySpec,
    config: ResearchConfig,
) -> ConditionEvaluation:
    """Replay one policy independently over every example in a cached condition."""

    threshold = _number(config.metrics.get("model_detection_threshold"), "detection threshold")
    examples = tuple(
        _evaluate_example(example, spec, threshold=threshold) for example in condition.examples
    )
    rows = _aggregate_rows(examples, spec, split=condition.split, chunk_words=condition.chunk_words)
    return ConditionEvaluation(
        spec=spec,
        split=condition.split,
        chunk_words=condition.chunk_words,
        aggregate_rows=rows,
        examples=examples,
    )


def _evaluate_example(
    replay: ExampleReplay,
    spec: PolicySpec,
    *,
    threshold: float,
) -> ExampleEvaluation:
    policy = build_policy(spec, replay.observations)
    run = simulate_commitments(
        policy,
        replay.observations,
        replay.example.labels,
        allow_analysis_only=spec.analysis_only,
    )
    committed = tuple(
        {
            "example_id": commit.key.example_id,
            "start_char": commit.decision.start_char,
            "end_char": commit.decision.end_char,
            "label": commit.key.label,
            "score": commit.decision.probability,
            "commit_step": commit.commit_step,
        }
        for commit in run.committed
    )
    gold = tuple(entity.to_dict() for entity in replay.gold_entities)
    silver = tuple(
        {"example_id": replay.example.example_id, **entity.to_dict()}
        for entity in replay.cold_full.public_entities
    )
    strict = strict_ner_metrics(committed, gold, scope_field="example_id")
    masking = piimb_masking_metrics(committed, gold, text_length=len(replay.example.text))
    outcomes = commitment_outcomes(committed, gold)
    prefixes = premature_prefix_rates(
        committed,
        gold_entities=gold,
        silver_entities=silver,
    )

    snapshot_by_step = {snapshot.step: snapshot for snapshot in replay.snapshots}
    context_delays: list[int] = []
    update_delays: list[int] = []
    model_delays: list[int] = []
    policy_delays: list[int] = []
    for commit in run.committed:
        matching_gold = next(
            (
                entity
                for entity in replay.gold_entities
                if (entity.start_char, entity.end_char, entity.label)
                == (commit.decision.start_char, commit.decision.end_char, commit.key.label)
            ),
            None,
        )
        if matching_gold is None:
            continue
        visibility = gold_visibility_point(matching_gold, replay.snapshots, replay.span_updates)
        detection = first_exact_detection_step(
            matching_gold,
            replay.snapshots,
            threshold=threshold,
        )
        commit_snapshot = snapshot_by_step.get(commit.commit_step)
        if commit_snapshot is None:
            raise ValueError("commit step is absent from cached snapshots")
        delay = commitment_delay_record(
            visibility,
            commit_step=commit.commit_step,
            visible_word_count_at_commit=commit_snapshot.visible_word_count,
            first_detection_step=detection,
        )
        context_delays.append(delay.commit_context_words)
        update_delays.append(delay.update_delay_steps)
        if delay.model_detection_delay_steps is not None:
            model_delays.append(delay.model_detection_delay_steps)
        if delay.policy_added_delay_steps is not None:
            policy_delays.append(delay.policy_added_delay_steps)

    snapshot_rows = tuple(snapshot.to_dict() for snapshot in replay.snapshots)
    churn = decoded_snapshot_churn(snapshot_rows)
    boundaries = boundary_revision_metrics(snapshot_rows)
    task = replay.example.task_name
    parent_id = replay.example.parent_id
    assert task is not None and parent_id is not None
    return ExampleEvaluation(
        task=task,
        parent_id=parent_id,
        strict=strict,
        masking=masking,
        commitment_count=outcomes.commitments,
        correct_commitments=outcomes.correct_commitments,
        wrong_commitments=outcomes.wrong_commitments,
        gold_count=outcomes.gold_entities,
        missed_entities=outcomes.missed_entities,
        gold_premature=prefixes.gold_relative.numerator,
        silver_premature=prefixes.silver_relative.numerator,
        blocked_revisions=run.blocked_revision_count,
        context_delays=tuple(context_delays),
        update_delays=tuple(update_delays),
        model_delays=tuple(model_delays),
        policy_delays=tuple(policy_delays),
        churn_difference=_integer(churn["total_symmetric_difference"]),
        churn_union=sum(_integer(item["union_count"]) for item in churn["transitions"]),
        boundary_revisions=_integer(boundaries["boundary_revision_count"]),
        boundary_comparisons=_integer(boundaries["comparable_transition_count"]),
    )


def _aggregate_rows(
    examples: Sequence[ExampleEvaluation],
    spec: PolicySpec,
    *,
    split: str,
    chunk_words: int,
) -> tuple[dict[str, object], ...]:
    tasks = sorted({example.task for example in examples})
    task_rows = [
        _aggregate_group(
            [example for example in examples if example.task == task],
            spec,
            split=split,
            chunk_words=chunk_words,
            aggregation="task",
            task=task,
        )
        for task in tasks
    ]
    overall = _aggregate_group(
        examples,
        spec,
        split=split,
        chunk_words=chunk_words,
        aggregation="overall",
        task="__overall__",
    )
    for field in (
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "masking_precision",
        "masking_recall",
        "masking_f1",
        "masking_f2",
        "masking_fpr",
        "mean_commit_context_words",
    ):
        values = [_number(row[field], field) for row in task_rows]
        overall[f"task_macro_{field}"] = statistics.fmean(values) if values else 0.0
    return tuple([*task_rows, overall])


def _aggregate_group(
    examples: Sequence[ExampleEvaluation],
    spec: PolicySpec,
    *,
    split: str,
    chunk_words: int,
    aggregation: str,
    task: str,
) -> dict[str, object]:
    strict = StrictNERMetrics(
        sum(item.strict.true_positives for item in examples),
        sum(item.strict.false_positives for item in examples),
        sum(item.strict.false_negatives for item in examples),
    )
    masking = MaskingMetrics(
        sum(item.masking.true_positive_chars for item in examples),
        sum(item.masking.predicted_chars for item in examples),
        sum(item.masking.gold_chars for item in examples),
        sum(item.masking.non_pii_chars for item in examples),
    )
    commitments = sum(item.commitment_count for item in examples)
    correct = sum(item.correct_commitments for item in examples)
    wrong = sum(item.wrong_commitments for item in examples)
    gold = sum(item.gold_count for item in examples)
    missed = sum(item.missed_entities for item in examples)
    premature = sum(item.gold_premature for item in examples)
    silver_premature = sum(item.silver_premature for item in examples)
    error_numerator = sum(item.selection_error_numerator for item in examples)
    error_denominator = sum(item.selection_error_denominator for item in examples)
    context = tuple(value for item in examples for value in item.context_delays)
    updates = tuple(value for item in examples for value in item.update_delays)
    model = tuple(value for item in examples for value in item.model_delays)
    policy = tuple(value for item in examples for value in item.policy_delays)
    churn_difference = sum(item.churn_difference for item in examples)
    churn_union = sum(item.churn_union for item in examples)
    boundary_revisions = sum(item.boundary_revisions for item in examples)
    boundary_comparisons = sum(item.boundary_comparisons for item in examples)
    return {
        "policy_id": spec.policy_id,
        "policy_family": spec.family,
        "policy_variant": spec.variant,
        "analysis_only": spec.analysis_only,
        "parameters_json": json.dumps(
            dict(spec.parameters), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "split": split,
        "chunk_words": chunk_words,
        "aggregation": aggregation,
        "task": task,
        "example_count": len(examples),
        "commitment_count": commitments,
        "correct_commitments": correct,
        "wrong_commitments": wrong,
        "gold_entity_count": gold,
        "missed_entity_count": missed,
        "gold_premature_count": premature,
        "silver_premature_count": silver_premature,
        "strict_true_positives": strict.true_positives,
        "strict_false_positives": strict.false_positives,
        "strict_false_negatives": strict.false_negatives,
        "strict_precision": strict.precision,
        "strict_recall": strict.recall,
        "strict_f1": strict.f1,
        "masking_true_positive_chars": masking.true_positive_chars,
        "masking_false_positive_chars": masking.false_positive_chars,
        "masking_predicted_chars": masking.predicted_chars,
        "masking_gold_chars": masking.gold_chars,
        "masking_non_pii_chars": masking.non_pii_chars,
        "masking_precision": masking.precision,
        "masking_recall": masking.recall,
        "masking_f1": masking.f1,
        "masking_f2": masking.f2,
        "masking_fpr": masking.fpr,
        "wrong_commitment_rate": _ratio(wrong, commitments),
        "missed_entity_rate": _ratio(missed, gold),
        "gold_premature_rate": _ratio(premature, commitments),
        "silver_premature_rate": _ratio(silver_premature, commitments),
        "selection_error_numerator": error_numerator,
        "selection_error_denominator": error_denominator,
        "selection_error_rate": _ratio(error_numerator, error_denominator),
        "mean_commit_context_words": _mean(context),
        "median_commit_context_words": _median(context),
        "mean_update_delay_steps": _mean(updates),
        "mean_model_detection_delay_steps": _mean(model),
        "mean_policy_added_delay_steps": _mean(policy),
        "blocked_revision_count": sum(item.blocked_revisions for item in examples),
        "blocked_revisions_per_100_commitments": 100.0
        * _ratio(sum(item.blocked_revisions for item in examples), commitments),
        "snapshot_churn": _ratio(churn_difference, churn_union),
        "boundary_revision_rate": _ratio(boundary_revisions, boundary_comparisons),
    }


def example_scalar_rows(
    evaluation: ConditionEvaluation,
    *,
    selection_mode: str,
) -> tuple[dict[str, object], ...]:
    """Return content-free parent-cluster rows for paired bootstrap callbacks."""

    return tuple(
        {
            "task": item.task,
            "parent_id": item.parent_id,
            "policy_id": evaluation.spec.policy_id,
            "policy_family": evaluation.spec.family,
            "selection_mode": selection_mode,
            "chunk_words": evaluation.chunk_words,
            "error_numerator": item.selection_error_numerator,
            "error_denominator": item.selection_error_denominator,
            "context_delay_sum": sum(item.context_delays),
            "context_delay_count": len(item.context_delays),
        }
        for item in evaluation.examples
    )


def _mean(values: Sequence[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: Sequence[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("metric count must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(value)


__all__ = [
    "ConditionEvaluation",
    "ExampleEvaluation",
    "evaluate_policy_condition",
    "example_scalar_rows",
]
