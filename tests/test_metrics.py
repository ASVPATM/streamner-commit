from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest

from streamner_commit.metrics import (
    MaskingMetrics,
    StrictNERMetrics,
    aggregate_masking_by_task,
    aggregate_ner_by_task,
    blocked_revision_metrics,
    boundary_revision_metrics,
    commitment_delay_record,
    commitment_outcomes,
    first_exact_detection_step,
    gold_visibility_point,
    material_revision_horizon_metrics,
    merge_half_open_intervals,
    paired_stratified_bootstrap,
    pareto_front,
    piimb_masking_metrics,
    premature_prefix_rates,
    strict_ner_metrics,
    trajectory_stability_metrics,
)


def span(start: int, end: int, label: str = "person", **extra: object) -> dict[str, object]:
    return {"start_char": start, "end_char": end, "label": label, **extra}


def test_strict_ner_is_exact_and_duplicate_safe() -> None:
    predicted = [span(0, 5), span(0, 5), span(8, 12, "place")]
    gold = [span(0, 5), span(8, 13, "place")]

    metrics = strict_ner_metrics(predicted, gold)

    assert metrics == StrictNERMetrics(1, 1, 1)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_piimb_coverage_merges_adjacent_spans_and_uses_additive_char_counts() -> None:
    assert merge_half_open_intervals([(0, 2), (2, 4), (8, 10), (9, 12)]) == (
        (0, 4),
        (8, 12),
    )

    metrics = piimb_masking_metrics(
        [(0, 2), (2, 4), (8, 12)],
        [(1, 5), (10, 12)],
        text_length=15,
    )

    assert metrics.true_positive_chars == 5
    assert metrics.predicted_chars == 8
    assert metrics.gold_chars == 6
    assert metrics.false_positive_chars == 3
    assert metrics.non_pii_chars == 9
    assert metrics.precision == pytest.approx(5 / 8)
    assert metrics.recall == pytest.approx(5 / 6)
    assert metrics.f2 == pytest.approx((5 * (5 / 8) * (5 / 6)) / (4 * (5 / 8) + 5 / 6))
    assert metrics.fpr == pytest.approx(1 / 3)


def test_zero_denominators_and_task_micro_then_unweighted_macro_are_explicit() -> None:
    empty = piimb_masking_metrics([], [], text_length=0)
    assert empty.to_dict() == {
        "true_positive_chars": 0,
        "false_positive_chars": 0,
        "predicted_chars": 0,
        "gold_chars": 0,
        "non_pii_chars": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "f2": 0.0,
        "fpr": 0.0,
    }

    ner = aggregate_ner_by_task(
        {"large": [StrictNERMetrics(9, 1, 1)], "small": [StrictNERMetrics(0, 1, 1)]}
    )
    assert ner.pooled_micro == StrictNERMetrics(9, 2, 2)
    assert ner.task_macro.precision == pytest.approx((0.9 + 0.0) / 2)
    assert ner.task_macro.recall == pytest.approx((0.9 + 0.0) / 2)

    masking = aggregate_masking_by_task(
        {
            "large": [MaskingMetrics(90, 100, 100, 900)],
            "small": [MaskingMetrics(0, 0, 1, 9)],
        }
    )
    assert masking.pooled_micro.recall == pytest.approx(90 / 101)
    assert masking.task_macro.recall == pytest.approx(0.45)


def test_gold_visibility_word_mapping_and_delay_decomposition_are_off_by_one_safe() -> None:
    gold = span(0, 13)
    steps = [
        {"step": 0, "visible_char_count": 5},
        {"step": 1, "visible_char_count": 13},
        {"step": 2, "visible_char_count": 18},
    ]
    point = gold_visibility_point(gold, steps, [(0, 5), (6, 13), (14, 18)])
    snapshots = [
        {"step": 1, "public_entities": [span(0, 13, score=0.4)]},
        {"step": 2, "public_entities": [span(0, 13, score=0.6)]},
    ]
    detection = first_exact_detection_step(gold, snapshots, threshold=0.5)
    delay = commitment_delay_record(
        point,
        commit_step=3,
        visible_word_count_at_commit=4,
        first_detection_step=detection,
    )

    assert point.gold_visible_step == 1
    assert point.gold_end_word == 1
    assert detection == 2
    assert delay.commit_context_words == 2
    assert delay.update_delay_steps == 2
    assert delay.model_detection_delay_steps == 1
    assert delay.policy_added_delay_steps == 1


def test_gold_and_silver_premature_prefix_definitions_are_not_collapsed() -> None:
    commitments = [span(0, 5), span(20, 25, "place")]
    gold = [span(0, 13), span(20, 25, "place")]
    cold_silver = [span(0, 5), span(20, 30, "place")]

    prefix = premature_prefix_rates(
        commitments,
        gold_entities=gold,
        silver_entities=cold_silver,
    )
    outcomes = commitment_outcomes(commitments, gold)

    assert prefix.gold_relative.numerator == 1
    assert prefix.silver_relative.numerator == 1
    assert prefix.gold_relative.rate == 0.5
    # Different records trigger the two definitions: Sarah for gold, place for silver.
    assert outcomes.correct_commitments == 1
    assert outcomes.wrong_commitments == 1
    assert outcomes.missed_entities == 1
    assert outcomes.precision == 0.5
    assert outcomes.wrong_commitment_rate == 0.5
    assert outcomes.missed_entity_rate == 0.5


def test_blocked_revisions_and_boundary_revisions_have_explicit_denominators() -> None:
    blocked = blocked_revision_metrics([{}, {}], example_count=2, commitment_count=4)
    assert blocked.per_example == 1.0
    assert blocked.per_committed_entity == 0.5
    assert blocked.per_100_commitments == 50.0

    snapshots = [
        {"step": 0, "public_entities": [span(0, 5, score=0.8)]},
        {"step": 1, "public_entities": [span(0, 13, score=0.9)]},
        {"step": 2, "public_entities": []},
        {"step": 3, "public_entities": [span(0, 13, score=0.9)]},
    ]
    revisions = boundary_revision_metrics(snapshots)
    assert revisions["family_count"] == 1
    assert revisions["revised_family_count"] == 1
    assert revisions["boundary_revision_count"] == 1
    assert revisions["boundary_revision_rate"] == 1.0


def test_material_horizon_and_trajectory_stability_use_actual_observations_only() -> None:
    updates = [
        {
            "step": 0,
            "start_word": 0,
            "end_word": 0,
            "visible_word_count": 1,
            "update_kind": "new",
            "top_label": "person",
            "top_probability_delta": None,
        },
        {
            "step": 1,
            "start_word": 0,
            "end_word": 0,
            "visible_word_count": 2,
            "update_kind": "rescore",
            "top_label": "person",
            "top_probability_delta": 0.01,
        },
        {
            "step": 2,
            "start_word": 0,
            "end_word": 0,
            "visible_word_count": 4,
            "update_kind": "rescore",
            "top_label": "person",
            "top_probability_delta": 0.1,
        },
    ]
    horizon = material_revision_horizon_metrics(updates, minimum_probability_delta=0.05)
    assert horizon["values_words"] == [3.0]

    stability = trajectory_stability_metrics(
        [
            {"top_probability": 0.1, "rank": 2},
            {"top_probability": 0.3, "rank": 1},
            {"top_probability": 0.2, "rank": 1},
            {"top_probability": 0.4, "rank": 2},
        ]
    )
    assert stability.variance == pytest.approx(0.0125)
    assert stability.max_absolute_delta == pytest.approx(0.2)
    assert stability.mean_absolute_delta == pytest.approx(1 / 6)
    assert stability.delta_sign_change_count == 2
    assert stability.rank_change_count == 2


def test_paired_parent_bootstrap_is_stratified_deterministic_and_input_order_invariant() -> None:
    rows = [
        {"task": "a", "parent_id": "a-1", "baseline": 0.0, "candidate": 1.0},
        {"task": "a", "parent_id": "a-1", "baseline": 2.0, "candidate": 3.0},
        {"task": "a", "parent_id": "a-2", "baseline": 4.0, "candidate": 5.0},
        {"task": "b", "parent_id": "b-1", "baseline": 6.0, "candidate": 7.0},
    ]

    def task_macro(records: Sequence[Mapping[str, Any]]) -> float:
        tasks = {str(row["task"]) for row in records}
        return statistics.fmean(
            statistics.fmean(
                cast(float, row["candidate"]) for row in records if row["task"] == task
            )
            for task in tasks
        )

    def baseline_task_macro(records: Sequence[Mapping[str, Any]]) -> float:
        tasks = {str(row["task"]) for row in records}
        return statistics.fmean(
            statistics.fmean(cast(float, row["baseline"]) for row in records if row["task"] == task)
            for task in tasks
        )

    first = paired_stratified_bootstrap(
        rows,
        baseline_statistic=baseline_task_macro,
        comparison_statistic=task_macro,
        replicates=50,
        seed=7,
    )
    second = paired_stratified_bootstrap(
        list(reversed(rows)),
        baseline_statistic=baseline_task_macro,
        comparison_statistic=task_macro,
        replicates=50,
        seed=7,
    )

    assert first == second
    assert first.difference == 1.0
    assert first.lower == 1.0
    assert first.upper == 1.0


def test_pareto_front_is_deterministic_and_keeps_tradeoff_points() -> None:
    rows = [
        {"name": "slow-clean", "delay": 2.0, "error": 0.1},
        {"name": "dominated", "delay": 2.0, "error": 0.3},
        {"name": "middle", "delay": 1.5, "error": 0.15},
        {"name": "fast-noisy", "delay": 1.0, "error": 0.2},
    ]

    front = pareto_front(rows, minimize=("delay", "error"))
    reversed_front = pareto_front(list(reversed(rows)), minimize=("delay", "error"))

    assert front == reversed_front
    assert [row["name"] for row in front] == ["fast-noisy", "middle", "slow-clean"]
