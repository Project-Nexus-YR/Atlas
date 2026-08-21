"""Tests for the comparison and the table it renders.

``run_comparison`` is checked for the one thing it exists to guarantee: both
arms driven over one world with one task suite, so the only difference between
the two columns is the memory system.

The table is checked against hand-built results rather than a real run, because
the cases that break a percentage are the degenerate ones - a family the world
had no material for, and a comparison with no tasks at all. Neither appears in a
healthy run, and both are where a rate divides by zero.
"""

from __future__ import annotations

from collections.abc import Iterable

from atlas.config import ExperimentConfig, WorldConfig
from atlas.experiment.compare import Comparison, format_table, run_comparison
from atlas.types import ArmResult, TaskFamily, TaskOutcome

_CONFIG = ExperimentConfig(seed=13, world=WorldConfig(subjects=4, tasks=8))


def _outcomes(family: TaskFamily, resolved: int, total: int) -> list[TaskOutcome]:
    return [
        TaskOutcome(
            task_id=f"task-{family.value}-{index}",
            family=family,
            resolved=index < resolved,
            retrieved_fact_ids=frozenset(),
            missing_fact_ids=frozenset(),
            stale_fact_ids=frozenset(),
            expanded_nodes=0,
            novel_items=0,
            latency_ms=1.0,
        )
        for index in range(total)
    ]


def _comparison(kee: Iterable[TaskOutcome], vector: Iterable[TaskOutcome]) -> Comparison:
    return Comparison(
        config=ExperimentConfig(),
        kee=ArmResult(arm="kee", seed=42, outcomes=tuple(kee)),
        vector=ArmResult(arm="vector-rag", seed=42, outcomes=tuple(vector)),
        kee_stats={"concepts": 9.0, "revisions": 2.0, "merges": 1.0},
        vector_stats={"concepts": 9.0, "vectors": 9.0, "dim": 256.0},
    )


def _row(table: str, label: str) -> list[str] | None:
    """The whitespace-split fields of the row headed ``label``, or None if absent."""
    for line in table.splitlines():
        if line.startswith(label):
            return line.split()
    return None


def test_both_arms_are_asked_the_identical_task_suite() -> None:
    comparison = run_comparison(_CONFIG)

    assert comparison.kee.outcomes, "the suite was empty, so the test proves nothing"
    assert [(o.task_id, o.family) for o in comparison.kee.outcomes] == [
        (o.task_id, o.family) for o in comparison.vector.outcomes
    ]


def test_each_arm_reports_its_own_end_state_statistics() -> None:
    comparison = run_comparison(_CONFIG)

    assert "revisions" in comparison.kee_stats
    assert "revisions" not in comparison.vector_stats
    assert comparison.vector_stats["dim"] == float(_CONFIG.embedding_dim)


def test_only_the_families_the_world_had_material_for_get_a_row() -> None:
    outcomes = _outcomes(TaskFamily.DIRECT, 1, 2) + _outcomes(TaskFamily.DRIFT, 0, 2)

    table = format_table(_comparison(outcomes, outcomes))

    assert _row(table, "direct") is not None
    assert _row(table, "drift") is not None
    assert _row(table, "distractor") is None
    assert _row(table, "compositional") is None


def test_the_overall_row_totals_across_every_family() -> None:
    kee = _outcomes(TaskFamily.DIRECT, 2, 4) + _outcomes(TaskFamily.DRIFT, 1, 4)
    vector = _outcomes(TaskFamily.DIRECT, 4, 4) + _outcomes(TaskFamily.DRIFT, 1, 4)

    table = format_table(_comparison(kee, vector))

    assert _row(table, "direct") == ["direct", "4", "50.0%", "100.0%", "-50.0"]
    assert _row(table, "overall") == ["overall", "8", "37.5%", "62.5%", "-25.0"]


def test_a_family_the_other_arm_never_saw_scores_zero_rather_than_dividing_by_zero() -> None:
    kee = _outcomes(TaskFamily.DIRECT, 1, 2) + _outcomes(TaskFamily.DRIFT, 0, 2)

    table = format_table(_comparison(kee, _outcomes(TaskFamily.DIRECT, 2, 2)))

    assert _row(table, "drift") == ["drift", "2", "0.0%", "0.0%", "+0.0"]


def test_a_comparison_with_no_tasks_renders_an_empty_table_rather_than_raising() -> None:
    table = format_table(_comparison([], []))

    assert _row(table, "overall") == ["overall", "0", "0.0%", "0.0%", "+0.0"]
    assert _row(table, "direct") is None
