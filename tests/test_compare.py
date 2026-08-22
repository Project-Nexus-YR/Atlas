"""Tests for the comparison and the table it renders.

``run_comparison`` is checked for the one thing it exists to guarantee: every
arm driven over one world with one task suite, so the only difference between
the columns is the memory system.

The table is checked against hand-built results rather than a real run, because
the cases that break a percentage are the degenerate ones - a family the world
had no material for, and a comparison with no tasks at all. Neither appears in a
healthy run, and both are where a rate divides by zero. The delta is checked
against two controls at once, because measuring it against the weaker one would
flatter the arm under test on exactly the families that matter.
"""

from __future__ import annotations

from collections.abc import Iterable

from atlas.config import ExperimentConfig, WorldConfig
from atlas.experiment.compare import ArmReport, Comparison, format_table, run_comparison
from atlas.types import ArmResult, TaskFamily, TaskOutcome

_CONFIG = ExperimentConfig(seed=13, world=WorldConfig(subjects=4, tasks=8))
_CONTROL_LABELS = ("Vector-RAG", "BM25")


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


def _report(label: str, outcomes: Iterable[TaskOutcome], stats: dict[str, float]) -> ArmReport:
    return ArmReport(
        label=label, result=ArmResult(arm=label, seed=42, outcomes=tuple(outcomes)), stats=stats
    )


def _comparison(subject: Iterable[TaskOutcome], *controls: Iterable[TaskOutcome]) -> Comparison:
    return Comparison(
        config=ExperimentConfig(),
        subject=_report("KEE", subject, {"concepts": 9.0, "revisions": 2.0, "merges": 1.0}),
        baselines=tuple(
            _report(_CONTROL_LABELS[index], outcomes, {"concepts": 9.0, "documents": 9.0})
            for index, outcomes in enumerate(controls)
        ),
    )


def _row(table: str, label: str) -> list[str] | None:
    """The whitespace-split fields of the row headed ``label``, or None if absent."""
    for line in table.splitlines():
        if line.startswith(label):
            return line.split()
    return None


def test_every_arm_is_asked_the_identical_task_suite() -> None:
    comparison = run_comparison(_CONFIG)
    subject = [(o.task_id, o.family) for o in comparison.subject.result.outcomes]

    assert subject, "the suite was empty, so the test proves nothing"
    for control in comparison.baselines:
        assert [(o.task_id, o.family) for o in control.result.outcomes] == subject


def test_each_arm_reports_its_own_end_state_statistics() -> None:
    comparison = run_comparison(_CONFIG)
    controls = {arm.label: arm.stats for arm in comparison.baselines}

    assert "revisions" in comparison.subject.stats
    assert not any("revisions" in stats for stats in controls.values())
    assert controls["Vector-RAG"]["dim"] == float(_CONFIG.embedding_dim)
    assert controls["BM25"]["documents"] == controls["BM25"]["concepts"]


def test_every_arm_gets_its_own_column() -> None:
    outcomes = _outcomes(TaskFamily.DIRECT, 1, 2)

    header = format_table(_comparison(outcomes, outcomes, outcomes)).splitlines()[2]

    assert header.split() == ["family", "n", "KEE", "Vector-RAG", "BM25", "vs", "best"]


def test_only_the_families_the_world_had_material_for_get_a_row() -> None:
    outcomes = _outcomes(TaskFamily.DIRECT, 1, 2) + _outcomes(TaskFamily.DRIFT, 0, 2)

    table = format_table(_comparison(outcomes, outcomes))

    assert _row(table, "direct") is not None
    assert _row(table, "drift") is not None
    assert _row(table, "distractor") is None
    assert _row(table, "compositional") is None


def test_the_overall_row_totals_across_every_family() -> None:
    subject = _outcomes(TaskFamily.DIRECT, 2, 4) + _outcomes(TaskFamily.DRIFT, 1, 4)
    control = _outcomes(TaskFamily.DIRECT, 4, 4) + _outcomes(TaskFamily.DRIFT, 1, 4)

    table = format_table(_comparison(subject, control))

    assert _row(table, "direct") == ["direct", "4", "50.0%", "100.0%", "-50.0"]
    assert _row(table, "overall") == ["overall", "8", "37.5%", "62.5%", "-25.0"]


def test_the_delta_is_measured_against_the_strongest_control() -> None:
    """A second, weaker control must not be allowed to improve the reported margin."""
    subject = _outcomes(TaskFamily.DIRECT, 2, 4)
    strong = _outcomes(TaskFamily.DIRECT, 4, 4)
    weak = _outcomes(TaskFamily.DIRECT, 0, 4)

    table = format_table(_comparison(subject, strong, weak))

    assert _row(table, "direct") == ["direct", "4", "50.0%", "100.0%", "0.0%", "-50.0"]


def test_a_family_the_other_arm_never_saw_scores_zero_rather_than_dividing_by_zero() -> None:
    subject = _outcomes(TaskFamily.DIRECT, 1, 2) + _outcomes(TaskFamily.DRIFT, 0, 2)

    table = format_table(_comparison(subject, _outcomes(TaskFamily.DIRECT, 2, 2)))

    assert _row(table, "drift") == ["drift", "2", "0.0%", "0.0%", "+0.0"]


def test_a_comparison_with_no_tasks_renders_an_empty_table_rather_than_raising() -> None:
    table = format_table(_comparison([], []))

    assert _row(table, "overall") == ["overall", "0", "0.0%", "0.0%", "+0.0"]
    assert _row(table, "direct") is None
