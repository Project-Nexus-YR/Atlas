"""Tests for the grader.

The signature test at the bottom is the regression guard for the bug that
motivated this rebuild: a grader that read a field the agent had written scored
every run at 100%.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable

from atlas.bench.grader import grade
from atlas.types import RetrievalResult, ScoredItem, Task, TaskFamily


def _task(required: Iterable[str], forbidden: Iterable[str] = ()) -> Task:
    return Task(
        id="task-0001",
        family=TaskFamily.DRIFT,
        problem_statement="report the current value",
        query_tokens=("cache", "timeout", "window"),
        required_fact_ids=frozenset(required),
        forbidden_fact_ids=frozenset(forbidden),
        available_at_timestep=9,
    )


def _returning(fact_ids: Iterable[str]) -> RetrievalResult:
    return RetrievalResult.from_items(
        (ScoredItem(concept_id="probe", score=1.0, fact_ids=frozenset(fact_ids)),),
        expanded_nodes=4,
    )


def test_exact_match_resolves() -> None:
    outcome = grade(_task({"f1"}), _returning({"f1"}), 1.5)

    assert outcome.resolved


def test_empty_retrieval_reports_every_required_fact_as_missing() -> None:
    outcome = grade(_task({"f1", "f2"}), RetrievalResult.from_items(()), 1.5)

    assert not outcome.resolved
    assert outcome.missing_fact_ids == frozenset({"f1", "f2"})


def test_extra_harmless_facts_do_not_prevent_resolution() -> None:
    outcome = grade(_task({"f1"}), _returning({"f1", "noise-1", "noise-2"}), 1.5)

    assert outcome.resolved


def test_superset_containing_a_forbidden_fact_does_not_resolve() -> None:
    task = _task({"f2"}, forbidden={"f1"})

    outcome = grade(task, _returning({"f1", "f2", "noise-1"}), 1.5)

    assert not outcome.resolved


def test_a_retrieved_forbidden_fact_is_named_in_the_stale_set() -> None:
    task = _task({"f2"}, forbidden={"f1"})

    outcome = grade(task, _returning({"f1", "f2"}), 1.5)

    assert outcome.stale_fact_ids == frozenset({"f1"})
    assert outcome.missing_fact_ids == frozenset()


def test_missing_one_of_two_required_facts_does_not_resolve() -> None:
    outcome = grade(_task({"f1", "f2"}), _returning({"f1"}), 1.5)

    assert not outcome.resolved


def test_the_missing_set_names_only_the_absent_fact() -> None:
    outcome = grade(_task({"f1", "f2"}), _returning({"f1"}), 1.5)

    assert outcome.missing_fact_ids == frozenset({"f2"})


def test_outcome_carries_the_audit_trail_of_the_attempt() -> None:
    outcome = grade(_task({"f1"}), _returning({"f1", "f9"}), 12.5)

    assert outcome.retrieved_fact_ids == frozenset({"f1", "f9"})
    assert outcome.expanded_nodes == 4
    assert outcome.latency_ms == 12.5


def test_grade_accepts_no_agent_authored_input() -> None:
    # The anti-rigging invariant is structural: there is no parameter through
    # which an agent's own claim of success could reach the verdict.
    parameters = list(inspect.signature(grade).parameters)

    assert parameters == ["task", "result", "latency_ms"]
