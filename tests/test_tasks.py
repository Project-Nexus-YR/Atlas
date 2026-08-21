"""Tests for the task suite.

Two of these are load-bearing for the whole project. The oracle test proves the
suite is winnable, so a low score means bad retrieval rather than an impossible
benchmark. The whole-corpus test proves it is not winnable by brute force, which
is the failure mode that made the previous harness worthless.
"""

from __future__ import annotations

from collections.abc import Iterable

from atlas.bench.grader import grade
from atlas.bench.tasks import build_tasks
from atlas.bench.world import World, build_world
from atlas.config import WorldConfig
from atlas.types import RetrievalResult, ScoredItem, Task, TaskFamily

_CONFIG = WorldConfig(
    subjects=6, facts_per_subject=4, drift_rate=0.25, distractor_ratio=2.0, tasks=40
)


def _suite(seed: int = 17) -> tuple[World, tuple[Task, ...]]:
    world = build_world(_CONFIG, seed)
    return world, build_tasks(world, _CONFIG, seed)


def _returning(fact_ids: Iterable[str]) -> RetrievalResult:
    """A retrieval result carrying exactly these facts, as any arm would deliver."""
    return RetrievalResult.from_items(
        (ScoredItem(concept_id="probe", score=1.0, fact_ids=frozenset(fact_ids)),)
    )


def test_every_task_is_solvable_by_an_oracle() -> None:
    _, tasks = _suite()

    outcomes = [grade(task, _returning(task.required_fact_ids), 0.0) for task in tasks]

    assert all(outcome.resolved for outcome in outcomes)


def test_an_arm_returning_everything_fails_drift_tasks() -> None:
    world, tasks = _suite()
    everything = _returning(world.facts)

    outcomes = {task.id: grade(task, everything, 0.0) for task in tasks}

    drift = [o for o in outcomes.values() if o.family is TaskFamily.DRIFT]
    assert drift, "no drift tasks were generated, so the test proves nothing"
    assert not any(outcome.resolved for outcome in drift)
    assert all(outcome.stale_fact_ids for outcome in drift)


def test_an_arm_returning_everything_still_wins_the_families_without_a_trap() -> None:
    # Stated openly rather than hidden: only DRIFT resists brute force. The other
    # three are hard because the runner caps retrieval at top_k, not absolutely.
    world, tasks = _suite()
    everything = _returning(world.facts)

    untrapped = [task for task in tasks if task.family is not TaskFamily.DRIFT]

    assert all(grade(task, everything, 0.0).resolved for task in untrapped)


def test_an_arm_returning_nothing_scores_zero() -> None:
    _, tasks = _suite()
    empty = RetrievalResult.from_items(())

    outcomes = [grade(task, empty, 0.0) for task in tasks]

    assert not any(outcome.resolved for outcome in outcomes)


def test_required_facts_are_always_available_before_the_task() -> None:
    world, tasks = _suite()

    for task in tasks:
        for fact_id in task.required_fact_ids:
            assert world.facts[fact_id].timestep < task.available_at_timestep


def test_forbidden_facts_have_already_arrived_when_the_task_is_asked() -> None:
    world, tasks = _suite()

    for task in tasks:
        for fact_id in task.forbidden_fact_ids:
            assert world.facts[fact_id].timestep < task.available_at_timestep


def test_same_seed_produces_an_identical_task_suite() -> None:
    _, first = _suite(seed=17)
    _, second = _suite(seed=17)

    assert first == second


def test_different_seeds_produce_different_task_suites() -> None:
    _, first = _suite(seed=17)
    _, second = _suite(seed=18)

    assert [task.query_tokens for task in first] != [task.query_tokens for task in second]


def test_all_four_families_are_represented() -> None:
    _, tasks = _suite()

    families = {task.family for task in tasks}

    assert families == set(TaskFamily)


def test_no_family_exceeds_its_even_share_of_the_requested_total() -> None:
    _, tasks = _suite()
    ceiling = -(-_CONFIG.tasks // len(TaskFamily))

    for family in TaskFamily:
        assert sum(1 for task in tasks if task.family is family) <= ceiling


def test_no_task_requires_a_distractor_fact() -> None:
    world, tasks = _suite()

    for task in tasks:
        assert not task.required_fact_ids & world.distractor_ids


def test_compositional_query_names_one_required_fact_and_not_the_other() -> None:
    world, tasks = _suite()
    compositional = [task for task in tasks if task.family is TaskFamily.COMPOSITIONAL]

    assert compositional
    for task in compositional:
        required = [world.facts[fid] for fid in task.required_fact_ids]
        named = [fact for fact in required if fact.tokens == task.query_tokens]
        unnamed = [fact for fact in required if fact.tokens != task.query_tokens]
        assert len(named) == 1
        assert len(unnamed) == 1
        overlap = set(unnamed[0].tokens) & set(task.query_tokens)
        assert overlap == {unnamed[0].subject}, f"{task.id} leaks its second fact"


def test_drift_query_cannot_discriminate_between_the_stale_and_current_fact() -> None:
    world, tasks = _suite()
    drift = [task for task in tasks if task.family is TaskFamily.DRIFT]

    assert drift
    for task in drift:
        current = world.facts[next(iter(task.required_fact_ids))]
        stale = world.facts[next(iter(task.forbidden_fact_ids))]
        query = set(task.query_tokens)
        assert query <= set(current.tokens)
        assert query <= set(stale.tokens), f"{task.id} favours the current fact lexically"


def test_distractor_target_is_shadowed_by_near_copies() -> None:
    world, tasks = _suite()
    distractor = [task for task in tasks if task.family is TaskFamily.DISTRACTOR]

    assert distractor
    for task in distractor:
        target = world.facts[next(iter(task.required_fact_ids))]
        copies = [
            fact
            for fact in world.facts.values()
            if fact.id != target.id and len(set(fact.tokens) & set(target.tokens)) >= 3
        ]
        assert len(copies) >= 2, f"{task.id} is not actually crowded"


def test_direct_target_is_not_shadowed_by_near_copies() -> None:
    # DIRECT must stay genuinely easy. Sabotaging the family a similarity
    # baseline should win would rig the comparison.
    world, tasks = _suite()
    direct = [task for task in tasks if task.family is TaskFamily.DIRECT]

    assert direct
    for task in direct:
        target = world.facts[next(iter(task.required_fact_ids))]
        rivals = [
            fact
            for fact in world.facts.values()
            if fact.id != target.id
            and fact.subject == target.subject
            and len(set(fact.tokens) & set(target.tokens)) >= 3
        ]
        assert rivals == [], f"{task.id} has a near copy in its own subject"


def test_task_ids_are_unique() -> None:
    _, tasks = _suite()

    assert len({task.id for task in tasks}) == len(tasks)


def test_suite_is_ordered_by_the_timestep_it_may_be_asked_at() -> None:
    _, tasks = _suite()

    steps = [task.available_at_timestep for task in tasks]

    assert steps == sorted(steps)


def test_every_query_token_appears_in_some_experience() -> None:
    """Fairness to the lexical arm, which cannot reach a token it has never seen.

    Vector-RAG matches tokens, not meaning. A query token absent from every
    experience is unreachable for it at any dimensionality, so a gap measured on
    such a task would be evidence about the encoder rather than about
    consolidation. This must hold for every family, including ones added later.
    """
    world, tasks = _suite()
    vocabulary = {token for experience in world.experiences for token in experience.tokens}

    unreachable = {
        task.id: sorted(set(task.query_tokens) - vocabulary)
        for task in tasks
        if set(task.query_tokens) - vocabulary
    }

    assert unreachable == {}


def test_no_task_has_a_single_token_query() -> None:
    """Query-side IDF is inert on one token, which would silently weaken the baseline.

    The baseline weights the query by IDF (SMART ``lnc.ltc``). With a single
    token the IDF scalar multiplies the only nonzero component and cosine
    normalisation divides it straight back out, so a one-token query benchmarks
    the pre-IDF baseline while the README describes the stronger one.
    """
    _, tasks = _suite()

    assert [task.id for task in tasks if len(set(task.query_tokens)) < 2] == []
