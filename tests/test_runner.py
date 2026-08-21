"""Tests for the evaluation protocol.

What is worth pinning here is not that ``run_arm`` produces numbers but the
order in which an arm is shown the world, because the protocol is the only thing
standing between the two arms and an unfair comparison. An arm handed an
experience one timestep early could answer a drift task by reading the future,
and an arm consolidated once per task rather than once per timestep would be
rewarded for how many questions happened to share a moment. Neither mistake
moves the resolve rate in a way a reader could recognise, so both are asserted
against the call log of a memory that records what it was asked to do.
"""

from __future__ import annotations

from collections.abc import Iterable

from atlas.bench.world import World
from atlas.config import ExperimentConfig
from atlas.experiment.runner import run_arm
from atlas.types import Experience, RetrievalResult, ScoredItem, Task, TaskFamily


class RecordingMemory:
    """A memory system that records each protocol call in the order it arrives.

    Retrieval hands back every fact ingested so far, so a task can be graded
    against exactly what the arm had been given by the time it was asked.
    """

    def __init__(self) -> None:
        self.name = "recorder"
        self.calls: list[str] = []
        self._fact_ids: set[str] = set()

    def ingest(self, experience: Experience) -> None:
        self.calls.append(f"ingest {experience.id}")
        self._fact_ids.add(experience.fact_id)

    def retrieve(self, query_tokens: tuple[str, ...], top_k: int) -> RetrievalResult:
        self.calls.append("retrieve")
        return RetrievalResult.from_items(
            (ScoredItem(concept_id="everything", score=1.0, fact_ids=frozenset(self._fact_ids)),)
        )

    def consolidate(self, timestep: int) -> None:
        self.calls.append(f"consolidate {timestep}")

    def statistics(self) -> dict[str, float]:
        return {"concepts": float(len(self._fact_ids))}


def _world(*timesteps: int) -> World:
    """A world carrying one experience per timestep, in the order given."""
    experiences = tuple(
        Experience(
            id=f"exp-{step:02d}",
            fact_id=f"fact-{step:02d}",
            text=f"subject value {step}",
            tokens=("subject", "value", str(step)),
            subject="subject",
            timestep=step,
        )
        for step in timesteps
    )
    return World(
        facts={},
        experiences=experiences,
        stable_by_subject={},
        crowded_ids=frozenset(),
        drift_pairs=(),
        distractor_ids=frozenset(),
    )


def _task(index: int, available_at: int, required: Iterable[str] = ()) -> Task:
    return Task(
        id=f"task-{index:04d}",
        family=TaskFamily.DIRECT,
        problem_statement=f"question {index}",
        query_tokens=("subject", "value"),
        required_fact_ids=frozenset(required),
        forbidden_fact_ids=frozenset(),
        available_at_timestep=available_at,
    )


def _ingests(memory: RecordingMemory) -> list[str]:
    return [call for call in memory.calls if call.startswith("ingest")]


def _consolidations(memory: RecordingMemory) -> list[str]:
    return [call for call in memory.calls if call.startswith("consolidate")]


def test_each_step_delivers_the_arrived_experiences_then_consolidates_then_asks() -> None:
    """The whole protocol in one assertion: what arrives, in what order, and what never does.

    Nothing stamped at or after a task's own timestep may reach the arm before
    that task is asked, so ``exp-06`` is never delivered at all.
    """
    memory = RecordingMemory()

    run_arm(memory, _world(1, 2, 3, 4, 5, 6), (_task(0, 3), _task(1, 6)), ExperimentConfig())

    assert memory.calls == [
        "ingest exp-01",
        "ingest exp-02",
        "consolidate 3",
        "retrieve",
        "ingest exp-03",
        "ingest exp-04",
        "ingest exp-05",
        "consolidate 6",
        "retrieve",
    ]


def test_an_experience_stamped_at_a_tasks_own_timestep_is_withheld_until_after_it() -> None:
    """The off-by-one that would let an arm read the answer out of the future."""
    tasks = (_task(0, 3, required={"fact-03"}), _task(1, 4, required={"fact-03"}))

    result = run_arm(RecordingMemory(), _world(1, 2, 3), tasks, ExperimentConfig())

    assert result.outcomes[0].missing_fact_ids == frozenset({"fact-03"})
    assert result.outcomes[1].resolved


def test_consolidation_runs_once_per_distinct_timestep_rather_than_once_per_task() -> None:
    """Three questions sharing a moment are one consolidation, at that moment.

    Once per task would pay an arm for how many questions happened to collide,
    and consolidating at a distant future step would age every concept past the
    prune threshold at once and collapse the store.
    """
    memory = RecordingMemory()
    tasks = (_task(0, 3), _task(1, 3), _task(2, 3), _task(3, 6), _task(4, 6))

    run_arm(memory, _world(1, 2, 3, 4, 5), tasks, ExperimentConfig())

    assert _consolidations(memory) == ["consolidate 3", "consolidate 6"]


def test_experiences_are_delivered_oldest_first_whatever_order_the_world_holds_them_in() -> None:
    memory = RecordingMemory()

    run_arm(memory, _world(3, 1, 2), (_task(0, 2), _task(1, 4)), ExperimentConfig())

    assert _ingests(memory) == ["ingest exp-01", "ingest exp-02", "ingest exp-03"]


def test_every_task_produces_one_outcome_in_task_order() -> None:
    tasks = (_task(0, 2), _task(1, 3), _task(2, 3), _task(3, 5))

    result = run_arm(RecordingMemory(), _world(1, 2, 3, 4), tasks, ExperimentConfig(seed=11))

    assert [outcome.task_id for outcome in result.outcomes] == [task.id for task in tasks]
    assert result.arm == "recorder"
    assert result.seed == 11
