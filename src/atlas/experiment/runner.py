"""The evaluation protocol: what each arm is shown, and when.

Both arms are driven through this one function, so neither can be given an
advantage by the harness. The protocol is a single pass along the timeline:

1. Deliver every experience whose timestep has arrived.
2. Let the arm consolidate, once per distinct timestep rather than once per task,
   so an arm is not rewarded or punished for how many questions happen to share
   a moment.
3. Ask the questions available at that moment, and grade each answer.

The ordering matters and is not cosmetic. A task is asked only after the
evidence that answers it has been ingested - the task builder asserts this when
it stamps ``available_at_timestep`` - and an arm never sees a future experience.
Ingesting the whole corpus up front and only then asking would erase the drift
family entirely, because a superseded fact and its replacement would arrive
together and nothing would have to be revised.

Consolidation is called at the current timeline position, not at some distant
future step. That is deliberate: utility decays exponentially with age, so
consolidating far past the last experience drives every concept below the prune
threshold at once and the store collapses to the single concept its never-empty
guard preserves. A collapsed store is a harness artefact, not a result.
"""

from __future__ import annotations

from collections import deque
from time import perf_counter

from atlas.bench.grader import grade
from atlas.bench.world import World
from atlas.config import ExperimentConfig
from atlas.logging import get_logger
from atlas.types import ArmResult, MemorySystem, Task, TaskOutcome

__all__ = ["run_arm"]

_log = get_logger(__name__)


def run_arm(
    memory: MemorySystem, world: World, tasks: tuple[Task, ...], config: ExperimentConfig
) -> ArmResult:
    """Drive one memory arm through the whole timeline and grade every task.

    ``tasks`` must be ordered by ``available_at_timestep``; the task builder
    guarantees it and a test pins it. The arm is handed nothing but experiences
    and query tokens, so it cannot read ``Fact.supersedes`` and cannot learn
    which of two similar claims the grader considers current.
    """
    pending = deque(sorted(world.experiences, key=lambda experience: experience.timestep))
    outcomes: list[TaskOutcome] = []
    consolidated_at = -1

    for task in tasks:
        while pending and pending[0].timestep < task.available_at_timestep:
            memory.ingest(pending.popleft())
        if task.available_at_timestep != consolidated_at:
            memory.consolidate(task.available_at_timestep)
            consolidated_at = task.available_at_timestep

        started = perf_counter()
        result = memory.retrieve(task.query_tokens, config.top_k)
        latency_ms = (perf_counter() - started) * 1000.0
        outcomes.append(grade(task, result, latency_ms))

    _log.info(
        "%s: %d/%d resolved, %d experiences never delivered",
        memory.name,
        sum(1 for outcome in outcomes if outcome.resolved),
        len(outcomes),
        len(pending),
    )
    return ArmResult(arm=memory.name, seed=config.seed, outcomes=tuple(outcomes))
