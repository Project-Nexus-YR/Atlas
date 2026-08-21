"""Scoring for one task instance.

ANTI-RIGGING INVARIANT: the grader reads the task and the retrieval result, and
nothing else. It never sees a status, a message, a patch, or any other value the
agent authored. That is deliberate and load-bearing. An earlier harness graded a
field the agent itself set, and so reported a perfect score for an agent that
did nothing; the only structural defence is a grader with no parameter through
which such a claim could arrive. Adding one would void every number this
benchmark produces.

The verdict is SYNTHETIC in the same sense as the world: it measures whether the
right fact ids came back, which is a proxy for grounded action, not the thing
itself. An agent can retrieve perfectly and still act badly, and this grader
would not notice.
"""

from __future__ import annotations

from atlas.types import RetrievalResult, Task, TaskOutcome

__all__ = ["grade"]


def grade(task: Task, result: RetrievalResult, latency_ms: float) -> TaskOutcome:
    """Resolve iff every required fact came back and no forbidden one did.

    The two conditions pull against each other on purpose: retrieving more raises
    recall of the required facts and raises the risk of dragging in a superseded
    one. ``missing_fact_ids`` and ``stale_fact_ids`` name which of the two failed,
    so an unresolved instance can be audited rather than merely counted.
    """
    missing = task.required_fact_ids - result.fact_ids
    stale = task.forbidden_fact_ids & result.fact_ids
    return TaskOutcome(
        task_id=task.id,
        family=task.family,
        resolved=not missing and not stale,
        retrieved_fact_ids=result.fact_ids,
        missing_fact_ids=missing,
        stale_fact_ids=stale,
        expanded_nodes=result.expanded_nodes,
        novel_items=result.novel_items,
        latency_ms=latency_ms,
    )
