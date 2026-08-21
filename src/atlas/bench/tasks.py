"""The task suite: four families, each hard for one nameable reason.

These are SYNTHETIC retrieval tasks. An instance is a query and a set of facts
that must come back; there is no code to write and no execution to observe, so a
score here is evidence about retrieval and about nothing else. Do not read a
result as a measure of agent competence.

The families are written as fair descriptions of retrieval difficulty, not as
traps set for one arm. What they measure was checked before publication, on the
default config at seed 42 against a plain token-overlap arm at ``top_k`` 5, and
the profile bounds every later claim:

* DIRECT 100% and DISTRACTOR 100%. Both are solved by similarity alone, and are
  controls rather than challenges. DIRECT proves the suite is answerable at all.
  DISTRACTOR holds the query form fixed and crowds the neighbourhood instead, so
  it is the family an arm loses by over-consolidating - by blurring near
  identical facts into one concept and forgetting which value was which. It is
  the baseline's family to win.
* COMPOSITIONAL 0%. The second fact shares only the subject word with the query,
  so nothing lexical points at it. This is the family most open to the charge of
  being built for a graph, because the hop it needs is exactly one subject edge.
* DRIFT 0%. The query holds only tokens both the stale and the current fact
  carry, so similarity cannot rank one above the other, and returning both
  fails. Nothing in the experience stream marks a fact as a replacement, so an
  arm has to infer drift from near-identical text arriving later.

Only DRIFT resists brute force. The other three carry no forbidden facts, so an
arm returning the whole corpus resolves all of them; their difficulty is
competition for a capped ``top_k``, which is a real constraint but a tunable one.

A family stops when the world runs out of material for it, so the suite may be
shorter than ``config.tasks``. Recycling instances to hit a round number would
manufacture statistical power that the world does not contain.
"""

from __future__ import annotations

import random

from atlas.bench.world import World
from atlas.config import WorldConfig
from atlas.types import Fact, Task, TaskFamily

__all__ = ["build_tasks"]


def build_tasks(world: World, config: WorldConfig, seed: int) -> tuple[Task, ...]:
    """Build the suite, ordered by the timestep at which each task may be asked."""
    rng = random.Random(seed)
    targets = _targets(config.tasks)
    tasks: list[Task] = []

    stable = [fid for ids in world.stable_by_subject.values() for fid in ids]
    clean = [fid for fid in stable if fid not in world.crowded_ids]
    crowded = [fid for fid in stable if fid in world.crowded_ids]
    anchors = list(stable)
    drifted = list(world.drift_pairs)
    for pool in (clean, crowded, anchors):
        rng.shuffle(pool)
    rng.shuffle(drifted)

    # DIRECT: the query is the fact itself, and the fact has no near-copies.
    # Similarity should win here and is meant to; a benchmark whose easy family
    # is not easy for the baseline is a rigged one.
    for fid in clean[: targets[TaskFamily.DIRECT]]:
        fact = world.facts[fid]
        tasks.append(
            _instance(
                world,
                len(tasks),
                TaskFamily.DIRECT,
                f"Recall the {fact.subject} record matching '{fact.text}'.",
                fact.tokens,
                {fid},
            )
        )

    # COMPOSITIONAL: the query is one fact verbatim; the second required fact
    # shares only the subject word with it, so nothing in the query points at it.
    built = 0
    for fid in anchors:
        if built >= targets[TaskFamily.COMPOSITIONAL]:
            break
        anchor = world.facts[fid]
        partner = _associate(world, anchor)
        if partner is None:
            continue
        built += 1
        tasks.append(
            _instance(
                world,
                len(tasks),
                TaskFamily.COMPOSITIONAL,
                f"While changing '{anchor.text}', report the other {anchor.subject} "
                f"setting that constrains the same work.",
                anchor.tokens,
                {fid, partner.id},
            )
        )

    # DRIFT: the query holds only the tokens the stale and current facts share,
    # so lexical similarity has no handle on which of the two is in force.
    for old_id, new_id in drifted[: targets[TaskFamily.DRIFT]]:
        shared = _shared_tokens(world.facts[old_id], world.facts[new_id])
        tasks.append(
            _instance(
                world,
                len(tasks),
                TaskFamily.DRIFT,
                f"Report the value now in force for '{' '.join(shared)}'.",
                shared,
                {new_id},
                frozenset({old_id}),
            )
        )

    # DISTRACTOR: the query is the fact itself, as in DIRECT, but this fact is
    # shadowed by near-copies differing only in their value. Holding the query
    # form fixed against DIRECT is deliberate: crowding is then the only variable
    # between the two families, and it is the variable an over-consolidating arm
    # loses to.
    for fid in crowded[: targets[TaskFamily.DISTRACTOR]]:
        fact = world.facts[fid]
        tasks.append(
            _instance(
                world,
                len(tasks),
                TaskFamily.DISTRACTOR,
                f"Identify the exact {fact.subject} record, not a near copy: '{fact.text}'.",
                fact.tokens,
                {fid},
            )
        )

    return tuple(sorted(tasks, key=lambda task: (task.available_at_timestep, task.id)))


def _instance(
    world: World,
    index: int,
    family: TaskFamily,
    statement: str,
    query_tokens: tuple[str, ...],
    required: set[str],
    forbidden: frozenset[str] = frozenset(),
) -> Task:
    """Assemble one task, asked only once every fact it touches has arrived."""
    available = max(world.facts[fid].timestep for fid in required) + 1
    assert all(world.facts[fid].timestep < available for fid in required), (
        "a task must not be asked before the evidence that answers it"
    )
    assert all(world.facts[fid].timestep < available for fid in forbidden), (
        "a forbidden fact must already be in memory, or the trap is not set"
    )
    return Task(
        id=f"task-{index:04d}",
        family=family,
        problem_statement=statement,
        query_tokens=query_tokens,
        required_fact_ids=frozenset(required),
        forbidden_fact_ids=forbidden,
        available_at_timestep=available,
    )


def _associate(world: World, anchor: Fact) -> Fact | None:
    """A same-subject fact whose only lexical tie to the anchor is the subject word.

    Reachable by knowing that facts about one subject belong together, and by no
    other route the query offers.
    """
    anchor_tokens = set(anchor.tokens)
    for fid in world.stable_by_subject[anchor.subject]:
        candidate = world.facts[fid]
        if fid != anchor.id and set(candidate.tokens) & anchor_tokens == {anchor.subject}:
            return candidate
    return None


def _shared_tokens(old: Fact, new: Fact) -> tuple[str, ...]:
    """Tokens both facts carry in the same position - subject and predicate.

    Anything left out distinguishes them, so a query built from these alone
    favours neither.
    """
    return tuple(a for a, b in zip(old.tokens, new.tokens, strict=False) if a == b)


def _targets(total: int) -> dict[TaskFamily, int]:
    """Split the requested count as evenly as four families allow."""
    share, extra = divmod(total, len(TaskFamily))
    return {family: share + (1 if i < extra else 0) for i, family in enumerate(TaskFamily)}
