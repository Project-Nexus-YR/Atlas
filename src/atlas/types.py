"""Shared data contracts.

Every module in Atlas speaks these types and nothing else. Keeping them in one
place is what lets the two memory arms be swapped without the benchmark, the
agent, or the grader knowing which one is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "ArmResult",
    "Concept",
    "Evidence",
    "Experience",
    "Fact",
    "MemorySystem",
    "RetrievalResult",
    "ScoredItem",
    "Task",
    "TaskFamily",
    "TaskOutcome",
]


@dataclass(frozen=True, slots=True)
class Fact:
    """An atomic piece of durable knowledge about the simulated codebase.

    ``supersedes`` is the identity of an earlier fact this one invalidates. It is
    the mechanism behind knowledge drift: an agent that still retrieves the
    superseded fact is acting on stale knowledge and will be graded as failing.
    """

    id: str
    subject: str
    text: str
    tokens: tuple[str, ...]
    supersedes: str | None = None
    timestep: int = 0


@dataclass(frozen=True, slots=True)
class Experience:
    """A raw episodic record: one thing the agent observed at one moment.

    Experiences are what a memory system ingests. A memory system never sees a
    :class:`Fact` directly - only the experience that carried it - which is why
    both arms are given exactly the same evidence.
    """

    id: str
    fact_id: str
    text: str
    tokens: tuple[str, ...]
    subject: str
    timestep: int
    success: bool = True


@dataclass(frozen=True, slots=True)
class Evidence:
    """A trace linking one experience to one concept, signed by ``supports``."""

    experience_id: str
    fact_id: str
    weight: float
    timestep: int
    supports: bool = True


@dataclass(slots=True)
class Concept:
    """A node in the knowledge graph.

    A concept is mutable by design: confidence, frequency and utility all move as
    evidence arrives. ``fact_ids`` is what the grader ultimately reads, so a
    concept that has absorbed several experiences can satisfy several facts at
    once - which is how consolidation earns its keep.
    """

    id: str
    name: str
    subject: str
    tokens: tuple[str, ...]
    abstraction_level: int = 0
    confidence: float = 0.5
    activation: float = 0.0
    utility: float = 0.5
    frequency: int = 1
    last_seen_timestep: int = 0
    created_timestep: int = 0
    fact_ids: set[str] = field(default_factory=set)
    supporting: list[Evidence] = field(default_factory=list)
    contradicting: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ScoredItem:
    """One retrieved concept and the score that got it there."""

    concept_id: str
    score: float
    fact_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What a memory system hands back for one query.

    ``fact_ids`` is the flattened union across ``items`` and is the only field the
    grader consults, so a memory system cannot earn credit for a fact it did not
    surface inside a returned concept.

    ``novel_items`` counts how many returned items the retrieval reached by
    expansion rather than by matching the query directly. It exists so a reader
    can tell whether the graph did any work: a run that wins while this stays at
    zero was produced entirely by the lexical seeder, and is no evidence for
    spreading activation.
    """

    items: tuple[ScoredItem, ...]
    fact_ids: frozenset[str]
    expanded_nodes: int = 0
    novel_items: int = 0

    @classmethod
    def from_items(
        cls, items: tuple[ScoredItem, ...], expanded_nodes: int = 0, novel_items: int = 0
    ) -> RetrievalResult:
        merged: set[str] = set()
        for item in items:
            merged |= item.fact_ids
        return cls(
            items=items,
            fact_ids=frozenset(merged),
            expanded_nodes=expanded_nodes,
            novel_items=novel_items,
        )


class TaskFamily(StrEnum):
    """Why a task is hard, which is what the per-family breakdown reports on."""

    DIRECT = "direct"
    COMPOSITIONAL = "compositional"
    DRIFT = "drift"
    DISTRACTOR = "distractor"


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark instance.

    Resolved iff every required fact is retrieved and no forbidden fact is. The
    two conditions pull against each other: retrieving more raises recall of
    ``required_fact_ids`` and raises the chance of dragging in a stale one.
    """

    id: str
    family: TaskFamily
    problem_statement: str
    query_tokens: tuple[str, ...]
    required_fact_ids: frozenset[str]
    forbidden_fact_ids: frozenset[str]
    available_at_timestep: int


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """The graded result of one task, carrying enough detail to audit the verdict."""

    task_id: str
    family: TaskFamily
    resolved: bool
    retrieved_fact_ids: frozenset[str]
    missing_fact_ids: frozenset[str]
    stale_fact_ids: frozenset[str]
    expanded_nodes: int
    novel_items: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Aggregate over one memory arm for one seed."""

    arm: str
    seed: int
    outcomes: tuple[TaskOutcome, ...]

    @property
    def resolved(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.resolved)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def resolve_rate(self) -> float:
        return self.resolved / self.total if self.outcomes else 0.0

    def by_family(self) -> dict[TaskFamily, tuple[int, int]]:
        """Return ``{family: (resolved, total)}`` for the per-family breakdown."""
        table: dict[TaskFamily, tuple[int, int]] = {}
        for outcome in self.outcomes:
            resolved, total = table.get(outcome.family, (0, 0))
            table[outcome.family] = (resolved + int(outcome.resolved), total + 1)
        return table


@runtime_checkable
class MemorySystem(Protocol):
    """The one interface the benchmark knows about.

    Both arms implement exactly this, so the harness cannot tell them apart and
    cannot give either one a shortcut.
    """

    name: str

    def ingest(self, experience: Experience) -> None:
        """Absorb one raw experience."""
        ...

    def retrieve(self, query_tokens: tuple[str, ...], top_k: int) -> RetrievalResult:
        """Return the most relevant concepts for a query."""
        ...

    def consolidate(self, timestep: int) -> None:
        """Run whatever offline maintenance this arm performs between tasks."""
        ...

    def statistics(self) -> dict[str, float]:
        """Report size and health, for logging and for the comparison table."""
        ...
