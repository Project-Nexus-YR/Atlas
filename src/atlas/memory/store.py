"""Concept storage and the graph the energy spreads through.

The store owns structure only - which concepts exist and how strongly they are
linked. It deliberately knows nothing about activation, utility or consolidation
so those three can be tested against a hand-built graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from atlas.types import Concept

__all__ = ["GraphStore", "InMemoryGraphStore"]


class GraphStore(Protocol):
    """Structural interface over a concept graph."""

    def add(self, concept: Concept) -> None:
        """Insert or replace a concept, reindexing its tokens."""
        ...

    def get(self, concept_id: str) -> Concept | None:
        """Return a concept by id, or ``None``."""
        ...

    def remove(self, concept_id: str) -> bool:
        """Delete a concept and every edge touching it. True if it existed."""
        ...

    def concepts(self) -> Iterator[Concept]:
        """Iterate every concept currently stored."""
        ...

    def link(self, a_id: str, b_id: str, weight: float) -> None:
        """Create or strengthen an undirected weighted edge."""
        ...

    def neighbours(self, concept_id: str) -> tuple[tuple[str, float], ...]:
        """Return ``(neighbour_id, weight)`` pairs, strongest first."""
        ...

    def seed(self, tokens: tuple[str, ...], width: int) -> tuple[tuple[str, float], ...]:
        """Return the ``width`` concepts most lexically overlapping ``tokens``.

        This is the only lexical step in the graph arm; everything after it is
        structural. Scored by Jaccard overlap so a concept cannot win by being
        merely long.
        """
        ...

    def __len__(self) -> int:
        """Number of concepts stored."""
        ...


class InMemoryGraphStore:
    """Dict-backed store. The default, and the one the benchmark runs on.

    Edges are held as a symmetric adjacency map rather than an edge list because
    spreading activation reads neighbours far more often than it writes them.
    """

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}
        self._edges: dict[str, dict[str, float]] = {}
        self._token_index: dict[str, set[str]] = {}

    def add(self, concept: Concept) -> None:
        existing = self._concepts.get(concept.id)
        if existing is not None:
            self._deindex(existing)
        self._concepts[concept.id] = concept
        self._edges.setdefault(concept.id, {})
        for token in concept.tokens:
            self._token_index.setdefault(token, set()).add(concept.id)

    def get(self, concept_id: str) -> Concept | None:
        return self._concepts.get(concept_id)

    def remove(self, concept_id: str) -> bool:
        concept = self._concepts.pop(concept_id, None)
        if concept is None:
            return False
        self._deindex(concept)
        for neighbour in self._edges.pop(concept_id, {}):
            self._edges.get(neighbour, {}).pop(concept_id, None)
        return True

    def concepts(self) -> Iterator[Concept]:
        return iter(list(self._concepts.values()))

    def link(self, a_id: str, b_id: str, weight: float) -> None:
        if a_id == b_id or a_id not in self._concepts or b_id not in self._concepts:
            return
        for source, target in ((a_id, b_id), (b_id, a_id)):
            current = self._edges.setdefault(source, {}).get(target, 0.0)
            self._edges[source][target] = max(current, weight)

    def neighbours(self, concept_id: str) -> tuple[tuple[str, float], ...]:
        edges = self._edges.get(concept_id, {})
        return tuple(sorted(edges.items(), key=lambda pair: (-pair[1], pair[0])))

    def seed(self, tokens: tuple[str, ...], width: int) -> tuple[tuple[str, float], ...]:
        query = set(tokens)
        if not query:
            return ()
        candidates: set[str] = set()
        for token in query:
            candidates |= self._token_index.get(token, set())
        scored: list[tuple[str, float]] = []
        for concept_id in candidates:
            concept = self._concepts[concept_id]
            concept_tokens = set(concept.tokens)
            union = query | concept_tokens
            if not union:
                continue
            scored.append((concept_id, len(query & concept_tokens) / len(union)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return tuple(scored[:width])

    def __len__(self) -> int:
        return len(self._concepts)

    def _deindex(self, concept: Concept) -> None:
        for token in concept.tokens:
            bucket = self._token_index.get(token)
            if bucket is None:
                continue
            bucket.discard(concept.id)
            if not bucket:
                del self._token_index[token]
