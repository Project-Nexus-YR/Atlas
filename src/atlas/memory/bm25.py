"""The BM25 baseline arm: the lexical control.

This arm exists to close a hole in the comparison rather than to win it. The
vector baseline in :mod:`atlas.memory.vector_rag` is dense only in
representation - it hashes tokens, so it matches tokens and not meaning - and it
already solves two of the four families outright. That leaves one reading of the
headline untested: that those families are lexically saturated, and the graph
arm's deficit on them measures the corpus rather than the method.

BM25 is the standard way to ask. It is the oldest and simplest ranking function
still taken seriously, it has no representation of meaning whatsoever, and it
has no hyper-parameter worth tuning. If it also scores near the ceiling on a
family, that family is degenerate and no conclusion about retrieval may be drawn
from it. If it does not, the vector arm's score on that family is evidence about
the vector arm.

Okapi BM25, with the two usual corrections to raw term frequency: saturation, so
the fifth occurrence of a term is worth less than the second, and length
normalisation, so a long document cannot outrank a short one by accumulating
matches. Both are governed by :class:`~atlas.config.BM25Config`.

Two documented asymmetries with the vector arm, neither of them a handicap:

- Statistics are read live from the whole corpus, where the vector arm freezes
  its document matrix and carries idf on the query side. BM25 has no stored
  representation to freeze, so there is nothing here that could be read as
  consolidation - the index is a count of what has been seen and nothing else.
- Only documents sharing at least one query token are scored and returned, which
  is what every BM25 implementation does. The vector arm always returns ``top_k``
  because hash collisions give unrelated documents a small non-zero cosine. So
  this arm can return fewer than ``top_k`` items, which under the grader is a
  genuine trade rather than an advantage: it forgoes required facts it might have
  stumbled into, in exchange for not volunteering forbidden ones.
"""

from __future__ import annotations

import math

from atlas.config import ExperimentConfig
from atlas.memory.store import InMemoryGraphStore
from atlas.types import Concept, Experience, RetrievalResult, ScoredItem

__all__ = ["BM25Memory"]


class BM25Memory:
    """Okapi BM25 over every experience ever ingested.

    The concepts live in an :class:`~atlas.memory.store.InMemoryGraphStore` so
    every arm reports its size through the same lens, but no edge is ever drawn:
    the store is a flat container here, exactly as in the vector arm.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Build an empty index."""
        self.name = "bm25"
        self._bm25 = config.bm25
        self._store = InMemoryGraphStore()
        # token -> [(row index, term frequency in that row)]. Postings are
        # appended in ingest order, so every list is already sorted by index.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._rows: list[Concept] = []
        self._lengths: list[int] = []
        self._total_length = 0

    def ingest(self, experience: Experience) -> None:
        """Store the experience verbatim as its own concept and posting list."""
        concept = Concept(
            id=experience.id,
            name=experience.text,
            subject=experience.subject,
            tokens=experience.tokens,
            last_seen_timestep=experience.timestep,
            created_timestep=experience.timestep,
            fact_ids={experience.fact_id},
        )
        self._store.add(concept)

        index = len(self._rows)
        frequencies: dict[str, int] = {}
        for token in experience.tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        for token, frequency in frequencies.items():
            self._postings.setdefault(token, []).append((index, frequency))

        self._rows.append(concept)
        self._lengths.append(len(experience.tokens))
        self._total_length += len(experience.tokens)

    def retrieve(self, query_tokens: tuple[str, ...], top_k: int) -> RetrievalResult:
        """Return up to ``top_k`` documents by BM25 score, best first."""
        total = len(self._rows)
        if total == 0:
            return RetrievalResult.from_items(())

        average_length = self._total_length / total
        scores: dict[int, float] = {}
        # Sorted so float addition happens in the same order on every run, which
        # is what makes two runs over the same corpus bit-identical. Set
        # iteration order is not stable across processes; this is.
        for token in sorted(set(query_tokens)):
            postings = self._postings.get(token)
            if postings is None:
                continue
            idf = self._idf(len(postings), total)
            for index, frequency in postings:
                saturation = self._bm25.k1 * (
                    1.0 - self._bm25.b + self._bm25.b * self._lengths[index] / average_length
                )
                scores[index] = scores.get(index, 0.0) + idf * frequency * (self._bm25.k1 + 1.0) / (
                    frequency + saturation
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], self._rows[pair[0]].id))
        items = tuple(
            ScoredItem(
                concept_id=self._rows[index].id,
                score=score,
                fact_ids=frozenset(self._rows[index].fact_ids),
            )
            for index, score in ranked[:top_k]
        )
        # Only documents sharing a query token were touched, so that - not the
        # corpus size - is what this arm actually read.
        return RetrievalResult.from_items(items, expanded_nodes=len(scores))

    def consolidate(self, timestep: int) -> None:
        """Do nothing, deliberately.

        A control that reorganised itself between tasks would stop being the
        control: the measured gap could then be attributed to its maintenance
        rather than to the graph arm's.
        """

    def statistics(self) -> dict[str, float]:
        """Report size only - an inverted index has no health to report."""
        total = len(self._rows)
        return {
            "concepts": float(len(self._store)),
            "documents": float(total),
            "vocabulary": float(len(self._postings)),
        }

    @staticmethod
    def _idf(document_frequency: int, total: int) -> float:
        """Robertson-Sparck Jones idf with the usual ``1 +`` floor.

        Without the floor this goes negative for a token present in more than
        half the corpus, which on a small vocabulary lets a common term subtract
        score from a document that genuinely contains it.
        """
        return math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
