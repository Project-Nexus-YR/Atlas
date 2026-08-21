"""The Vector-RAG baseline arm.

One vector per experience, exact cosine over all of them. This is the control
the Knowledge Evolution Engine has to beat, so it is built to be as good as
flat dense retrieval actually gets - full-recall search, no approximation, no
handicap - and it differs from the graph arm only in what it refuses to do:
link, abstract, or forget.

Scoring is SMART ``lnc.ltc``: documents carry log tf with cosine normalisation
and no idf, while the query carries log tf times idf measured from the corpus at
retrieval time. Keeping idf on the query side is what lets the stored matrix
stay frozen for the life of the run - nothing is recomputed as the corpus grows,
so no reviewer can read the weighting as a form of consolidation.

Stated limitation: this arm is lexical. It matches tokens, not meaning, so a
query that paraphrases an experience is unreachable for it at any dimensionality
and no amount of tuning would change that. The experiment therefore constrains
its world generator to draw query tokens from the experience vocabulary, which
keeps a measured gap attributable to consolidation rather than to the encoder.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from atlas.config import ExperimentConfig
from atlas.embedding import Embedder
from atlas.memory.store import InMemoryGraphStore
from atlas.types import Concept, Experience, RetrievalResult, ScoredItem

__all__ = ["VectorRAGMemory"]


class VectorRAGMemory:
    """Dense retrieval over every experience ever ingested.

    The concepts live in an :class:`~atlas.memory.store.InMemoryGraphStore` so
    both arms report their size through the same lens, but no edge is ever
    drawn: the store is used as a flat container on purpose.
    """

    def __init__(self, embedder: Embedder, config: ExperimentConfig) -> None:
        """Build an empty index.

        ``config`` is accepted so the harness constructs both arms the same way;
        a flat index has no hyper-parameters of its own to read from it.
        """
        self.name = "vector-rag"
        self._embedder = embedder
        self._store = InMemoryGraphStore()
        # Rows of ``_matrix`` line up with ``_rows``; both are truncated to
        # ``_size`` because the matrix is over-allocated and doubled on growth.
        self._matrix: NDArray[np.float64] = np.zeros((0, embedder.dim), dtype=np.float64)
        self._rows: list[Concept] = []
        # Hashing collides, so document frequency cannot be read back off the
        # matrix without conflating tokens that share a bucket. Counted here.
        self._document_frequency: dict[str, int] = {}

    def ingest(self, experience: Experience) -> None:
        """Store the experience verbatim as its own concept and vector."""
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
        for token in set(experience.tokens):  # distinct: df counts documents, not occurrences
            self._document_frequency[token] = self._document_frequency.get(token, 0) + 1
        self._append(self._embedder.encode(experience.tokens), concept)

    def retrieve(self, query_tokens: tuple[str, ...], top_k: int) -> RetrievalResult:
        """Return the ``top_k`` experiences closest to the idf-weighted query, best first."""
        size = len(self._rows)
        if size == 0:
            return RetrievalResult.from_items(())
        query = self._embedder.encode_query(query_tokens, self._idf(query_tokens))
        # Rows and query are unit vectors, so the matrix product is cosine.
        scores: NDArray[np.float64] = self._matrix[:size] @ query
        k = min(top_k, size)
        candidates = np.argpartition(-scores, k - 1)[:k]
        ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
        items = tuple(
            ScoredItem(
                concept_id=self._rows[index].id,
                score=float(scores[index]),
                fact_ids=frozenset(self._rows[index].fact_ids),
            )
            for index in ranked
        )
        return RetrievalResult.from_items(items, expanded_nodes=size)

    def consolidate(self, timestep: int) -> None:
        """Do nothing, deliberately.

        Not consolidating is the whole content of this arm: it is what makes the
        comparison measure consolidation rather than measure two different
        retrievers. A vector store that reorganised itself between tasks would
        no longer be the baseline the paper claims to compare against.
        """

    def statistics(self) -> dict[str, float]:
        """Report size only - a flat index has no health to report."""
        return {
            "concepts": float(len(self._store)),
            "vectors": float(len(self._rows)),
            "dim": float(self._embedder.dim),
        }

    def _idf(self, tokens: tuple[str, ...]) -> dict[str, float]:
        """Smoothed inverse document frequency for each distinct query token.

        The two ``+ 1`` terms are what make the smoothing worth having: a token
        present in every document keeps a small positive weight instead of
        collapsing to exactly zero, and a token never seen before is finite
        rather than a division by zero.
        """
        total = len(self._rows)
        return {
            token: math.log((total + 1) / (self._document_frequency.get(token, 0) + 1)) + 1.0
            for token in set(tokens)
        }

    def _append(self, vector: NDArray[np.float64], concept: Concept) -> None:
        """Add one row, doubling the backing matrix when it is full."""
        size = len(self._rows)
        if size == self._matrix.shape[0]:
            grown: NDArray[np.float64] = np.zeros(
                (max(1, size * 2), self._embedder.dim), dtype=np.float64
            )
            grown[:size] = self._matrix
            self._matrix = grown
        self._matrix[size] = vector
        self._rows.append(concept)
