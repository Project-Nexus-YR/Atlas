"""The Knowledge Evolution Engine: the arm under test.

The KEE differs from the vector baseline in one structural way and one
behavioural way.

Structurally, it stores concepts in a graph and retrieves by spreading
activation, so a fact the query never mentions can still be reached through an
edge. Behaviourally, it revises: an incoming experience that closely resembles a
known concept without matching it is treated as *news about that concept* rather
than as an unrelated document, and the old concept is marked as contradicted.

That second rule is the entire drift story, and it is worth being precise about
why it is not cheating. A memory arm never sees a :class:`~atlas.types.Fact`, so
it cannot read ``supersedes`` and cannot know that one fact invalidates another.
It sees only two experiences that say almost, but not quite, the same thing about
the same subject, and must decide for itself which reading is right. The rule
here is a heuristic and it is genuinely fallible: a distractor built from the
same subject vocabulary lands in the same overlap band as a real revision, so the
engine will sometimes discredit a perfectly good concept on the strength of
noise. Those are real losses and they show up in the results table.
"""

from __future__ import annotations

from atlas.config import ExperimentConfig
from atlas.logging import get_logger
from atlas.memory.activation import spread
from atlas.memory.consolidation import consolidate
from atlas.memory.scoring import apply_evidence
from atlas.memory.store import GraphStore, InMemoryGraphStore
from atlas.types import Concept, Evidence, Experience, RetrievalResult

__all__ = ["KnowledgeEvolutionEngine"]

_REVISION_CANDIDATES = 6
"""Lexical neighbours considered when deciding whether an experience is news."""

_EVIDENCE_WEIGHT = 1.0
"""Every experience counts the same. Weighting them would need a notion of source
reliability the world does not model, and inventing one would be a free parameter
tuned against the very number this experiment reports."""


class KnowledgeEvolutionEngine:
    """Graph-structured memory with belief revision and utility-based forgetting."""

    name = "kee"

    def __init__(self, config: ExperimentConfig, store: GraphStore | None = None) -> None:
        self._config = config
        self._kee = config.kee
        self._store: GraphStore = store if store is not None else InMemoryGraphStore()
        self._log = get_logger(__name__)
        self._now = 0
        self._revisions = 0
        self._merges = 0

    def ingest(self, experience: Experience) -> None:
        """Absorb an experience as a restatement, a revision, or a new concept."""
        self._now = max(self._now, experience.timestep)
        match, overlap = self._closest(experience)

        if match is not None and overlap >= self._kee.merge_threshold:
            # Same claim, said again. Reinforces what is already there.
            apply_evidence(match, self._supporting(experience), self._now, self._kee)
            self._merges += 1
            return

        if match is not None and overlap >= self._kee.revision_threshold:
            # Close but not the same: read as news that contradicts the old claim.
            apply_evidence(match, self._contradicting(experience), self._now, self._kee)
            self._revisions += 1

        self._store.add(self._new_concept(experience))

    def retrieve(self, query_tokens: tuple[str, ...], top_k: int) -> RetrievalResult:
        """Spread activation from the query's lexical seeds and return the top nodes."""
        return spread(self._store, query_tokens, self._kee, top_k)

    def consolidate(self, timestep: int) -> None:
        """Link related concepts and forget the ones that stopped earning their place."""
        self._now = max(self._now, timestep)
        report = consolidate(self._store, self._now, self._kee)
        self._log.debug(
            "consolidated: +%d edges, -%d concepts, %d remain",
            report.edges_added,
            len(report.pruned_ids),
            report.concepts_after,
        )

    def statistics(self) -> dict[str, float]:
        """Report graph size and how often each ingest branch fired.

        ``revisions`` is the diagnostic that matters: it is how many times the
        engine decided an experience contradicted something it already believed.
        If it is zero, the drift mechanism never fired and any win on drift tasks
        came from somewhere else.
        """
        return {
            "concepts": float(len(self._store)),
            "revisions": float(self._revisions),
            "merges": float(self._merges),
        }

    def _closest(self, experience: Experience) -> tuple[Concept | None, float]:
        """Return the most lexically similar concept on the same subject, and its overlap.

        Restricting to the same subject is what keeps the revision rule from
        firing across unrelated areas of the codebase, where high token overlap
        is coincidence rather than contradiction.
        """
        best: Concept | None = None
        best_overlap = 0.0
        for concept_id, overlap in self._store.seed(experience.tokens, _REVISION_CANDIDATES):
            concept = self._store.get(concept_id)
            if concept is None or concept.subject != experience.subject:
                continue
            if overlap > best_overlap:
                best, best_overlap = concept, overlap
        return best, best_overlap

    def _new_concept(self, experience: Experience) -> Concept:
        return Concept(
            id=f"c-{experience.id}",
            name=experience.text,
            subject=experience.subject,
            tokens=experience.tokens,
            last_seen_timestep=experience.timestep,
            created_timestep=experience.timestep,
            fact_ids={experience.fact_id},
        )

    def _supporting(self, experience: Experience) -> Evidence:
        return Evidence(
            experience_id=experience.id,
            fact_id=experience.fact_id,
            weight=_EVIDENCE_WEIGHT,
            timestep=experience.timestep,
            supports=True,
        )

    def _contradicting(self, experience: Experience) -> Evidence:
        return Evidence(
            experience_id=experience.id,
            fact_id=experience.fact_id,
            weight=_EVIDENCE_WEIGHT,
            timestep=experience.timestep,
            supports=False,
        )
