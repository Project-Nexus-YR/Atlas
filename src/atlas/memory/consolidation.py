"""Offline maintenance: link what belongs together, drop what stopped earning.

Consolidation is the half of the Knowledge Evolution Engine that runs *between*
tasks rather than during them. It does two things the Vector-RAG baseline
deliberately does not:

* **Edge formation.** Concepts get an undirected weighted edge when they overlap
  lexically *or* share a subject, which is what later lets spreading activation
  reach a fact the query never mentioned. The subject case is the one that
  matters: a lexical edge joins concepts a cosine search would already have
  scored together, so a graph built from lexical edges alone is a redundant copy
  of its own seed index and the arm is just a worse cosine search.
* **Forgetting.** Concepts whose utility has fallen below
  ``KEEConfig.utility_threshold`` are removed outright. This is the mechanism
  that is supposed to beat a vector index on drift tasks: a superseded fact
  accumulates contradicting evidence, its confidence falls, and it eventually
  stops being retrievable at all.

Neither step consults :class:`~atlas.types.Fact`. Consolidation only ever sees
concepts built from experiences, so it cannot cheat by reading ``supersedes``.
Whether it correctly forgets a stale fact is an empirical question about the
belief-revision rule in :mod:`atlas.memory.kee`, not something arranged here.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.config import KEEConfig
from atlas.memory.scoring import co_occurrence_weight, prune, utility
from atlas.memory.store import GraphStore

__all__ = ["ConsolidationReport", "consolidate", "form_edges"]

_LINK_CANDIDATES = 8
"""How many lexical neighbours each concept is offered as link candidates.

Comparing every pair in the store would be quadratic in the graph size, so
lexical candidates are drawn from the store's own seed index instead. Same-subject
candidates are enumerated separately and are *not* subject to this cap: see
:func:`form_edges` for why a lexical cap alone silently deletes the only edge
type this arm has that the baseline does not.
"""


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    """What one consolidation pass changed, for logging and for the table."""

    edges_added: int
    pruned_ids: tuple[str, ...]
    concepts_before: int
    concepts_after: int


def form_edges(store: GraphStore, config: KEEConfig) -> int:
    """Link related concepts, returning the number of edges written.

    An edge is written only when :func:`co_occurrence_weight` clears
    ``config.edge_threshold``. The store's ``link`` is idempotent under repeated
    passes because it keeps the strongest weight rather than accumulating, so
    running consolidation often cannot inflate the graph's connectivity.

    Candidates come from two places, and the second one is load-bearing.
    Lexical candidates come from the store's seed index, which ranks by Jaccard.
    Same-subject candidates are enumerated directly, because a pair that shares a
    subject but few tokens can never appear in a Jaccard-ranked shortlist - and
    that pair is exactly the one :func:`co_occurrence_weight` grants its subject
    bonus to. Offering only lexical candidates makes that bonus unreachable, and
    the graph degenerates into a redundant copy of the seeder: spreading
    activation can then only reach nodes lexical scoring already ranked highly,
    which is the one thing this arm is supposed to do better than a vector index.

    The subject pass is quadratic within a subject group rather than across the
    store, so its cost is set by how many concepts share one subject.
    """
    written = 0
    by_subject: dict[str, list[str]] = {}
    for concept in store.concepts():
        by_subject.setdefault(concept.subject, []).append(concept.id)

    for concept in sorted(store.concepts(), key=lambda item: item.id):
        lexical = {candidate_id for candidate_id, _ in store.seed(concept.tokens, _LINK_CANDIDATES)}
        for candidate_id in sorted(lexical | set(by_subject.get(concept.subject, ()))):
            if candidate_id == concept.id:
                continue
            other = store.get(candidate_id)
            if other is None:
                continue
            weight = co_occurrence_weight(concept, other)
            if weight >= config.edge_threshold:
                store.link(concept.id, candidate_id, weight)
                written += 1
    return written


def consolidate(store: GraphStore, now: int, config: KEEConfig) -> ConsolidationReport:
    """Run one full maintenance pass: refresh utilities, link, then forget.

    Order matters. Utilities are recomputed first so pruning judges every concept
    at the same timestep; edges are formed before pruning so a concept about to
    be removed cannot leave a dangling edge behind (``remove`` drops its edges,
    but forming them first keeps the survivors' weights correct).
    """
    before = len(store)
    for concept in store.concepts():
        concept.utility = utility(concept, now, config)
    edges_added = form_edges(store, config)
    pruned = prune(store, now, config)
    return ConsolidationReport(
        edges_added=edges_added,
        pruned_ids=tuple(pruned),
        concepts_before=before,
        concepts_after=len(store),
    )
