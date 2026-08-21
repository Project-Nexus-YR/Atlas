"""Spreading activation over the concept graph.

This is the retrieval mechanism of the graph arm: the classical spreading
activation search used over semantic networks. A query energises a small set of
lexically matching nodes and that energy flows outward along weighted edges,
attenuating at every hop by the edge weight and a fixed decay factor.

Its one interesting property is convergence. A node's activation is the SUM of
the energy reaching it along every path, so two seeds that are each only weakly
related to the query can jointly push a shared neighbour above nodes that either
seed lights on its own. That is the whole reason spreading activation beats
top-k cosine similarity on compositional queries: a vector index scores every
candidate independently against the query, so two weak matches can never
combine, whereas the node sitting between two weak matches is exactly where the
answer to a compositional question tends to live.

Energy is propagated as a delta: each round a node re-emits only what it
received in the previous round, never its running total. On a cyclic graph that
is what keeps re-entrant energy shrinking geometrically instead of compounding,
while still summing correctly along genuinely distinct paths.
"""

from __future__ import annotations

from atlas.config import KEEConfig
from atlas.memory.store import GraphStore
from atlas.types import RetrievalResult, ScoredItem

__all__ = ["activation_map", "spread"]


def _propagate(store: GraphStore, seeded: dict[str, float], config: KEEConfig) -> dict[str, float]:
    """Flow the seeded energy outward and return the total each node accumulated.

    Split out from :func:`activation_map` only so :func:`spread` can hold on to
    the seed set it started from without asking the store to seed twice.
    """
    frontier: dict[str, float] = dict(seeded)
    energy: dict[str, float] = dict(seeded)

    for _ in range(config.max_hops):
        arriving: dict[str, float] = {}
        for source, delta in sorted(frontier.items()):
            # A node whose last-round intake was negligible is a dead end: it
            # would forward rounding noise into the rest of the graph.
            if delta < config.activation_floor:
                continue
            for target, weight in store.neighbours(source):
                if weight < config.edge_threshold:
                    continue
                arriving[target] = arriving.get(target, 0.0) + delta * weight * config.decay
        if not arriving:
            break
        # Sorted so float addition happens in the same order on every run, which
        # is what makes two runs over the same graph bit-identical.
        for target in sorted(arriving):
            energy[target] = energy.get(target, 0.0) + arriving[target]
        frontier = arriving

    return energy


def activation_map(
    store: GraphStore, query_tokens: tuple[str, ...], config: KEEConfig
) -> dict[str, float]:
    """Return the total energy each node accumulated, keyed by concept id.

    Exposed alongside :func:`spread` so the mechanism can be asserted on
    directly rather than only through the ranked slice it produces.
    """
    return _propagate(store, dict(store.seed(query_tokens, config.seed_width)), config)


def spread(
    store: GraphStore, query_tokens: tuple[str, ...], config: KEEConfig, top_k: int
) -> RetrievalResult:
    """Retrieve the ``top_k`` most activated concepts for ``query_tokens``.

    ``expanded_nodes`` counts every distinct node that received any energy, not
    just the ones returned. That is the honest cost of this arm: it is what the
    comparison table reads to show the graph is not simply reading more of the
    corpus than the vector baseline.

    ``novel_items`` counts returned items whose id was not in the seed set for
    this query - strictly that, not "reached at hop >= 1" and not "fed by more
    than one path". It audits one specific charge: that a win here could have
    been produced by the lexical seeder alone, with the propagation contributing
    nothing but a re-ordering of concepts the seeder had already found.

    Ties break on concept id, so the same graph and query always rank the same.
    """
    seeded = dict(store.seed(query_tokens, config.seed_width))
    energy = _propagate(store, seeded, config)
    items: list[ScoredItem] = []
    for concept_id, score in sorted(energy.items(), key=lambda pair: (-pair[1], pair[0])):
        if len(items) == top_k:
            break
        concept = store.get(concept_id)
        if concept is None:
            continue
        items.append(
            ScoredItem(concept_id=concept_id, score=score, fact_ids=frozenset(concept.fact_ids))
        )
    novel = sum(1 for item in items if item.concept_id not in seeded)
    return RetrievalResult.from_items(tuple(items), expanded_nodes=len(energy), novel_items=novel)
