"""Tests for the offline maintenance pass."""

from __future__ import annotations

from atlas.config import KEEConfig
from atlas.memory.consolidation import consolidate, form_edges
from atlas.memory.store import InMemoryGraphStore
from atlas.types import Concept, Evidence


def make_concept(
    concept_id: str,
    tokens: tuple[str, ...],
    *,
    subject: str = "parser",
    confidence: float = 0.9,
    frequency: int = 8,
    last_seen: int = 100,
) -> Concept:
    return Concept(
        id=concept_id,
        name=concept_id,
        subject=subject,
        tokens=tokens,
        confidence=confidence,
        frequency=frequency,
        last_seen_timestep=last_seen,
        fact_ids={f"fact-{concept_id}"},
    )


def test_overlapping_concepts_are_linked() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("retry", "backoff", "policy")))
    store.add(make_concept("b", ("retry", "backoff", "limit")))
    form_edges(store, KEEConfig())

    assert [neighbour for neighbour, _ in store.neighbours("a")] == ["b"]


def test_concepts_sharing_no_tokens_are_not_linked() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("retry", "backoff"), subject="retry"))
    store.add(make_concept("b", ("migration", "rollback"), subject="migration"))
    form_edges(store, KEEConfig())

    assert store.neighbours("a") == ()
    assert store.neighbours("b") == ()


def test_same_subject_concepts_are_linked_even_with_no_shared_tokens() -> None:
    """The one edge type a vector index cannot make, and the one this arm exists for.

    ``co_occurrence_weight`` grants a subject bonus precisely so this pair links.
    That bonus is unreachable if candidates are drawn only from the lexical seed
    index, because a concept sharing no token is never in the index's output at
    any shortlist width - so this pair is the sharpest possible probe of where
    link candidates come from.
    """
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("throttle", "log", "sink"), subject="throttle"))
    store.add(make_concept("b", ("batch", "size", "partition"), subject="throttle"))
    form_edges(store, KEEConfig())

    assert [neighbour for neighbour, _ in store.neighbours("a")] == ["b"]


def test_a_subject_edge_survives_a_crowded_lexical_shortlist() -> None:
    """The benchmark's actual case: one shared token, outranked by lexical rivals.

    Both concepts are about ``throttle`` and share only that token, for a Jaccard
    of 1/9 - below ``edge_threshold``. Ten better-matching concepts on other
    subjects fill the lexical shortlist ahead of the partner, so an implementation
    that caps candidates lexically drops the edge no matter how the weight
    function would have scored it.
    """
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("throttle", "log", "sink", "ring", "buffer"), subject="throttle"))
    store.add(
        make_concept("b", ("throttle", "batch", "size", "partition", "wide"), subject="throttle")
    )
    for index in range(10):
        store.add(
            make_concept(
                f"rival{index}",
                ("throttle", "log", "sink", f"filler{index}"),
                subject=f"other{index}",
            )
        )
    form_edges(store, KEEConfig())

    assert "b" in [neighbour for neighbour, _ in store.neighbours("a")]


def test_repeated_passes_do_not_inflate_edge_weight() -> None:
    """Consolidation runs many times per experiment; connectivity must not drift up."""
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("retry", "backoff", "policy")))
    store.add(make_concept("b", ("retry", "backoff", "limit")))
    config = KEEConfig()

    form_edges(store, config)
    first = store.neighbours("a")
    for _ in range(5):
        form_edges(store, config)

    assert store.neighbours("a") == first


def test_edges_below_threshold_are_not_written() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("retry", "backoff", "policy"), subject="retry"))
    store.add(make_concept("b", ("retry", "unrelated", "words", "here"), subject="migration"))
    form_edges(store, KEEConfig(edge_threshold=0.9))

    assert store.neighbours("a") == ()


def test_stale_contradicted_concept_is_pruned_while_fresh_one_survives() -> None:
    """The drift mechanism end to end: contradiction plus age must lose to freshness."""
    store = InMemoryGraphStore()
    stale = make_concept("stale", ("retry", "backoff", "policy"), confidence=0.1, last_seen=0)
    stale.contradicting = [
        Evidence(experience_id=f"e{i}", fact_id="f", weight=1.0, timestep=i, supports=False)
        for i in range(6)
    ]
    fresh = make_concept("fresh", ("retry", "backoff", "ceiling"), confidence=0.95, last_seen=200)
    store.add(stale)
    store.add(fresh)

    report = consolidate(store, now=200, config=KEEConfig())

    assert "stale" in report.pruned_ids
    assert store.get("fresh") is not None


def test_consolidate_never_empties_the_store() -> None:
    """An empty graph makes the next retrieval meaningless - a harness artefact, not a result."""
    store = InMemoryGraphStore()
    for index in range(3):
        store.add(make_concept(f"c{index}", ("token", f"t{index}"), confidence=0.0, frequency=1))

    report = consolidate(store, now=10_000, config=KEEConfig())

    assert len(store) == 1
    assert report.concepts_after == 1


def test_report_counts_match_the_store() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("a", ("retry", "backoff")))
    store.add(make_concept("b", ("retry", "limit")))

    report = consolidate(store, now=100, config=KEEConfig())

    assert report.concepts_before == 2
    assert report.concepts_after == len(store)
    assert report.concepts_after == report.concepts_before - len(report.pruned_ids)


def test_consolidate_on_empty_store_reports_nothing_and_does_not_raise() -> None:
    report = consolidate(InMemoryGraphStore(), now=5, config=KEEConfig())

    assert report == report.__class__(
        edges_added=0, pruned_ids=(), concepts_before=0, concepts_after=0
    )


def test_utilities_are_refreshed_before_pruning() -> None:
    """A concept whose stored utility is stale must be judged on its recomputed one."""
    store = InMemoryGraphStore()
    concept = make_concept("a", ("retry", "backoff"), confidence=0.95, last_seen=500)
    concept.utility = 0.0  # stale value that would prune it if trusted
    store.add(concept)

    consolidate(store, now=500, config=KEEConfig())

    assert store.get("a") is not None
    assert concept.utility > KEEConfig().utility_threshold
