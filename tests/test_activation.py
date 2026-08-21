"""Spreading activation, checked against graphs small enough to compute by hand.

Every number asserted here is a power of two so the expected value is exact in
binary floating point and can be written down rather than approximated.
"""

from __future__ import annotations

import math

import pytest

from atlas.config import KEEConfig
from atlas.memory.activation import activation_map, spread
from atlas.memory.store import InMemoryGraphStore
from atlas.types import Concept


def _concept(concept_id: str, tokens: tuple[str, ...], facts: tuple[str, ...] = ()) -> Concept:
    return Concept(
        id=concept_id, name=concept_id, subject="core", tokens=tokens, fact_ids=set(facts)
    )


def _config(*, max_hops: int = 3, activation_floor: float = 0.01) -> KEEConfig:
    return KEEConfig(
        decay=0.5,
        max_hops=max_hops,
        activation_floor=activation_floor,
        seed_width=6,
        edge_threshold=0.25,
    )


def _chain() -> InMemoryGraphStore:
    """a -- b -- c, every edge at weight 0.5. Only "alpha" seeds, at Jaccard 1.0."""
    store = InMemoryGraphStore()
    for concept_id, token in (("a", "alpha"), ("b", "beta"), ("c", "gamma")):
        store.add(_concept(concept_id, (token,)))
    store.link("a", "b", 0.5)
    store.link("b", "c", 0.5)
    return store


def _triangle() -> InMemoryGraphStore:
    """a -- b -- c -- a, every edge at weight 0.5."""
    store = _chain()
    store.link("c", "a", 0.5)
    return store


def test_energy_decays_by_configured_factor_per_hop() -> None:
    # Seed a = 1.0 (the query is exactly a's token set).
    # hop 1: 1.0  * 0.5 * 0.5 = 0.25
    # hop 2: 0.25 * 0.5 * 0.5 = 0.0625
    energy = activation_map(_chain(), ("alpha",), _config(max_hops=2))

    assert energy["b"] == 0.25
    assert energy["c"] == 0.0625


def test_two_weak_seeds_jointly_activate_shared_neighbour_above_either_alone() -> None:
    # Each seed matches half the query, so each seeds at Jaccard 1/2. One hop at
    # weight 0.5 and decay 0.5 delivers 0.5 * 0.5 * 0.5 = 0.125 to the target.
    # With both seeds present the target sums both paths and reaches 0.25.
    def store_with(*seed_ids: str) -> InMemoryGraphStore:
        store = InMemoryGraphStore()
        store.add(_concept("target", ("gamma",)))
        for seed_id, token in (("s1", "alpha"), ("s2", "beta")):
            if seed_id in seed_ids:
                store.add(_concept(seed_id, (token,)))
                store.link(seed_id, "target", 0.5)
        return store

    query = ("alpha", "beta")
    config = _config(max_hops=1)
    alone_a = activation_map(store_with("s1"), query, config)["target"]
    alone_b = activation_map(store_with("s2"), query, config)["target"]
    together = activation_map(store_with("s1", "s2"), query, config)["target"]

    assert alone_a == 0.125
    assert alone_b == 0.125
    assert together == 0.25
    assert together > max(alone_a, alone_b)


def test_cycle_does_not_cause_unbounded_activation() -> None:
    # In the triangle a node hands each of its 2 neighbours delta * 0.5 * 0.5, so
    # each round circulates exactly half the previous round's energy. Total
    # activation is therefore bounded by 1.0 / (1 - 0.5) = 2.0 at any hop count.
    # Reaching this assertion at all is the termination half of the claim.
    energy = activation_map(_triangle(), ("alpha",), _config(max_hops=8))

    assert all(math.isfinite(value) for value in energy.values())
    assert sum(energy.values()) < 2.0


def test_reentrant_energy_is_the_delta_not_the_running_total() -> None:
    # Triangle, three hops, tracing a alone. Every transfer is delta * 0.5 * 0.5.
    #   hop 1: a sends 0.25 to each of b and c, and receives nothing.
    #   hop 2: b and c each return 0.0625, so a = 1.0 + 0.125 = 1.125.
    #   hop 3: b and c carry a delta of 0.0625 each and return 0.015625 each,
    #          so a = 1.125 + 0.03125 = 1.15625.
    # Re-spreading a running total would put a's whole 1.125 back on the wire at
    # hop 3 and land far above this. Three hops is the shortest run that tells
    # the two apart, because before then every node's delta is its total.
    energy = activation_map(_triangle(), ("alpha",), _config(max_hops=3))

    assert energy["a"] == 1.15625


def test_edges_below_threshold_do_not_conduct() -> None:
    store = _chain()
    store.add(_concept("faint", ("delta",)))
    store.link("a", "faint", 0.1)  # below the 0.25 edge_threshold

    energy = activation_map(store, ("alpha",), _config())

    assert "faint" not in energy


def test_node_below_the_activation_floor_does_not_spread_further() -> None:
    # b receives 0.25; with the floor raised above that it becomes a dead end, so
    # c never lights up even though max_hops still allows the hop.
    energy = activation_map(_chain(), ("alpha",), _config(max_hops=3, activation_floor=0.3))

    assert energy["b"] == 0.25
    assert "c" not in energy


def test_ranking_is_stable_across_runs_on_identical_input() -> None:
    store = _triangle()
    config = _config()

    first = spread(store, ("alpha",), config, top_k=3)
    second = spread(store, ("alpha",), config, top_k=3)

    assert first.items == second.items


def test_ties_break_on_concept_id() -> None:
    # z_early is reached at hop 1 and b_late only at hop 2, yet both land on
    # 0.25, so the tied pair is discovered in the opposite order to its id order:
    #   hop 1: m_mid = 1.0 * 1.0 * 0.5 = 0.5,  z_early = 1.0 * 0.5 * 0.5 = 0.25
    #   hop 2: b_late = 0.5 * 1.0 * 0.5 = 0.25
    # A sort on score alone is stable and would therefore leave z_early first.
    store = InMemoryGraphStore()
    for concept_id, token in (
        ("a", "alpha"),
        ("m_mid", "mu"),
        ("z_early", "zeta"),
        ("b_late", "beta"),
    ):
        store.add(_concept(concept_id, (token,)))
    store.link("a", "m_mid", 1.0)
    store.link("m_mid", "b_late", 1.0)
    store.link("a", "z_early", 0.5)

    result = spread(store, ("alpha",), _config(max_hops=2), top_k=4)

    assert [item.concept_id for item in result.items] == ["a", "m_mid", "b_late", "z_early"]
    assert result.items[2].score == result.items[3].score


def test_retrieved_items_carry_the_fact_ids_of_their_concepts() -> None:
    store = InMemoryGraphStore()
    store.add(_concept("a", ("alpha",), ("f1",)))
    store.add(_concept("b", ("beta",), ("f2",)))
    store.link("a", "b", 0.5)

    result = spread(store, ("alpha",), _config(), top_k=2)

    assert result.fact_ids == frozenset({"f1", "f2"})


def test_empty_store_returns_nothing() -> None:
    result = spread(InMemoryGraphStore(), ("alpha",), _config(), top_k=5)

    assert result.items == ()
    assert result.fact_ids == frozenset()
    assert result.expanded_nodes == 0


def test_empty_query_returns_nothing() -> None:
    result = spread(_chain(), (), _config(), top_k=5)

    assert result.items == ()
    assert result.expanded_nodes == 0


def test_query_matching_no_concept_returns_nothing() -> None:
    result = spread(_chain(), ("unindexed",), _config(), top_k=5)

    assert result.items == ()
    assert result.expanded_nodes == 0


def test_single_hop_does_not_reach_the_second_ring() -> None:
    energy = activation_map(_chain(), ("alpha",), _config(max_hops=1))

    assert energy["b"] == 0.25
    assert "c" not in energy


def test_isolated_seed_activates_only_itself() -> None:
    store = InMemoryGraphStore()
    store.add(_concept("a", ("alpha",)))

    result = spread(store, ("alpha",), _config(), top_k=5)

    assert [item.concept_id for item in result.items] == ["a"]
    assert result.expanded_nodes == 1


def test_expanded_nodes_counts_every_energised_node_not_only_the_returned_ones() -> None:
    # The cost metric has to report the whole subgraph the query touched, or the
    # comparison table would credit this arm with reading less than it did.
    result = spread(_chain(), ("alpha",), _config(), top_k=1)

    assert len(result.items) == 1
    assert result.expanded_nodes == 3


def test_novel_items_is_zero_when_every_returned_item_was_a_seed() -> None:
    # Two nodes, both lexical matches, linked to each other. Everything returned
    # was already found by the seeder, so propagation contributed no new concept.
    store = InMemoryGraphStore()
    store.add(_concept("s1", ("alpha",)))
    store.add(_concept("s2", ("beta",)))
    store.link("s1", "s2", 0.5)

    result = spread(store, ("alpha", "beta"), _config(), top_k=2)

    assert len(result.items) == 2
    assert result.novel_items == 0


def test_novel_items_counts_a_node_reached_only_by_propagation() -> None:
    # Three seeds at Jaccard 1/3 each all feed one target over weight-1.0 edges,
    # so the target lands on 3 * (1/3 * 1.0 * 0.5) = 0.5 and outranks every seed.
    # It carries none of the query's tokens, so only propagation could reach it.
    store = InMemoryGraphStore()
    store.add(_concept("target", ("omega",)))
    for seed_id, token in (("s1", "alpha"), ("s2", "beta"), ("s3", "gamma")):
        store.add(_concept(seed_id, (token,)))
        store.link(seed_id, "target", 1.0)

    result = spread(store, ("alpha", "beta", "gamma"), _config(max_hops=1), top_k=2)

    assert result.items[0].concept_id == "target"
    assert result.novel_items == 1


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_novel_items_never_exceeds_the_number_of_returned_items(top_k: int) -> None:
    result = spread(_chain(), ("alpha",), _config(), top_k=top_k)

    assert 0 <= result.novel_items <= len(result.items)


def test_top_k_larger_than_the_activated_set_returns_everything_activated() -> None:
    result = spread(_chain(), ("alpha",), _config(), top_k=50)

    assert len(result.items) == 3
    assert result.expanded_nodes == 3
