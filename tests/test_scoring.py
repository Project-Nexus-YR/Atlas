"""Behavioural tests for the concept lifecycle.

The two regressions these guard hardest are the ones the previous scoring had:
recency collapsing to a constant floor for anything old, and frequency clipping
flat so every well-seen concept tied. Both are tested as ordering properties
across a sweep, because a boundary assertion alone would pass on the old code.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from atlas.config import KEEConfig
from atlas.memory.scoring import (
    apply_evidence,
    co_occurrence_weight,
    contradiction_ratio,
    frequency_factor,
    prune,
    recency_factor,
    utility,
)
from atlas.memory.store import InMemoryGraphStore
from atlas.types import Concept, Evidence

CONFIG = KEEConfig()


def make_concept(
    concept_id: str,
    *,
    subject: str = "auth",
    tokens: tuple[str, ...] = ("alpha", "beta"),
    confidence: float = 0.5,
    frequency: int = 1,
    last_seen: int = 0,
) -> Concept:
    return Concept(
        id=concept_id,
        name=concept_id,
        subject=subject,
        tokens=tokens,
        confidence=confidence,
        frequency=frequency,
        last_seen_timestep=last_seen,
    )


def make_evidence(*, supports: bool, weight: float = 1.0, fact_id: str = "f1") -> Evidence:
    return Evidence(
        experience_id="e1", fact_id=fact_id, weight=weight, timestep=0, supports=supports
    )


# --- recency ---------------------------------------------------------------


def test_recency_factor_is_one_when_now_equals_last_seen() -> None:
    assert recency_factor(7, 7, CONFIG.recency_halflife) == 1.0


def test_recency_factor_clamps_to_one_when_now_precedes_last_seen() -> None:
    assert recency_factor(50, 10, CONFIG.recency_halflife) == 1.0


def test_recency_factor_halves_after_exactly_one_halflife() -> None:
    assert recency_factor(0, CONFIG.recency_halflife, CONFIG.recency_halflife) == pytest.approx(0.5)


def test_recency_factor_stays_above_zero_far_beyond_the_halflife() -> None:
    # The linear predecessor returned exactly 0.0 here, flattening all old concepts.
    assert recency_factor(0, CONFIG.recency_halflife * 50, CONFIG.recency_halflife) > 0.0


def test_recency_factor_strictly_decreases_across_an_age_sweep() -> None:
    scores = [recency_factor(0, age, CONFIG.recency_halflife) for age in range(0, 400, 7)]

    assert all(later < earlier for earlier, later in pairwise(scores))


# --- frequency -------------------------------------------------------------


def test_frequency_factor_is_zero_for_a_never_seen_concept() -> None:
    assert frequency_factor(0) == 0.0


def test_frequency_factor_never_reaches_one_at_extreme_frequency() -> None:
    assert frequency_factor(1_000_000) < 1.0


def test_frequency_factor_still_separates_concepts_far_past_one_hundred() -> None:
    # The clipping predecessor tied every concept with frequency >= 100 at 1.0.
    assert frequency_factor(1000) > frequency_factor(100)


def test_frequency_factor_strictly_increases_across_a_frequency_sweep() -> None:
    scores = [frequency_factor(freq) for freq in range(0, 500, 3)]

    assert all(higher > lower for lower, higher in pairwise(scores))


# --- contradiction ---------------------------------------------------------


def test_contradiction_ratio_is_zero_for_a_concept_with_no_evidence() -> None:
    assert contradiction_ratio(make_concept("c1")) == 0.0


def test_contradiction_ratio_is_one_when_every_piece_of_evidence_contradicts() -> None:
    concept = make_concept("c1")
    for _ in range(3):
        apply_evidence(concept, make_evidence(supports=False), now=1, config=CONFIG)

    assert contradiction_ratio(concept) == 1.0


def test_contradiction_ratio_is_half_when_evidence_is_evenly_split() -> None:
    concept = make_concept("c1")
    apply_evidence(concept, make_evidence(supports=True), now=1, config=CONFIG)
    apply_evidence(concept, make_evidence(supports=False), now=1, config=CONFIG)

    assert contradiction_ratio(concept) == pytest.approx(0.5)


# --- evidence and confidence ----------------------------------------------


def test_contradicting_evidence_lowers_confidence() -> None:
    concept = make_concept("c1")
    for _ in range(3):
        apply_evidence(concept, make_evidence(supports=True), now=1, config=CONFIG)
    before = concept.confidence

    apply_evidence(concept, make_evidence(supports=False, weight=2.0), now=2, config=CONFIG)

    assert concept.confidence < before


def test_supporting_evidence_raises_confidence() -> None:
    concept = make_concept("c1")
    before = concept.confidence

    apply_evidence(concept, make_evidence(supports=True), now=1, config=CONFIG)

    assert concept.confidence > before


def test_confidence_stays_within_unit_range_under_overwhelming_contradiction() -> None:
    concept = make_concept("c1")
    for _ in range(50):
        apply_evidence(concept, make_evidence(supports=False, weight=9.0), now=1, config=CONFIG)

    assert 0.0 < concept.confidence < 0.5


def test_apply_evidence_records_the_fact_id_of_supporting_evidence() -> None:
    concept = make_concept("c1")

    apply_evidence(concept, make_evidence(supports=True, fact_id="fact-9"), now=1, config=CONFIG)

    assert concept.fact_ids == {"fact-9"}


def test_apply_evidence_omits_the_fact_id_of_contradicting_evidence() -> None:
    concept = make_concept("c1")

    apply_evidence(concept, make_evidence(supports=False, fact_id="fact-9"), now=1, config=CONFIG)

    assert concept.fact_ids == set()


def test_apply_evidence_bumps_frequency_by_one() -> None:
    concept = make_concept("c1", frequency=4)

    apply_evidence(concept, make_evidence(supports=True), now=1, config=CONFIG)

    assert concept.frequency == 5


def test_apply_evidence_moves_last_seen_to_now() -> None:
    concept = make_concept("c1", last_seen=3)

    apply_evidence(concept, make_evidence(supports=True), now=99, config=CONFIG)

    assert concept.last_seen_timestep == 99


def test_apply_evidence_refreshes_the_stored_utility() -> None:
    concept = make_concept("c1", last_seen=0)
    concept.utility = -1.0

    apply_evidence(concept, make_evidence(supports=True), now=500, config=CONFIG)

    assert concept.utility == utility(concept, 500, CONFIG)


# --- utility ---------------------------------------------------------------


def test_utility_of_a_concept_with_no_evidence_and_no_frequency_stays_in_unit_range() -> None:
    concept = make_concept("c1", confidence=0.0, frequency=0)

    assert 0.0 <= utility(concept, 0, CONFIG) <= 1.0


def test_utility_clamps_to_zero_when_contradiction_outweighs_every_positive_term() -> None:
    concept = make_concept("c1", confidence=0.0, frequency=0, last_seen=0)
    concept.contradicting.append(make_evidence(supports=False))

    assert utility(concept, CONFIG.recency_halflife * 100, CONFIG) == 0.0


def test_utility_clamps_to_one_when_configured_weights_exceed_a_unit_sum() -> None:
    generous = KEEConfig(alpha_confidence=1.0, beta_recency=1.0, gamma_frequency=1.0)
    concept = make_concept("c1", confidence=1.0, frequency=1000, last_seen=10)

    assert utility(concept, 10, generous) == 1.0


def test_utility_is_lower_for_a_stale_concept_than_for_its_fresh_twin() -> None:
    stale = make_concept("stale", last_seen=0)
    fresh = make_concept("fresh", last_seen=100)

    assert utility(stale, 100, CONFIG) < utility(fresh, 100, CONFIG)


def test_utility_stays_in_unit_range_for_many_random_concepts() -> None:
    rng = random.Random(20260821)
    out_of_range = []
    for index in range(2000):
        concept = make_concept(
            f"c{index}",
            confidence=rng.uniform(0.0, 1.0),
            frequency=rng.randint(0, 5000),
            last_seen=rng.randint(-500, 500),
        )
        for _ in range(rng.randint(0, 6)):
            apply_evidence(
                concept,
                make_evidence(supports=rng.random() < 0.5, weight=rng.uniform(0.0, 5.0)),
                now=rng.randint(-500, 500),
                config=CONFIG,
            )
        score = utility(concept, rng.randint(-500, 500), CONFIG)
        if not 0.0 <= score <= 1.0:
            out_of_range.append((concept.id, score))

    assert out_of_range == []


# --- pruning ---------------------------------------------------------------


def test_prune_returns_no_ids_for_an_empty_store() -> None:
    assert prune(InMemoryGraphStore(), 10, CONFIG) == []


def test_prune_keeps_every_concept_scoring_above_the_threshold() -> None:
    store = InMemoryGraphStore()
    for index in range(3):
        store.add(make_concept(f"c{index}", confidence=1.0, frequency=500, last_seen=10))

    assert prune(store, 10, CONFIG) == []


def test_prune_removes_stale_low_confidence_concept_but_keeps_fresh_one() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("stale", confidence=0.0, frequency=1, last_seen=0))
    store.add(make_concept("fresh", confidence=1.0, frequency=200, last_seen=900))

    assert prune(store, 900, CONFIG) == ["stale"]


def test_prune_keeps_highest_utility_concept_when_all_below_threshold() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("weakest", confidence=0.0, frequency=0, last_seen=0))
    store.add(make_concept("best", confidence=0.2, frequency=1, last_seen=0))
    store.add(make_concept("middle", confidence=0.1, frequency=0, last_seen=0))

    prune(store, 900, CONFIG)

    assert [concept.id for concept in store.concepts()] == ["best"]


def test_prune_reports_the_survivor_as_not_removed_when_all_below_threshold() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("weakest", confidence=0.0, frequency=0, last_seen=0))
    store.add(make_concept("best", confidence=0.2, frequency=1, last_seen=0))

    assert prune(store, 900, CONFIG) == ["weakest"]


def test_prune_breaks_a_utility_tie_by_smallest_id() -> None:
    store = InMemoryGraphStore()
    for concept_id in ("b", "a", "c"):
        store.add(make_concept(concept_id, confidence=0.0, frequency=0, last_seen=0))

    prune(store, 900, CONFIG)

    assert [concept.id for concept in store.concepts()] == ["a"]


def test_prune_returns_removed_ids_in_sorted_order() -> None:
    store = InMemoryGraphStore()
    store.add(make_concept("keep", confidence=1.0, frequency=500, last_seen=900))
    for concept_id in ("z", "a", "m"):
        store.add(make_concept(concept_id, confidence=0.0, frequency=0, last_seen=0))

    assert prune(store, 900, CONFIG) == ["a", "m", "z"]


# --- edge formation --------------------------------------------------------


def test_co_occurrence_weight_is_zero_for_disjoint_tokens_and_subjects() -> None:
    a = make_concept("a", subject="auth", tokens=("alpha",))
    b = make_concept("b", subject="cache", tokens=("omega",))

    assert co_occurrence_weight(a, b) == 0.0


def test_co_occurrence_weight_is_zero_when_neither_concept_has_tokens() -> None:
    a = make_concept("a", subject="auth", tokens=())
    b = make_concept("b", subject="cache", tokens=())

    assert co_occurrence_weight(a, b) == 0.0


def test_co_occurrence_weight_is_one_for_identical_concepts() -> None:
    a = make_concept("a", subject="auth", tokens=("alpha", "beta"))
    b = make_concept("b", subject="auth", tokens=("alpha", "beta"))

    assert co_occurrence_weight(a, b) == 1.0


def test_co_occurrence_weight_rewards_a_shared_subject_over_disjoint_tokens() -> None:
    partner = make_concept("p", subject="auth", tokens=("zeta",))
    same_subject = make_concept("a", subject="auth", tokens=("alpha",))
    other_subject = make_concept("b", subject="cache", tokens=("alpha",))

    assert co_occurrence_weight(same_subject, partner) > co_occurrence_weight(
        other_subject, partner
    )


def test_co_occurrence_weight_is_symmetric() -> None:
    a = make_concept("a", subject="auth", tokens=("alpha", "beta"))
    b = make_concept("b", subject="cache", tokens=("beta", "gamma", "delta"))

    assert co_occurrence_weight(a, b) == co_occurrence_weight(b, a)
