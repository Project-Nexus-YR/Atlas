"""The concept lifecycle: what a concept is worth, and what survives.

This module owns the half of the Knowledge Evolution Engine that decides which
concepts persist. Every function takes :class:`KEEConfig` explicitly rather than
reading a module-level default, so a sweep over hyper-parameters is a loop over
configs and never a mutation of global state.

Two shape choices here are the actual contribution, and both replace a form the
previous Atlas used:

* recency decays exponentially, not linearly to a hard floor;
* frequency saturates asymptotically, not by clipping at a ceiling.

Both old forms destroyed ordering information among exactly the concepts the
engine most needs to rank - the old ones and the frequently-seen ones.
"""

from __future__ import annotations

from atlas.config import KEEConfig
from atlas.memory.store import GraphStore
from atlas.types import Concept, Evidence

__all__ = [
    "apply_evidence",
    "co_occurrence_weight",
    "contradiction_ratio",
    "frequency_factor",
    "prune",
    "recency_factor",
    "utility",
]

_FREQUENCY_SATURATION = 10.0
"""Frequency at which :func:`frequency_factor` returns 0.5."""

_CONFIDENCE_PRIOR = 1.0
"""Laplace prior: an unobserved concept sits at 0.5, not at certainty."""

_SUBJECT_BONUS = 0.25
"""Edge credit for sharing a subject when tokens happen not to overlap."""


def recency_factor(last_seen: int, now: int, halflife: int) -> float:
    """Return ``0.5 ** (age / halflife)`` in ``(0, 1]``.

    Exponential rather than linear. A linear ``1 - age / window`` reaches exactly
    0.0 at the window edge and stays there, so every concept older than the
    window scores identically and the engine loses its ability to prefer the
    less stale of two stale concepts. Exponential decay never reaches zero, so
    ordering by age survives at every age, which is what the pruner needs.

    ``age`` is clamped at 0 so a concept stamped with a future timestep scores as
    perfectly fresh instead of being rewarded past 1.0. ``halflife`` is validated
    ``>= 1`` by :class:`KEEConfig`.
    """
    age = max(0, now - last_seen)
    return float(0.5 ** (age / halflife))


def frequency_factor(frequency: int) -> float:
    """Return saturating ``f / (f + k)`` in ``[0, 1)``, strictly increasing in ``f``.

    Chosen over ``min(1.0, f / 100)``, which clips: every concept seen 100 or
    more times tied at 1.0, and ties are what make a ranked retrieval arbitrary.
    This form is monotonic forever and never actually attains 1.0, so repetition
    always buys a little more - just less each time, which is the diminishing
    return we want repetition to have.
    """
    seen = max(0, frequency)
    return seen / (seen + _FREQUENCY_SATURATION)


def contradiction_ratio(concept: Concept) -> float:
    """Return contradicting evidence as a share of all evidence, in ``[0, 1]``.

    Counted, not weight-summed, deliberately. Confidence already integrates the
    evidence *weights*; if this term reused them it would be close to a monotone
    restatement of ``1 - confidence`` and the ``alpha`` and ``delta`` terms of
    :func:`utility` would partly cancel. Counting instead makes it a separate
    signal: how disputed a concept is, independent of how strongly.

    Returns 0.0 with no evidence at all - an unexamined concept is not a
    contradicted one.
    """
    total = len(concept.supporting) + len(concept.contradicting)
    if total == 0:
        return 0.0
    return len(concept.contradicting) / total


def utility(concept: Concept, now: int, config: KEEConfig) -> float:
    """Score a concept's worth as ``a*conf + b*recency + c*freq - d*contradiction``.

    The four weights are config (``alpha_confidence``, ``beta_recency``,
    ``gamma_frequency``, ``delta_contradiction``), not constants, because their
    balance is exactly what an ablation varies.

    The result is clamped to ``[0, 1]`` so it is directly comparable with
    ``config.utility_threshold``, which is expressed on that scale. Without the
    clamp a weight sweep that raises the positive weights would silently move
    what the threshold means, and a heavily contradicted concept would score
    negative - unbounded below, so no threshold could be interpreted.
    """
    score = (
        config.alpha_confidence * concept.confidence
        + config.beta_recency
        * recency_factor(concept.last_seen_timestep, now, config.recency_halflife)
        + config.gamma_frequency * frequency_factor(concept.frequency)
        - config.delta_contradiction * contradiction_ratio(concept)
    )
    return min(1.0, max(0.0, score))


def apply_evidence(concept: Concept, evidence: Evidence, now: int, config: KEEConfig) -> None:
    """Fold one piece of evidence into ``concept``, mutating it in place.

    Confidence is recomputed as a Laplace-smoothed beta mean over evidence
    weights, ``(S + 1) / (S + C + 2)``. This is chosen over the usual
    increment-on-agreement rule because that rule can only ever climb:
    contradiction either does nothing or is capped, and the graph fills with
    confidently wrong concepts that outlive their corrections. Here contradicting
    weight enters the denominator only, so it genuinely drives confidence down,
    and a concept with no evidence sits at the prior 0.5 rather than at 0 or 1.

    Only supporting evidence contributes a ``fact_id``: the grader reads
    ``fact_ids`` as facts this concept can answer for, and evidence that the fact
    is wrong is not an ability to answer for it.
    """
    bucket = concept.supporting if evidence.supports else concept.contradicting
    bucket.append(evidence)
    if evidence.supports:
        concept.fact_ids.add(evidence.fact_id)
    concept.frequency += 1
    concept.last_seen_timestep = now
    supporting_weight = sum(item.weight for item in concept.supporting)
    contradicting_weight = sum(item.weight for item in concept.contradicting)
    concept.confidence = (supporting_weight + _CONFIDENCE_PRIOR) / (
        supporting_weight + contradicting_weight + 2 * _CONFIDENCE_PRIOR
    )
    concept.utility = utility(concept, now, config)


def prune(store: GraphStore, now: int, config: KEEConfig) -> list[str]:
    """Remove every concept below ``config.utility_threshold``; return removed ids sorted.

    Guard: the store is never emptied. If every concept scores below the
    threshold, the single highest-utility one is kept. An empty graph would make
    the next retrieval trivially return nothing, and a memory arm that answers
    nothing scores as a clean zero rather than as broken - a benchmark artefact
    that flatters the comparison instead of measuring it. Keeping one concept
    costs almost nothing and keeps the failure legible.

    Ties on utility are broken by the lexicographically smallest id, matching the
    ordering convention in :mod:`atlas.memory.store`, so a run is reproducible.
    """
    scored = [(concept.id, utility(concept, now, config)) for concept in store.concepts()]
    doomed = [concept_id for concept_id, score in scored if score < config.utility_threshold]
    if scored and len(doomed) == len(scored):
        keeper = min(scored, key=lambda pair: (-pair[1], pair[0]))[0]
        doomed = [concept_id for concept_id in doomed if concept_id != keeper]
    for concept_id in doomed:
        store.remove(concept_id)
    return sorted(doomed)


def co_occurrence_weight(a: Concept, b: Concept) -> float:
    """Score how strongly two concepts should be linked, in ``[0, 1]``.

    Jaccard over tokens rather than raw overlap count, so a verbose concept
    cannot collect edges to everything simply by having more tokens to match
    with. The subject bonus exists because two concepts about the same area of
    the codebase are genuinely related even when their wordings share nothing,
    and that case - same subject, disjoint tokens - is precisely the multi-hop
    link a vector baseline cannot make.
    """
    a_tokens = set(a.tokens)
    b_tokens = set(b.tokens)
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
    bonus = _SUBJECT_BONUS if a.subject == b.subject else 0.0
    return min(1.0, jaccard + bonus)
