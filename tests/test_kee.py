"""Tests for the Knowledge Evolution Engine's ingest rule.

The retrieval half is covered in ``test_activation.py``; what is tested here is
the belief-revision decision, because that is the only place the engine gets to
be cleverer than the baseline and the only place it can be wrong in an
interesting way.
"""

from __future__ import annotations

from atlas.config import ExperimentConfig, KEEConfig
from atlas.memory.kee import KnowledgeEvolutionEngine
from atlas.types import Experience


def experience(
    exp_id: str,
    fact_id: str,
    tokens: tuple[str, ...],
    *,
    subject: str = "retry",
    timestep: int = 1,
) -> Experience:
    return Experience(
        id=exp_id,
        fact_id=fact_id,
        text=" ".join(tokens),
        tokens=tokens,
        subject=subject,
        timestep=timestep,
    )


def engine(**kee_overrides: float) -> KnowledgeEvolutionEngine:
    config = ExperimentConfig(kee=KEEConfig(**kee_overrides))  # type: ignore[arg-type]
    return KnowledgeEvolutionEngine(config)


BACKOFF = ("retry", "backoff", "ceiling", "is", "500", "ms")
BACKOFF_REVISED = ("retry", "backoff", "ceiling", "is", "900", "ms")
UNRELATED = ("migration", "rollback", "runs", "nightly")


def test_an_unrelated_experience_creates_a_second_concept() -> None:
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", UNRELATED, subject="migration", timestep=2))

    assert kee.statistics()["concepts"] == 2.0
    assert kee.statistics()["revisions"] == 0.0


def test_an_identical_restatement_reinforces_rather_than_duplicating() -> None:
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f1", BACKOFF, timestep=2))

    stats = kee.statistics()
    assert stats["concepts"] == 1.0
    assert stats["merges"] == 1.0
    assert stats["revisions"] == 0.0


def test_a_near_duplicate_on_the_same_subject_is_read_as_a_revision() -> None:
    """The drift mechanism: same subject, one detail changed, so the old claim is news."""
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, timestep=2))

    stats = kee.statistics()
    assert stats["revisions"] == 1.0
    assert stats["concepts"] == 2.0


def test_a_revision_lowers_the_superseded_concept_confidence_below_the_new_one() -> None:
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, timestep=2))

    old = kee._store.get("c-e1")
    new = kee._store.get("c-e2")
    assert old is not None
    assert new is not None
    assert old.confidence < new.confidence


def test_a_revision_does_not_credit_the_old_concept_with_the_new_fact() -> None:
    """Contradicting evidence is not an ability to answer for the fact that contradicts."""
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, timestep=2))

    old = kee._store.get("c-e1")
    assert old is not None
    assert old.fact_ids == {"f1"}


def test_high_overlap_on_a_different_subject_never_triggers_a_revision() -> None:
    """Coincidental wording across unrelated areas of the codebase is not contradiction."""
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF, subject="retry"))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, subject="cache", timestep=2))

    assert kee.statistics()["revisions"] == 0.0
    assert kee.statistics()["concepts"] == 2.0


def test_overlap_below_the_revision_threshold_leaves_the_old_concept_untouched() -> None:
    kee = engine(revision_threshold=0.95, merge_threshold=0.99)
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, timestep=2))

    old = kee._store.get("c-e1")
    assert old is not None
    assert old.contradicting == []
    assert kee.statistics()["revisions"] == 0.0


def test_retrieval_surfaces_an_ingested_fact() -> None:
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))

    result = kee.retrieve(BACKOFF, top_k=5)

    assert "f1" in result.fact_ids


def test_retrieve_on_an_empty_engine_returns_nothing_rather_than_raising() -> None:
    result = engine().retrieve(BACKOFF, top_k=5)

    assert result.items == ()
    assert result.fact_ids == frozenset()


def test_consolidate_on_an_empty_engine_does_not_raise() -> None:
    engine().consolidate(timestep=10)


def drifted_engine(**kee_overrides: float) -> KnowledgeEvolutionEngine:
    """A world where one claim is superseded and an unrelated claim is not."""
    kee = engine(**kee_overrides)
    kee.ingest(experience("e1", "f1", BACKOFF))
    kee.ingest(experience("e2", "f2", BACKOFF_REVISED, timestep=2))
    kee.ingest(experience("e3", "f3", UNRELATED, subject="migration", timestep=2))
    kee.consolidate(timestep=40)
    return kee


def test_a_superseded_concept_is_eventually_forgotten_while_its_peers_survive() -> None:
    """The whole drift claim, end to end: contradiction plus age must lose."""
    kee = drifted_engine()

    assert kee._store.get("c-e1") is None
    assert kee._store.get("c-e2") is not None
    assert kee._store.get("c-e3") is not None


def test_the_superseded_fact_is_no_longer_retrievable_after_consolidation() -> None:
    """Forgetting only counts if it changes what a grader sees."""
    result = drifted_engine().retrieve(BACKOFF, top_k=5)

    assert result.fact_ids == frozenset({"f2"})


def test_without_the_revision_branch_the_stale_fact_survives_and_is_returned() -> None:
    """Pins the previous two tests to belief revision rather than to age alone.

    Raising ``revision_threshold`` above the observed overlap disables the
    revision branch and nothing else. If the stale concept still disappeared,
    the tests above would be measuring the recency decay every concept gets.
    """
    result = drifted_engine(revision_threshold=0.99).retrieve(BACKOFF, top_k=5)

    assert result.fact_ids == frozenset({"f1", "f2"})


def test_statistics_reports_a_float_for_every_key() -> None:
    kee = engine()
    kee.ingest(experience("e1", "f1", BACKOFF))

    assert all(isinstance(value, float) for value in kee.statistics().values())
