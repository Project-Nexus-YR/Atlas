"""Tests for the generated world.

The determinism pair matters most here: a generator that quietly ignored its
seed would pass a same-seed test on its own, so it is always accompanied by a
different-seed test that proves the seed reaches the output.
"""

from __future__ import annotations

from dataclasses import fields

from atlas.bench.world import build_world
from atlas.config import WorldConfig
from atlas.types import Experience


def _config(
    subjects: int = 6,
    facts_per_subject: int = 4,
    drift_rate: float = 0.25,
    distractor_ratio: float = 2.0,
) -> WorldConfig:
    return WorldConfig(
        subjects=subjects,
        facts_per_subject=facts_per_subject,
        drift_rate=drift_rate,
        distractor_ratio=distractor_ratio,
        tasks=40,
    )


def test_same_seed_produces_identical_world() -> None:
    first = build_world(_config(), seed=7)
    second = build_world(_config(), seed=7)

    assert first.facts == second.facts
    assert first.experiences == second.experiences


def test_different_seeds_produce_different_worlds() -> None:
    first = build_world(_config(), seed=7)
    second = build_world(_config(), seed=8)

    assert [e.text for e in first.experiences] != [e.text for e in second.experiences]


def test_superseded_fact_has_strictly_later_timestep_than_its_predecessor() -> None:
    world = build_world(_config(), seed=11)

    assert world.drift_pairs
    for old_id, new_id in world.drift_pairs:
        old, new = world.facts[old_id], world.facts[new_id]
        assert new.timestep > old.timestep, f"{new_id} arrived before {old_id}"
        assert new.supersedes == old_id


def test_replacement_changes_one_detail_of_an_otherwise_identical_fact() -> None:
    world = build_world(_config(), seed=11)

    for old_id, new_id in world.drift_pairs:
        old, new = world.facts[old_id], world.facts[new_id]
        shared = sum(1 for a, b in zip(old.tokens, new.tokens, strict=False) if a == b)
        assert old.subject == new.subject
        assert shared >= 3, f"{new_id} is not lexically confusable with {old_id}"
        assert old.text != new.text


def test_every_fact_is_delivered_as_exactly_one_experience() -> None:
    world = build_world(_config(), seed=3)

    delivered = [experience.fact_id for experience in world.experiences]

    assert sorted(delivered) == sorted(world.facts)


def test_experience_stream_is_ordered_by_timestep_without_gaps() -> None:
    world = build_world(_config(), seed=3)

    steps = [experience.timestep for experience in world.experiences]

    assert steps == list(range(1, len(world.facts) + 1))


def test_corpus_size_follows_drift_rate_and_distractor_ratio() -> None:
    config = _config()
    world = build_world(config, seed=5)
    base = config.subjects * config.facts_per_subject

    assert len(world.drift_pairs) == round(config.drift_rate * base)
    assert len(world.distractor_ids) == round(config.distractor_ratio * base)
    assert len(world.facts) == base + len(world.drift_pairs) + len(world.distractor_ids)


def test_no_two_facts_carry_identical_text() -> None:
    world = build_world(_config(), seed=5)

    texts = [fact.text for fact in world.facts.values()]

    assert len(set(texts)) == len(texts)


def test_facts_in_the_stable_pool_overlap_only_on_the_subject_word() -> None:
    world = build_world(_config(), seed=13)

    for subject, ids in world.stable_by_subject.items():
        for first in ids:
            for second in ids:
                if first == second:
                    continue
                overlap = set(world.facts[first].tokens) & set(world.facts[second].tokens)
                assert overlap == {subject}, f"{first} and {second} share {overlap}"


def test_the_experience_stream_does_not_reveal_which_facts_were_superseded() -> None:
    # The whole DRIFT result rests on this. An arm sees experiences only, so if
    # one ever carried the supersedes label, drift would stop being inferred and
    # start being read off a planted field, and the family would measure nothing.
    world = build_world(_config(), seed=13)

    visible = {field.name for field in fields(Experience)}

    assert "supersedes" not in visible
    assert all("supersed" not in experience.text for experience in world.experiences)


def test_superseded_facts_are_absent_from_the_stable_pool() -> None:
    world = build_world(_config(), seed=13)
    stable = {fid for ids in world.stable_by_subject.values() for fid in ids}

    for old_id, _ in world.drift_pairs:
        assert old_id not in stable


def test_world_without_drift_or_distractors_still_builds() -> None:
    config = _config(drift_rate=0.0, distractor_ratio=0.0)

    world = build_world(config, seed=2)

    assert world.drift_pairs == ()
    assert world.distractor_ids == frozenset()
    assert len(world.facts) == config.subjects * config.facts_per_subject


def test_subject_vocabulary_extends_beyond_the_built_in_word_list() -> None:
    world = build_world(_config(subjects=30), seed=2)

    subjects = {fact.subject for fact in world.facts.values()}

    assert len(subjects) == 30
