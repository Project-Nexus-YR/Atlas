"""Behaviour of the Vector-RAG baseline arm."""

from __future__ import annotations

import math
import subprocess
import sys
import warnings

import pytest

from atlas.config import ExperimentConfig
from atlas.embedding import HashingEmbedder
from atlas.memory.vector_rag import VectorRAGMemory
from atlas.types import Experience, MemorySystem, RetrievalResult

_NEAR = ("cache", "invalidation", "parser")
_FAR = ("kubernetes", "ingress", "yaml")

_SUBPROCESS_CORPUS = (
    ("parser", "cache"),
    ("parser", "cache"),
    ("timeout", "parser"),
    ("kubernetes", "ingress"),
)
_SUBPROCESS_QUERY = ("parser", "cache", "timeout")

# Idf is computed from a dict of document frequencies at retrieval time, so the
# ranking has a second chance to pick up per-process state. Rebuild the whole
# corpus in a fresh interpreter and compare the scores bit for bit.
_CHILD_RANKING = f"""
import sys
from atlas.config import ExperimentConfig
from atlas.embedding import HashingEmbedder
from atlas.memory.vector_rag import VectorRAGMemory
from atlas.types import Experience

memory = VectorRAGMemory(
    HashingEmbedder(dim=256, seed=13), ExperimentConfig(seed=13, embedding_dim=256)
)
for index, tokens in enumerate({_SUBPROCESS_CORPUS!r}):
    memory.ingest(
        Experience(
            id="e" + str(index),
            fact_id="f" + str(index),
            text=" ".join(tokens),
            tokens=tokens,
            subject="parser",
            timestep=index,
        )
    )
result = memory.retrieve({_SUBPROCESS_QUERY!r}, top_k=4)
sys.stdout.write("|".join(item.concept_id + ":" + repr(item.score) for item in result.items))
"""


def _memory(dim: int = 256) -> VectorRAGMemory:
    config = ExperimentConfig(seed=13, embedding_dim=dim)
    return VectorRAGMemory(HashingEmbedder(dim=config.embedding_dim, seed=config.seed), config)


def _experience(index: int, tokens: tuple[str, ...]) -> Experience:
    return Experience(
        id=f"e{index}",
        fact_id=f"f{index}",
        text=" ".join(tokens),
        tokens=tokens,
        subject="parser",
        timestep=index,
    )


def _score_of(result: RetrievalResult, concept_id: str) -> float:
    return next(item.score for item in result.items if item.concept_id == concept_id)


def _ranking(result: RetrievalResult) -> str:
    return "|".join(item.concept_id + ":" + repr(item.score) for item in result.items)


def _score_with_repeater(repeater_tokens: tuple[str, ...]) -> float:
    """Score of one fixed target document, in a corpus whose other doc repeats "term"."""
    memory = _memory()
    memory.ingest(_experience(0, repeater_tokens))
    memory.ingest(_experience(1, ("term", "alpha")))
    for index in range(2, 5):
        memory.ingest(_experience(index, (f"filler{index}", "noise")))
    return _score_of(memory.retrieve(("term", "alpha"), top_k=5), "e1")


def test_arm_conforms_to_the_memory_system_protocol() -> None:
    assert isinstance(_memory(), MemorySystem)


def test_retrieve_on_an_empty_store_returns_nothing() -> None:
    result = _memory().retrieve(_NEAR, top_k=5)

    assert result.items == ()
    assert result.fact_ids == frozenset()
    assert result.expanded_nodes == 0


def test_single_experience_is_retrieved_by_its_own_tokens() -> None:
    memory = _memory()
    memory.ingest(_experience(0, _NEAR))

    result = memory.retrieve(_NEAR, top_k=5)

    assert [item.concept_id for item in result.items] == ["e0"]
    assert result.fact_ids == frozenset({"f0"})


def test_lexically_near_experience_outranks_a_far_one() -> None:
    memory = _memory()
    memory.ingest(_experience(0, _FAR))
    memory.ingest(_experience(1, _NEAR))

    result = memory.retrieve(("cache", "parser"), top_k=2)

    assert [item.concept_id for item in result.items] == ["e1", "e0"]
    assert result.items[0].score > result.items[1].score


def test_top_k_larger_than_the_corpus_returns_every_experience() -> None:
    memory = _memory()
    memory.ingest(_experience(0, _NEAR))
    memory.ingest(_experience(1, _FAR))

    result = memory.retrieve(_NEAR, top_k=10)

    assert {item.concept_id for item in result.items} == {"e0", "e1"}


def test_query_of_unknown_tokens_scores_below_a_matching_query() -> None:
    memory = _memory()
    memory.ingest(_experience(0, _NEAR))
    memory.ingest(_experience(1, _FAR))

    unknown = memory.retrieve(("zzzznothing", "qqqqabsent"), top_k=1)
    matching = memory.retrieve(_NEAR, top_k=1)

    assert unknown.items[0].score < matching.items[0].score


def test_expanded_nodes_counts_every_vector_scored() -> None:
    memory = _memory()
    for index in range(4):
        memory.ingest(_experience(index, (f"topic{index}", "shared")))

    result = memory.retrieve(("shared",), top_k=2)

    assert len(result.items) == 2
    assert result.expanded_nodes == 4


def test_index_still_ranks_correctly_after_growing_past_its_initial_capacity() -> None:
    memory = _memory()
    for index in range(50):
        memory.ingest(_experience(index, (f"topic{index}", "shared")))

    # The first row is the one that has to survive every doubling, so it is the
    # one that exposes a growth step that dropped what it was copying.
    result = memory.retrieve(("topic0", "shared"), top_k=1)

    assert result.items[0].concept_id == "e0"


def test_retrieved_fact_ids_are_the_union_over_returned_items() -> None:
    memory = _memory()
    for index in range(3):
        memory.ingest(_experience(index, (f"topic{index}", "shared")))

    result = memory.retrieve(("shared",), top_k=3)

    assert result.fact_ids == frozenset({"f0", "f1", "f2"})


def test_consolidate_leaves_retrieval_and_size_unchanged() -> None:
    memory = _memory()
    for index in range(3):
        memory.ingest(_experience(index, (f"topic{index}", "shared")))
    before = memory.retrieve(("shared",), top_k=3)
    statistics_before = memory.statistics()

    memory.consolidate(timestep=99)

    assert memory.retrieve(("shared",), top_k=3) == before
    assert memory.statistics() == statistics_before


def test_a_token_common_to_every_document_is_downweighted_relative_to_a_rare_one() -> None:
    memory = _memory()
    for index in range(10):
        memory.ingest(_experience(index, ("common", f"filler{index}")))
    memory.ingest(_experience(10, ("unique", "beta")))

    result = memory.retrieve(("common", "unique"), top_k=11)

    # Both documents match exactly one query term, so without idf they tie.
    assert _score_of(result, "e10") > _score_of(result, "e0")


def test_idf_promotes_the_discriminator_over_distractors_sharing_subject_vocabulary() -> None:
    memory = _memory()
    for index in range(12):
        memory.ingest(_experience(index, ("parser", "cache")))
    memory.ingest(_experience(12, ("timeout", "parser")))

    result = memory.retrieve(("parser", "cache", "timeout"), top_k=13)

    # Distractors overlap the query on two of three tokens, as the DISTRACTOR
    # family is built to; only "timeout" discriminates, and only idf knows that.
    assert _score_of(result, "e12") > max(_score_of(result, f"e{index}") for index in range(12))


def test_unseen_query_token_does_not_divide_by_zero() -> None:
    memory = _memory()
    memory.ingest(_experience(0, _NEAR))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = memory.retrieve(("neverseenbefore", "parser"), top_k=1)

    assert math.isfinite(result.items[0].score)


def test_document_frequency_tracks_documents_not_occurrences() -> None:
    repeated = _score_with_repeater(("term",) * 5 + ("noise",))
    once = _score_with_repeater(("term", "noise"))

    assert repeated == pytest.approx(once)


def test_ranking_and_scores_are_identical_in_a_separate_process() -> None:
    memory = _memory()
    for index, tokens in enumerate(_SUBPROCESS_CORPUS):
        memory.ingest(_experience(index, tokens))
    expected = _ranking(memory.retrieve(_SUBPROCESS_QUERY, top_k=4))

    child = subprocess.run(
        [sys.executable, "-c", _CHILD_RANKING],
        capture_output=True,
        text=True,
        check=True,
    )

    assert child.stdout == expected


def test_statistics_report_one_concept_and_one_vector_per_experience() -> None:
    memory = _memory(dim=64)
    for index in range(3):
        memory.ingest(_experience(index, (f"topic{index}", "shared")))

    assert memory.statistics() == {"concepts": 3.0, "vectors": 3.0, "dim": 64.0}
