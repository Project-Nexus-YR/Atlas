"""Behaviour of the BM25 control arm.

The arm is a control, so what is pinned here is that it behaves like textbook
Okapi BM25 and not like something tuned to lose: saturation and length
normalisation both bite, the idf floor never subtracts score from a document
that genuinely matches, and the ranking is bit-identical in a fresh interpreter.
A control that quietly under-performed would make the graph arm look better than
it is, which is the one failure mode this file exists to rule out.
"""

from __future__ import annotations

import subprocess
import sys

from atlas.config import BM25Config, ExperimentConfig
from atlas.memory.bm25 import BM25Memory
from atlas.types import Experience, MemorySystem

_SUBPROCESS_CORPUS = (
    ("parser", "cache"),
    ("parser", "cache"),
    ("timeout", "parser"),
    ("kubernetes", "ingress"),
)
_SUBPROCESS_QUERY = ("parser", "cache", "timeout")

# Scores are accumulated by summing over query tokens, and set iteration order
# for strings is not stable across processes. Rebuild the corpus in a fresh
# interpreter and compare the scores bit for bit.
_CHILD_RANKING = f"""
import sys
from atlas.config import ExperimentConfig
from atlas.memory.bm25 import BM25Memory
from atlas.types import Experience

memory = BM25Memory(ExperimentConfig(seed=13))
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


def _memory(**overrides: float) -> BM25Memory:
    config = ExperimentConfig(seed=13, bm25=BM25Config(**overrides)) if overrides else None
    return BM25Memory(config or ExperimentConfig(seed=13))


def _ingest(memory: BM25Memory, index: int, tokens: tuple[str, ...]) -> None:
    memory.ingest(
        Experience(
            id=f"e{index}",
            fact_id=f"f{index}",
            text=" ".join(tokens),
            tokens=tokens,
            subject="parser",
            timestep=index,
        )
    )


def _scores(memory: BM25Memory, query: tuple[str, ...], top_k: int = 10) -> dict[str, float]:
    return {item.concept_id: item.score for item in memory.retrieve(query, top_k).items}


def test_it_satisfies_the_memory_system_protocol() -> None:
    assert isinstance(_memory(), MemorySystem)


def test_an_empty_index_returns_nothing() -> None:
    assert _memory().retrieve(("parser",), top_k=5).items == ()


def test_the_document_matching_more_query_tokens_ranks_first() -> None:
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache", "timeout"))
    _ingest(memory, 1, ("parser", "ingress", "yaml"))

    ranked = memory.retrieve(("parser", "cache", "timeout"), top_k=2).items

    assert [item.concept_id for item in ranked] == ["e0", "e1"]


def test_a_document_sharing_no_query_token_is_never_returned() -> None:
    """The documented asymmetry with the vector arm: fewer than ``top_k`` is allowed."""
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache"))
    _ingest(memory, 1, ("kubernetes", "ingress"))
    _ingest(memory, 2, ("terraform", "state"))

    result = memory.retrieve(("parser",), top_k=3)

    assert [item.concept_id for item in result.items] == ["e0"]


def test_expanded_nodes_counts_only_the_documents_the_query_touched() -> None:
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache"))
    _ingest(memory, 1, ("parser", "ingress"))
    _ingest(memory, 2, ("terraform", "state"))

    assert memory.retrieve(("parser",), top_k=3).expanded_nodes == 2


def test_term_frequency_saturates_instead_of_accumulating_linearly() -> None:
    """Length normalisation is switched off so the effect measured is saturation alone."""
    memory = _memory(b=0.0)
    _ingest(memory, 0, ("parser",))
    _ingest(memory, 1, ("parser",) * 5)

    scores = _scores(memory, ("parser",))

    assert scores["e0"] < scores["e1"] < 5 * scores["e0"]


def test_length_normalisation_prefers_the_shorter_of_two_equal_matches() -> None:
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache"))
    _ingest(memory, 1, ("parser", "a", "b", "c", "d", "e"))

    scores = _scores(memory, ("parser",))

    assert scores["e0"] > scores["e1"]


def test_a_token_present_in_every_document_still_scores_above_zero() -> None:
    """The ``1 +`` floor on idf. Without it this term goes negative on a small corpus."""
    memory = _memory()
    for index in range(4):
        _ingest(memory, index, ("parser", f"token{index}"))

    assert all(score > 0.0 for score in _scores(memory, ("parser",)).values())


def test_retrieved_items_carry_the_fact_ids_the_grader_reads() -> None:
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache"))

    assert _memory_fact_ids(memory) == {"f0"}


def _memory_fact_ids(memory: BM25Memory) -> set[str]:
    return {
        fact_id for item in memory.retrieve(("parser",), top_k=1).items for fact_id in item.fact_ids
    }


def test_consolidating_changes_nothing_at_all() -> None:
    memory = _memory()
    _ingest(memory, 0, ("parser", "cache"))
    before = _scores(memory, ("parser",)), memory.statistics()

    memory.consolidate(timestep=99)

    assert (_scores(memory, ("parser",)), memory.statistics()) == before


def test_the_ranking_is_identical_in_a_fresh_interpreter() -> None:
    memory = _memory()
    for index, tokens in enumerate(_SUBPROCESS_CORPUS):
        _ingest(memory, index, tokens)
    here = "|".join(
        f"{item.concept_id}:{item.score!r}"
        for item in memory.retrieve(_SUBPROCESS_QUERY, top_k=4).items
    )

    child = subprocess.run(
        [sys.executable, "-c", _CHILD_RANKING], capture_output=True, text=True, check=True
    )

    assert child.stdout == here
