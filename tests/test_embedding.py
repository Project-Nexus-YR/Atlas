"""Behaviour of the seeded hashing embedder."""

from __future__ import annotations

import math
import random
import subprocess
import sys
import warnings

import numpy as np
import pytest

from atlas.embedding import HashingEmbedder, cosine

# Run in a fresh interpreter with a fresh hash salt. An in-process assertion
# cannot catch builtin hash() being used, because the salt is fixed for the life
# of one process; only a second process exposes it.
_CHILD_PROGRAM = """
import sys
from atlas.embedding import HashingEmbedder
vector = HashingEmbedder(dim=64, seed=7).encode(("alpha", "beta", "alpha"))
sys.stdout.write(vector.tobytes().hex())
"""


def test_same_tokens_same_seed_produce_identical_vector() -> None:
    tokens = ("parser", "cache", "parser")

    first = HashingEmbedder(dim=128, seed=3).encode(tokens)
    second = HashingEmbedder(dim=128, seed=3).encode(tokens)

    assert np.array_equal(first, second)


def test_different_seeds_produce_different_vectors() -> None:
    tokens = ("parser", "cache", "retry")

    low = HashingEmbedder(dim=128, seed=3).encode(tokens)
    high = HashingEmbedder(dim=128, seed=4).encode(tokens)

    assert not np.array_equal(low, high)


def test_vector_is_bit_identical_in_a_separate_process() -> None:
    expected = HashingEmbedder(dim=64, seed=7).encode(("alpha", "beta", "alpha"))

    child = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
    )

    assert child.stdout == expected.tobytes().hex()


def test_token_order_does_not_change_the_vector() -> None:
    forward = HashingEmbedder(dim=128, seed=3).encode(("alpha", "beta", "gamma"))
    reversed_ = HashingEmbedder(dim=128, seed=3).encode(("gamma", "beta", "alpha"))

    assert np.array_equal(forward, reversed_)


def test_empty_token_tuple_returns_the_zero_vector_without_warning() -> None:
    embedder = HashingEmbedder(dim=32, seed=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vector = embedder.encode(())

    assert not vector.any()
    assert vector.shape == (32,)


def test_repeating_a_token_raises_its_weight_less_than_proportionally() -> None:
    # "alpha" and "beta" land in distinct buckets at this dim and seed, so the
    # cosine below is decided purely by their two term weights.
    embedder = HashingEmbedder(dim=4096, seed=11)
    alpha = embedder.encode(("alpha",))
    flat_weighting = 1 / math.sqrt(2)
    linear_weighting = 10 / math.sqrt(101)

    similarity = cosine(embedder.encode(("alpha",) * 10 + ("beta",)), alpha)

    assert flat_weighting < similarity < linear_weighting


def test_every_encoded_vector_is_unit_length_or_zero() -> None:
    rng = random.Random(20260821)
    embedder = HashingEmbedder(dim=32, seed=5)
    vocabulary = [f"token{index}" for index in range(40)]

    for _ in range(400):
        tokens = tuple(rng.choice(vocabulary) for _ in range(rng.randint(0, 12)))
        norm = float(np.linalg.norm(embedder.encode(tokens)))
        # Zero is reachable without being a bug: signed collisions can cancel.
        assert norm == pytest.approx(1.0) or norm == 0.0


def test_encode_query_with_uniform_idf_matches_plain_encode() -> None:
    embedder = HashingEmbedder(dim=128, seed=3)
    tokens = ("parser", "cache", "parser")

    weighted = embedder.encode_query(tokens, dict.fromkeys(tokens, 1.0))

    assert np.array_equal(weighted, embedder.encode(tokens))


def test_encode_query_tilts_the_vector_towards_the_higher_idf_term() -> None:
    embedder = HashingEmbedder(dim=4096, seed=11)
    alpha = embedder.encode(("alpha",))
    beta = embedder.encode(("beta",))

    query = embedder.encode_query(("alpha", "beta"), {"alpha": 1.0, "beta": 3.0})

    assert cosine(query, beta) > cosine(query, alpha)


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    vector = HashingEmbedder(dim=64, seed=2).encode(("alpha", "beta"))

    assert cosine(vector, vector) == pytest.approx(1.0)


def test_cosine_with_the_zero_vector_is_zero_not_nan() -> None:
    embedder = HashingEmbedder(dim=64, seed=2)

    assert cosine(embedder.encode(("alpha",)), embedder.encode(())) == 0.0
