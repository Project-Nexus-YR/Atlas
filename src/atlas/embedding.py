"""Deterministic text embedding for the vector arm.

A published number has to be reproducible on someone else's machine, so the
embedding is seeded feature hashing rather than a learned model: no download, no
vocabulary to drift, and a vector that is bit-identical in any process.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["Embedder", "HashingEmbedder", "cosine"]


class Embedder(Protocol):
    """Turns a bag of tokens into a fixed-width L2-normalised vector."""

    dim: int

    def encode(self, tokens: tuple[str, ...]) -> NDArray[np.float64]:
        """Return the unit vector for ``tokens``, or the zero vector for no tokens."""
        ...

    def encode_query(
        self, tokens: tuple[str, ...], idf: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return the unit vector for ``tokens`` with each term scaled by its idf."""
        ...


class HashingEmbedder:
    """Seeded feature hashing - the hashing trick - with sublinear tf weighting.

    Two details are what keep the baseline honest rather than weak. The sign
    drawn per token makes collisions cancel in expectation instead of piling up,
    so a small ``dim`` degrades gracefully. And weighting a token by
    ``1 + log(count)`` stops one term repeated ten times from dominating the
    direction of the vector, which is the standard tf damping a real retriever
    would have.

    Documents are encoded with :meth:`encode` and queries with
    :meth:`encode_query`, which additionally multiplies in the idf its caller
    measured. That asymmetry is the SMART ``lnc.ltc`` split: idf never enters a
    stored vector, so it can move with the corpus without one being recomputed.
    """

    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        # blake2b is keyed, so the seed is baked into the digest itself. Python's
        # builtin hash() is salted per process and would silently make two runs
        # of the same experiment incomparable.
        self._key = str(seed).encode("utf-8")

    def encode(self, tokens: tuple[str, ...]) -> NDArray[np.float64]:
        """Return the L2-normalised bag-of-tokens vector for ``tokens``."""
        return self._encode(tokens, None)

    def encode_query(
        self, tokens: tuple[str, ...], idf: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """Return the query vector, each term additionally scaled by ``idf[token]``.

        ``idf`` must cover every distinct token in ``tokens``; supplying it per
        call is what keeps the weights current without touching stored vectors.
        """
        return self._encode(tokens, idf)

    def _encode(
        self, tokens: tuple[str, ...], idf: Mapping[str, float] | None
    ) -> NDArray[np.float64]:
        """Accumulate weighted token buckets and normalise.

        Iteration follows ``Counter`` insertion order rather than set order, so
        tokens colliding into one bucket are summed in the same sequence in
        every process and the result stays bit-identical.
        """
        vector: NDArray[np.float64] = np.zeros(self.dim, dtype=np.float64)
        for token, count in Counter(tokens).items():
            bucket, sign = self._bucket(token)
            weight = 1.0 + math.log(count)
            if idf is not None:
                weight *= idf[token]
            vector[bucket] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        normalised: NDArray[np.float64] = vector / norm
        return normalised

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to its bucket and sign, deterministically for this seed."""
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=self._key).digest()
        value = int.from_bytes(digest, "big")
        # One digest, two independent draws: the top bit signs, the rest buckets.
        return value % self.dim, 1.0 if value & (1 << 63) else -1.0


def cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cosine similarity, defined as 0.0 when either side is the zero vector."""
    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
