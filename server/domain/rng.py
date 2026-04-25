"""Seeded, deterministic RNG helper.

Deterministic RNG is critical for an enterprise environment so that training
runs, evaluations, and bug reports can be reproduced exactly. We expose a
small wrapper around `random.Random` that cannot be confused with the global
`random` module.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


class SeededRNG:
    """Deterministic RNG with a human-readable episode seed."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    @property
    def seed(self) -> int:
        return self._seed

    def child(self, label: str) -> "SeededRNG":
        """Derive a deterministic child RNG keyed by `label`.

        This lets us isolate randomness per incident / per signal stream so
        adding a new incident cannot shift outcomes in unrelated incidents.
        """
        digest = hashlib.sha256(f"{self._seed}:{label}".encode()).digest()
        derived = int.from_bytes(digest[:8], "big", signed=False)
        return SeededRNG(derived)

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("Cannot choose from an empty sequence.")
        return self._rng.choice(list(seq))

    def shuffled(self, items: Iterable[T]) -> list[T]:
        materialized = list(items)
        self._rng.shuffle(materialized)
        return materialized

    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def randint(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        k = max(0, min(k, len(seq)))
        if k == 0:
            return []
        return self._rng.sample(list(seq), k)
