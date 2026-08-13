"""Explicit, bounded lifetime for reused computations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ReuseStats:
    hits: int
    misses: int
    evictions: int
    entries: int


class ReuseScope(Generic[T]):
    """Cache values within one algorithmic scope, with a fixed entry bound."""

    def __init__(self, max_entries: int):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._values: OrderedDict[Hashable, T] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        try:
            value = self._values.pop(key)
        except KeyError:
            self._misses += 1
            value = compute()
            if len(self._values) == self._max_entries:
                self._values.popitem(last=False)
                self._evictions += 1
        else:
            self._hits += 1
        self._values[key] = value
        return value

    def discard(self, key: Hashable) -> None:
        self._values.pop(key, None)

    def clear(self) -> None:
        self._values.clear()

    def stats(self) -> ReuseStats:
        return ReuseStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            entries=len(self._values),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.clear()
        return False
