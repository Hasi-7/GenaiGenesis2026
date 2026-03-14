from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class BufferStats:
    """Aggregated buffer counters."""

    dropped_count: int = 0


class LatestValueBuffer(Generic[T]):
    """Thread-safe bounded buffer that always returns the newest item."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: deque[T] = deque()
        self._lock = Lock()
        self._dropped_count = 0

    def push(self, item: T) -> None:
        with self._lock:
            if len(self._items) == self._capacity:
                self._items.popleft()
                self._dropped_count += 1
            self._items.append(item)

    def get_latest(self) -> T | None:
        with self._lock:
            if not self._items:
                return None
            latest = self._items[-1]
            self._items.clear()
            return latest

    def stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(dropped_count=self._dropped_count)
