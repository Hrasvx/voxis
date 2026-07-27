"""Bounded handoff from timestamped analysis to the render clock."""

from __future__ import annotations

from collections import deque
import threading

from ..models import AnalysisFrame


class TimelineSynchronizer:
    """Stores a short ordered analysis buffer and never permits backlog growth."""

    def __init__(self, capacity: int = 24, stale_after_s: float = 0.25) -> None:
        self.capacity = max(2, capacity)
        self.stale_after_s = stale_after_s
        self._frames: deque[AnalysisFrame] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def reset(self, generation: int) -> None:
        with self._lock:
            self._frames.clear()
            self._generation = generation

    def push(self, frame: AnalysisFrame) -> bool:
        with self._lock:
            if frame.generation != self._generation:
                return False
            if self._frames and frame.timestamp_s < self._frames[-1].timestamp_s:
                return False
            self._frames.append(frame)
            return True

    def consume_ready(
        self,
        media_time_s: float,
        generation: int,
        future_tolerance_s: float = 0.012,
    ) -> list[AnalysisFrame]:
        """Return frames due at ``media_time_s`` and discard excessively late ones."""
        ready: list[AnalysisFrame] = []
        with self._lock:
            if generation != self._generation:
                return ready
            cutoff = media_time_s + future_tolerance_s
            while self._frames and self._frames[0].timestamp_s <= cutoff:
                frame = self._frames.popleft()
                if frame.timestamp_s >= media_time_s - self.stale_after_s:
                    ready.append(frame)
            while (
                len(self._frames) > 2
                and self._frames[0].timestamp_s < media_time_s - self.stale_after_s
            ):
                self._frames.popleft()
        return ready

    def buffered_count(self) -> int:
        with self._lock:
            return len(self._frames)
