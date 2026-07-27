"""Thread-safe media clock driven by QMediaPlayer timestamps."""

from __future__ import annotations

import threading
import time

from ..models import ClockSnapshot


class MediaClock:
    """A media-time clock with short monotonic interpolation between Qt updates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._position_s = 0.0
        self._playing = False
        self._generation = 0
        self._changed_at = time.monotonic()

    def update_position(self, position_s: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._position_s = max(0.0, position_s)
            self._changed_at = now

    def set_playing(self, playing: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if self._playing:
                self._position_s += max(0.0, now - self._changed_at)
            self._playing = playing
            self._changed_at = now

    def seek(self, position_s: float) -> int:
        with self._lock:
            self._position_s = max(0.0, position_s)
            self._changed_at = time.monotonic()
            self._generation += 1
            return self._generation

    def reset(self) -> int:
        return self.seek(0.0)

    def snapshot(self) -> ClockSnapshot:
        now = time.monotonic()
        with self._lock:
            position = self._position_s
            if self._playing:
                position += min(0.25, max(0.0, now - self._changed_at))
            return ClockSnapshot(
                position_s=position,
                playing=self._playing,
                generation=self._generation,
                monotonic_s=now,
            )
