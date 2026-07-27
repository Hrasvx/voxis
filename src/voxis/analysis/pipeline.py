"""Pre-analysis worker and timestamped cache feeder for synchronized preview."""

from __future__ import annotations

import bisect
import math
import threading

from PySide6.QtCore import QObject, Signal

from ..config import SettingsState
from ..models import MediaInfo
from ..playback.clock import MediaClock
from ..playback.synchronizer import TimelineSynchronizer
from .feature_cache import AnalysisCancelled, FeatureCache, build_feature_cache


class AnalysisPipeline(QObject):
    """Build one stable global feature space, then feed it by media timestamp."""

    status = Signal(str)
    error = Signal(str)
    progress = Signal(int)
    ready = Signal()
    history_ready = Signal(float, int)

    def __init__(
        self,
        clock: MediaClock,
        synchronizer: TimelineSynchronizer,
        settings: SettingsState,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.clock = clock
        self.synchronizer = synchronizer
        self.settings = settings
        self._condition = threading.Condition()
        self._media: MediaInfo | None = None
        self._running = True
        self._serial = 0
        self._cancel = threading.Event()
        self._cache: FeatureCache | None = None
        self._thread = threading.Thread(
            target=self._run, name="audio-preanalysis", daemon=True
        )
        self._thread.start()

    @property
    def cache(self) -> FeatureCache | None:
        with self._condition:
            return self._cache

    def load(self, media: MediaInfo) -> None:
        with self._condition:
            self._cancel.set()
            self._cancel = threading.Event()
            self._media = media
            self._cache = None
            self._serial += 1
            self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def rebuild(self) -> None:
        with self._condition:
            if self._media is None:
                return
            self._cancel.set()
            self._cancel = threading.Event()
            self._cache = None
            self._serial += 1
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._running = False
            self._cancel.set()
            self._condition.notify_all()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        active_serial = -1
        active_generation = -1
        cursor = 0
        reconstruction_pending = False
        cache: FeatureCache | None = None
        timestamps: list[float] = []
        while self._running:
            with self._condition:
                if self._media is None:
                    self._condition.wait(timeout=0.2)
                    continue
                media = self._media
                serial = self._serial
                cancel = self._cancel

            if serial != active_serial:
                active_serial = serial
                active_generation = -1
                cache = None
                if not media.has_audio:
                    self.status.emit("No audio stream — visualization is idle.")
                    self._wait(0.2)
                    continue
                try:
                    settings = self.settings.snapshot()
                    last_percent = -1

                    def on_progress(value: float) -> None:
                        nonlocal last_percent
                        percent = round(value * 100.0)
                        if percent != last_percent:
                            last_percent = percent
                            self.progress.emit(percent)
                            self.status.emit(f"Analyzing audio… {percent}%")

                    cache = build_feature_cache(
                        media.path,
                        media.duration_s,
                        settings,
                        progress=on_progress,
                        cancelled=cancel,
                    )
                except AnalysisCancelled:
                    continue
                except Exception as exc:
                    if serial == self._serial:
                        self.error.emit(f"Audio analysis stopped: {exc}")
                    self._wait(0.25)
                    continue
                with self._condition:
                    if serial != self._serial:
                        continue
                    self._cache = cache
                self.status.emit(
                    f"Analysis ready — {len(cache.frames):,} STFT frames"
                )
                timestamps = [frame.timestamp_s for frame in cache.frames]
                self.ready.emit()

            if cache is None:
                self._wait(0.05)
                continue
            snapshot = self.clock.snapshot()
            settings = self.settings.snapshot()
            if snapshot.generation != active_generation:
                active_generation = snapshot.generation
                self.synchronizer.reset(active_generation)
                history = (
                    snapshot.position_s
                    if settings.persistent_history
                    else settings.historical_duration
                )
                start = max(0.0, snapshot.position_s - history)
                if not settings.persistent_history:
                    start = min(
                        start,
                        max(
                            0.0,
                            math.floor(snapshot.position_s / 2.0) * 2.0
                            - settings.historical_duration,
                        ),
                    )
                    start = max(
                        0.0, start - settings.node_interval_s * 1.25
                    )
                cursor = bisect.bisect_left(timestamps, start - 1e-6)
                reconstruction_pending = True

            lookahead = settings.analysis_lookahead_ms / 1000.0
            target = snapshot.position_s + lookahead
            pushed = 0
            while cursor < len(cache.frames) and cache.frames[cursor].timestamp_s <= target:
                original = cache.frames[cursor]
                original.generation = active_generation
                if not self.synchronizer.push(original):
                    break
                cursor += 1
                pushed += 1
                if pushed >= 512:
                    break
            if reconstruction_pending and (
                cursor >= len(cache.frames)
                or cache.frames[cursor].timestamp_s > target
            ):
                reconstruction_pending = False
                self.history_ready.emit(snapshot.position_s, active_generation)
            self._wait(0.004 if pushed else 0.018)

    def _wait(self, timeout: float) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)
