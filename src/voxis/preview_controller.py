"""Lifecycle boundary for synchronized audio preview and live analysis."""

from __future__ import annotations

from PySide6.QtCore import QObject

from .analysis.pipeline import AnalysisPipeline
from .config import SettingsState
from .models import MediaInfo
from .playback.controller import PlaybackController
from .playback.synchronizer import TimelineSynchronizer


class PreviewController(QObject):
    def __init__(self, settings: SettingsState, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.synchronizer = TimelineSynchronizer(
            capacity=20000, stale_after_s=3600.0
        )
        self.playback = PlaybackController(self)
        self.analysis = AnalysisPipeline(
            self.playback.clock, self.synchronizer, settings, self
        )

    def load(self, media: MediaInfo) -> None:
        self.playback.load(media.path)
        self.analysis.load(media)

    def shutdown(self) -> None:
        self.analysis.shutdown()
        self.playback.shutdown()
