"""Qt multimedia controller and the authoritative playback clock."""

from __future__ import annotations

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from .clock import MediaClock


class PlaybackController(QObject):
    position_changed = Signal(float)
    duration_changed = Signal(float)
    playing_changed = Signal(bool)
    generation_changed = Signal(int)
    ended = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.clock = MediaClock()
        self._duration_s = 0.0

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.errorOccurred.connect(self._on_error)

    def load(self, path: str) -> None:
        self.player.stop()
        generation = self.clock.reset()
        self.generation_changed.emit(generation)
        self.player.setSource(QUrl.fromLocalFile(path))

    def play(self) -> None:
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self.seek(0.0)
        self.player.play()

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        self.player.stop()
        generation = self.clock.seek(0.0)
        self.generation_changed.emit(generation)
        self.position_changed.emit(0.0)

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def seek(self, position_s: float) -> None:
        bounded = max(0.0, min(position_s, self._duration_s or position_s))
        generation = self.clock.seek(bounded)
        self.generation_changed.emit(generation)
        self.player.setPosition(round(bounded * 1000.0))
        self.position_changed.emit(bounded)

    def set_volume(self, normalized: float) -> None:
        self.audio_output.setVolume(max(0.0, min(1.0, normalized)))

    def shutdown(self) -> None:
        self.player.stop()
        self.clock.set_playing(False)

    def _on_position(self, milliseconds: int) -> None:
        seconds = milliseconds / 1000.0
        self.clock.update_position(seconds)
        self.position_changed.emit(seconds)

    def _on_duration(self, milliseconds: int) -> None:
        self._duration_s = max(0.0, milliseconds / 1000.0)
        self.duration_changed.emit(self._duration_s)

    def _on_state(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.clock.set_playing(playing)
        self.playing_changed.emit(playing)

    def _on_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.clock.set_playing(False)
            self.ended.emit()

    def _on_error(
        self, _error: QMediaPlayer.Error, error_string: str = ""
    ) -> None:
        self.error.emit(error_string or "Qt could not decode this media file.")
