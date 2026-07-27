"""Audio-only application shell, preview transport, and export orchestration."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeyEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..config import SettingsState, VisualizationSettings, save_settings
from ..export.worker import ExportWorker
from ..media_import import MediaProbeError, inspect_audio_source
from ..models import MediaInfo
from ..preview_controller import PreviewController
from ..presets import DEFAULT_PRESET, PRESETS
from ..visualization.renderer import VisualizationWidget
from .export_dialog import ExportDialog
from .settings_panel import SettingsPanel


class ProbeSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(str)


class ProbeTask(QRunnable):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.signals = ProbeSignals()

    def run(self) -> None:
        try:
            self.signals.succeeded.emit(inspect_audio_source(self.path))
        except MediaProbeError as exc:
            self.signals.failed.emit(str(exc))
        except Exception as exc:
            self.signals.failed.emit(f"Unexpected media inspection error: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, initial_settings: VisualizationSettings) -> None:
        super().__init__()
        self.setWindowTitle("Voxis")
        self.resize(1420, 900)
        self.setMinimumSize(940, 620)
        self.settings_state = SettingsState(initial_settings)
        self.preview = PreviewController(self.settings_state, self)
        self.synchronizer = self.preview.synchronizer
        self.playback = self.preview.playback
        self.analysis = self.preview.analysis
        self.thread_pool = QThreadPool.globalInstance()
        self._duration_s = 0.0
        self._scrubbing = False
        self._resume_after_scrub = False
        self._fullscreen_visualization = False
        self._previous_window_state = self.windowState()
        self._loaded_media: MediaInfo | None = None
        self._probe_task: ProbeTask | None = None
        self._export_worker: ExportWorker | None = None

        self._build_ui(initial_settings)
        self._connect_signals()
        self._install_actions()
        self._apply_style()
        self._set_media_controls_enabled(False)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(
            lambda: save_settings(self.settings_state.snapshot())
        )

    def _build_ui(self, settings: VisualizationSettings) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("topBar")
        top = QHBoxLayout(self.top_bar)
        top.setContentsMargins(12, 8, 12, 8)
        brand = QLabel("VOXIS")
        brand.setObjectName("brand")
        top.addWidget(brand)
        self.source_label = QLabel("NO AUDIO SOURCE")
        self.source_label.setObjectName("sourceLabel")
        top.addSpacing(18)
        top.addWidget(self.source_label, 1)
        self.import_button = QPushButton("Import file")
        self.import_button.setObjectName("primaryButton")
        top.addWidget(self.import_button)
        top.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.setCurrentText(
            settings.palette if settings.palette in PRESETS else DEFAULT_PRESET
        )
        top.addWidget(self.preset_combo)
        self.reset_camera_button = QPushButton("Reset camera")
        self.reset_visualization_button = QPushButton("Reset visualization")
        self.fullscreen_button = QPushButton("Full screen")
        top.addWidget(self.reset_camera_button)
        top.addWidget(self.reset_visualization_button)
        top.addWidget(self.fullscreen_button)
        root.addWidget(self.top_bar)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.visualization_panel = QFrame()
        self.visualization_panel.setObjectName("displayPanel")
        visual_layout = QVBoxLayout(self.visualization_panel)
        visual_layout.setContentsMargins(1, 1, 1, 1)
        header = QHBoxLayout()
        title = QLabel("  ABSTRACT AUDIO STRUCTURE")
        title.setObjectName("panelTitle")
        self.performance_label = QLabel("60 fps  ·  0 points")
        self.performance_label.setObjectName("performanceLabel")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.performance_label)
        visual_layout.addLayout(header)
        self.visualization = VisualizationWidget(
            self.playback.clock, self.synchronizer, self.settings_state
        )
        self.visualization.set_preset(self.preset_combo.currentText())
        visual_layout.addWidget(self.visualization, 1)
        self.settings_panel = SettingsPanel(settings)
        self.content_splitter.addWidget(self.visualization_panel)
        self.content_splitter.addWidget(self.settings_panel)
        self.content_splitter.setSizes([1080, 320])
        self.content_splitter.setStretchFactor(0, 1)
        root.addWidget(self.content_splitter, 1)

        self.transport = QFrame()
        self.transport.setObjectName("transport")
        controls = QHBoxLayout(self.transport)
        controls.setContentsMargins(12, 8, 12, 8)
        self.preview_button = QPushButton("Preview")
        self.preview_button.setObjectName("primaryButton")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        controls.addWidget(self.preview_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.stop_button)
        self.current_label = QLabel("00:00.000")
        self.total_label = QLabel("00:00.000")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        controls.addWidget(self.current_label)
        controls.addWidget(self.timeline, 1)
        controls.addWidget(self.total_label)
        controls.addSpacing(8)
        controls.addWidget(QLabel("VOL"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(100)
        controls.addWidget(self.volume)
        root.addWidget(self.transport)

        self.export_bar = QFrame()
        self.export_bar.setObjectName("transport")
        export_layout = QHBoxLayout(self.export_bar)
        export_layout.setContentsMargins(12, 7, 12, 7)
        self.export_button = QPushButton("Export MP4")
        self.export_button.setObjectName("primaryButton")
        self.cancel_export_button = QPushButton("Cancel export")
        self.cancel_export_button.hide()
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setFormat("Ready to export")
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.cancel_export_button)
        export_layout.addWidget(self.export_progress, 1)
        root.addWidget(self.export_bar)

        status = QStatusBar()
        status.setSizeGripEnabled(False)
        status.showMessage("Import a video or audio file to begin")
        author = QLabel("Made by Hrasvx")
        author.setObjectName("authorCredit")
        status.addPermanentWidget(author)
        self.setStatusBar(status)

    def _connect_signals(self) -> None:
        self.import_button.clicked.connect(self.open_media)
        self.preview_button.clicked.connect(self._preview_from_start)
        self.play_button.clicked.connect(self.playback.play)
        self.pause_button.clicked.connect(self.playback.pause)
        self.stop_button.clicked.connect(self.playback.stop)
        self.volume.valueChanged.connect(
            lambda value: self.playback.set_volume(value / 100.0)
        )
        self.playback.set_volume(self.volume.value() / 100.0)
        self.timeline.sliderPressed.connect(self._scrub_started)
        self.timeline.sliderMoved.connect(self._scrub_moved)
        self.timeline.sliderReleased.connect(self._scrub_finished)
        self.playback.position_changed.connect(self._position_changed)
        self.playback.duration_changed.connect(self._duration_changed)
        self.playback.playing_changed.connect(self._playing_changed)
        self.playback.generation_changed.connect(self._generation_changed)
        self.playback.ended.connect(
            lambda: self.statusBar().showMessage(
                "End of source — Preview replays deterministically"
            )
        )
        self.playback.error.connect(self._playback_error)
        self.analysis.status.connect(self.statusBar().showMessage)
        self.analysis.error.connect(self.statusBar().showMessage)
        self.analysis.progress.connect(
            lambda value: self.statusBar().showMessage(
                f"Analyzing audio… {value}%"
            )
        )
        self.analysis.ready.connect(self._analysis_complete)
        self.analysis.history_ready.connect(
            self.visualization.finalize_seek_history
        )
        self.settings_panel.setting_changed.connect(self._setting_changed)
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.reset_camera_button.clicked.connect(self.visualization.reset_camera)
        self.reset_visualization_button.clicked.connect(
            lambda: self.visualization.clear_timeline()
        )
        self.fullscreen_button.clicked.connect(self.toggle_visualization_fullscreen)
        self.visualization.fullscreen_requested.connect(
            self.toggle_visualization_fullscreen
        )
        self.visualization.renderer_error.connect(self._renderer_error)
        self.visualization.performance_changed.connect(
            lambda fps, points: self.performance_label.setText(
                f"{fps:0.0f} fps  ·  {points} points"
            )
        )
        self.export_button.clicked.connect(self.start_export)
        self.cancel_export_button.clicked.connect(self.cancel_export)

    def _install_actions(self) -> None:
        import_action = QAction("Import file", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self.open_media)
        self.addAction(import_action)
        play_action = QAction("Play/pause", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self.playback.toggle)
        self.addAction(play_action)
        fullscreen_action = QAction("Full-screen preview", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_visualization_fullscreen)
        self.addAction(fullscreen_action)

    def open_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import audio source",
            str(Path.home()),
            (
                "Supported media (*.mp4 *.mov *.mkv *.avi *.webm "
                "*.mp3 *.wav *.flac *.aac *.ogg *.m4a);;"
                "Video (*.mp4 *.mov *.mkv *.avi *.webm);;"
                "Audio (*.mp3 *.wav *.flac *.aac *.ogg *.m4a);;All files (*)"
            ),
        )
        if not path:
            return
        self.playback.pause()
        self.statusBar().showMessage("Inspecting audio source…")
        self.import_button.setEnabled(False)
        task = ProbeTask(path)
        self._probe_task = task
        task.signals.succeeded.connect(self._media_ready)
        task.signals.failed.connect(self._media_failed)
        self.thread_pool.start(task)

    def _media_ready(self, media: MediaInfo) -> None:
        self._probe_task = None
        self.import_button.setEnabled(True)
        self._loaded_media = media
        self.source_label.setText(
            f"{Path(media.path).name}  /  {media.audio_codec.upper()}  /  "
            f"{format_time(media.duration_s)}"
        )
        self._duration_s = media.duration_s
        self.preview.load(media)
        self._set_media_controls_enabled(False)
        source_kind = "video audio track" if media.has_video else "audio file"
        self.statusBar().showMessage(
            f"Ready — {source_kind}, {media.audio_codec}; image frames are ignored"
        )

    def _media_failed(self, message: str) -> None:
        self._probe_task = None
        self.import_button.setEnabled(True)
        self.statusBar().showMessage("Could not import audio source")
        QMessageBox.critical(self, "Unsupported or damaged media", message)

    def _preview_from_start(self) -> None:
        self.playback.seek(0.0)
        self.playback.play()

    def _scrub_started(self) -> None:
        self._scrubbing = True
        self._resume_after_scrub = (
            self.playback.player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )
        self.playback.pause()

    def _scrub_moved(self, milliseconds: int) -> None:
        self.current_label.setText(format_time(milliseconds / 1000.0))

    def _scrub_finished(self) -> None:
        self.playback.seek(self.timeline.value() / 1000.0)
        self._scrubbing = False
        if self._resume_after_scrub:
            self.playback.play()

    def _position_changed(self, seconds: float) -> None:
        if not self._scrubbing:
            self.timeline.setValue(round(seconds * 1000.0))
            self.current_label.setText(format_time(seconds))

    def _duration_changed(self, seconds: float) -> None:
        self._duration_s = seconds or self._duration_s
        self.timeline.setRange(0, max(0, round(self._duration_s * 1000.0)))
        self.total_label.setText(format_time(self._duration_s))

    def _playing_changed(self, playing: bool) -> None:
        self.play_button.setEnabled(not playing and self._loaded_media is not None)
        self.pause_button.setEnabled(playing)
        self.analysis.wake()

    def _generation_changed(self, generation: int) -> None:
        self.synchronizer.reset(generation)
        self.visualization.clear_timeline(generation)
        self.analysis.wake()

    def _setting_changed(self, name: str, value: object) -> None:
        updated = self.settings_state.update(name, value)
        self.settings_panel.set_values(updated)
        if name == "performance_limit_fps":
            self.visualization.refresh_timer()
        if name == "random_seed":
            self.visualization.clear_timeline()
        if name == "palette":
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(str(value))
            self.preset_combo.blockSignals(False)
            self.visualization.set_preset(str(value))
        if name in {
            "audio_sensitivity",
            "fft_size",
            "hop_size",
            "audio_smoothing",
            "frequency_min",
            "frequency_max",
            "min_peaks_per_frame",
            "max_peaks_per_frame",
            "peak_prominence_db",
            "spectral_noise_floor_db",
            "silence_threshold_db",
            "phrase_detection_sensitivity",
            "analysis_lookahead_ms",
        }:
            self.playback.pause()
            self._set_media_controls_enabled(False)
            self.visualization.clear_timeline()
            self.analysis.rebuild()
        self._save_timer.start(500)

    def _preset_changed(self, name: str) -> None:
        preset = PRESETS[name]
        self.settings_state.update("palette", name)
        for setting_name, value in preset.overrides.items():
            self.settings_state.update(setting_name, value)
        self.settings_panel.set_values(self.settings_state.snapshot())
        self.visualization.set_preset(name)
        self._save_timer.start(500)

    def start_export(self) -> None:
        if self._loaded_media is None or self._export_worker is not None:
            return
        dialog = ExportDialog(self._loaded_media.path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        worker = ExportWorker(
            self._loaded_media,
            self.settings_state.snapshot(),
            self.preset_combo.currentText(),
            options,
            self,
            feature_cache=self.analysis.cache,
        )
        self._export_worker = worker
        worker.progress.connect(self._export_progress_changed)
        worker.completed.connect(self._export_completed)
        worker.failed.connect(self._export_failed)
        worker.cancelled.connect(self._export_cancelled)
        worker.finished.connect(self._export_thread_finished)
        self.export_button.setEnabled(False)
        self.cancel_export_button.show()
        self.import_button.setEnabled(False)
        self.export_progress.setValue(0)
        worker.start()

    def cancel_export(self) -> None:
        if self._export_worker is not None:
            self.cancel_export_button.setEnabled(False)
            self.export_progress.setFormat("Cancelling…")
            self._export_worker.request_cancel()

    def _export_progress_changed(self, value: int, stage: str) -> None:
        self.export_progress.setValue(value)
        self.export_progress.setFormat(f"{stage} — {value}%")

    def _export_completed(self, path: str) -> None:
        self.export_progress.setValue(100)
        self.export_progress.setFormat("Export complete")
        self.statusBar().showMessage(f"Exported visualization: {path}")
        QMessageBox.information(
            self, "Export complete", f"Visualization exported to:\n{path}"
        )

    def _export_failed(self, message: str) -> None:
        self.export_progress.setFormat("Export failed")
        self.statusBar().showMessage(f"Export failed: {message}")
        QMessageBox.critical(self, "Export failed", message)

    def _export_cancelled(self) -> None:
        self.export_progress.setValue(0)
        self.export_progress.setFormat("Export cancelled")
        self.statusBar().showMessage("Export cancelled; partial file removed")

    def _export_thread_finished(self) -> None:
        if self._export_worker is not None:
            self._export_worker.deleteLater()
        self._export_worker = None
        self.export_button.setEnabled(self._loaded_media is not None)
        self.cancel_export_button.setEnabled(True)
        self.cancel_export_button.hide()
        self.import_button.setEnabled(True)

    def _set_media_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.preview_button,
            self.play_button,
            self.pause_button,
            self.stop_button,
            self.timeline,
            self.export_button,
            self.reset_visualization_button,
        ):
            widget.setEnabled(enabled)
        if enabled:
            self.pause_button.setEnabled(False)

    def _analysis_complete(self) -> None:
        if self._loaded_media is None:
            return
        self._set_media_controls_enabled(True)
        self.statusBar().showMessage(
            "Analysis ready — playback, seeking, and deterministic export available"
        )

    def toggle_visualization_fullscreen(self) -> None:
        if not self._fullscreen_visualization:
            self._previous_window_state = self.windowState()
            self.top_bar.hide()
            self.settings_panel.hide()
            self.transport.hide()
            self.export_bar.hide()
            self.statusBar().hide()
            self._fullscreen_visualization = True
            self.showFullScreen()
        else:
            self.top_bar.show()
            self.settings_panel.show()
            self.transport.show()
            self.export_bar.show()
            self.statusBar().show()
            self._fullscreen_visualization = False
            self.setWindowState(self._previous_window_state)
            if self._previous_window_state == Qt.WindowState.WindowNoState:
                self.showNormal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._fullscreen_visualization:
            self.toggle_visualization_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def _playback_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.warning(
            self,
            "Audio playback error",
            message + "\n\nThe source audio codec may not be available to Qt.",
        )

    def _renderer_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QMessageBox.critical(self, "3D renderer unavailable", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.request_cancel()
            self._export_worker.wait(5000)
        self._save_timer.stop()
        save_settings(self.settings_state.snapshot())
        self.preview.shutdown()
        super().closeEvent(event)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d0f11; color: #c8cbd0;
                font-family: "Inter", "Noto Sans", sans-serif; font-size: 12px;
            }
            QFrame#topBar, QFrame#transport, QFrame#settingsPanel {
                background: #15181c; border: 1px solid #252a30; border-radius: 6px;
            }
            QFrame#displayPanel {
                background: #08090a; border: 1px solid #292d33; border-radius: 5px;
            }
            QLabel#brand {
                color: #e5b65a; font-weight: 700; letter-spacing: 2px; font-size: 13px;
            }
            QLabel#sourceLabel { color: #747b84; }
            QLabel#sectionTitle, QLabel#panelTitle, QLabel#parameterHeading {
                color: #8d949d; font-size: 10px; font-weight: 700;
                letter-spacing: 1px; padding: 5px;
            }
            QLabel#parameterHeading {
                color: #c19043; border-bottom: 1px solid #292d33; margin-top: 8px;
            }
            QLabel#performanceLabel { color: #656c75; font-size: 10px; padding-right: 8px; }
            QLabel#authorCredit {
                color: #7f858d; font-size: 10px; letter-spacing: 1px;
                padding: 0 8px;
            }
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #20242a; border: 1px solid #333943;
                border-radius: 4px; padding: 6px 9px; color: #d5d8dc;
            }
            QPushButton:hover, QComboBox:hover { border-color: #676f7a; background: #282d34; }
            QPushButton#primaryButton {
                background: #b77a2e; color: #0b0c0d; border-color: #d49a4d; font-weight: 700;
            }
            QPushButton:disabled { color: #555b63; background: #171a1e; border-color: #252a30; }
            QSlider::groove:horizontal { height: 3px; background: #343941; border-radius: 1px; }
            QSlider::sub-page:horizontal { background: #b77a2e; }
            QSlider::handle:horizontal {
                background: #e3b560; width: 11px; margin: -5px 0; border-radius: 5px;
            }
            QProgressBar {
                background: #0d0f11; border: 1px solid #30353d;
                border-radius: 4px; text-align: center; color: #aeb3ba;
            }
            QProgressBar::chunk { background: #a96f29; border-radius: 3px; }
            QScrollArea { background: transparent; }
            QStatusBar { background: #0d0f11; color: #737a83; }
            QSplitter::handle { background: #0d0f11; width: 8px; }
            """
        )


def format_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"
