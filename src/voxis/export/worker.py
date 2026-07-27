"""Cancelable export worker: analyze, simulate, render, encode, and mux."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import threading

from PySide6.QtCore import QThread, Signal

from ..analysis.feature_cache import (
    AnalysisCancelled,
    FeatureCache,
    build_feature_cache,
)
from ..config import VisualizationSettings
from ..models import MediaInfo
from ..presets import PRESETS
from ..visualization.offscreen import OffscreenRenderer
from ..visualization.points import PointManager
from .ffmpeg_encoder import ExportOptions, FFmpegEncoder


def _analysis_frame_due(frame_timestamp: float, render_timestamp: float) -> bool:
    return frame_timestamp <= render_timestamp + 1e-9


class ExportWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        media: MediaInfo,
        settings: VisualizationSettings,
        preset_name: str,
        options: ExportOptions,
        parent=None,
        feature_cache: FeatureCache | None = None,
    ) -> None:
        super().__init__(parent)
        self.media = media
        self.settings = settings.copy()
        self.preset_name = preset_name
        self.options = options.validated()
        self.cancel_event = threading.Event()
        self.encoder: FFmpegEncoder | None = None
        self.feature_cache = feature_cache

    def request_cancel(self) -> None:
        self.cancel_event.set()
        if self.encoder is not None:
            self.encoder.cancel()

    def run(self) -> None:
        output = Path(self.options.output_path)
        renderer: OffscreenRenderer | None = None
        try:
            self._validate_destination(output)
            cached = self.feature_cache
            cache_matches = (
                cached is not None
                and cached.source_path == self.media.path
                and cached.fft_size == self.settings.fft_size
                and cached.hop_size == self.settings.hop_size
            )
            if cache_matches:
                cache = cached
                self.progress.emit(20, "Using preview analysis")
            else:
                self.progress.emit(0, "Analyzing audio")
                cache = build_feature_cache(
                    self.media.path,
                    self.media.duration_s,
                    self.settings,
                    progress=lambda value: self.progress.emit(
                        round(value * 20.0), "Analyzing audio"
                    ),
                    cancelled=self.cancel_event,
                )
            if self.cancel_event.is_set():
                raise AnalysisCancelled()

            self.progress.emit(20, "Initializing offline GPU renderer")
            renderer = OffscreenRenderer(self.options.width, self.options.height)
            manager = PointManager(self.settings.random_seed)
            preset = PRESETS[self.preset_name]
            self.encoder = FFmpegEncoder(
                self.media.path, self.media.duration_s, self.options
            )
            total_frames = max(
                1, int(math.ceil(self.media.duration_s * self.options.fps))
            )
            feature_index = 0
            fixed_delta = 1.0 / self.options.fps
            for frame_index in range(total_frames):
                if self.cancel_event.is_set():
                    raise AnalysisCancelled()
                timestamp = frame_index * fixed_delta
                while (
                    feature_index < len(cache.frames)
                    and _analysis_frame_due(
                        cache.frames[feature_index].timestamp_s,
                        timestamp,
                    )
                ):
                    manager.ingest(
                        cache.frames[feature_index],
                        self.settings,
                        preset,
                        density_scale=1.0,
                    )
                    feature_index += 1
                manager.update_at(timestamp, self.settings)
                rgb = renderer.render(manager, self.settings, preset, timestamp)
                self.encoder.write(rgb)
                if frame_index % max(1, self.options.fps // 2) == 0:
                    percent = 20 + round((frame_index + 1) / total_frames * 78)
                    self.progress.emit(percent, "Rendering and encoding")

            self.progress.emit(99, "Finalizing MP4")
            self.encoder.finish()
            self.encoder = None
            if self.cancel_event.is_set():
                raise AnalysisCancelled()
            self.progress.emit(100, "Export complete")
            self.completed.emit(str(output))
        except AnalysisCancelled:
            self._remove_partial(output)
            self.cancelled.emit()
        except Exception as exc:
            if self.encoder is not None:
                self.encoder.cancel()
            self._remove_partial(output)
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
        finally:
            self.encoder = None
            if renderer is not None:
                renderer.close()

    def _validate_destination(self, output: Path) -> None:
        if output.suffix.lower() != ".mp4":
            raise ValueError("The export path must end with .mp4.")
        if not output.parent.is_dir():
            raise ValueError("The export directory does not exist.")
        if not os.access(output.parent, os.W_OK):
            raise PermissionError("The export directory is not writable.")
        if output.exists() and not output.is_file():
            raise ValueError("The export path is not a regular file.")
        estimate = (
            self.options.video_bitrate_kbps
            * 1000
            / 8
            * self.media.duration_s
            * 1.25
        )
        free = shutil.disk_usage(output.parent).free
        if free < estimate + 64 * 1024 * 1024:
            raise RuntimeError(
                "Insufficient free disk space for the requested export."
            )

    @staticmethod
    def _remove_partial(output: Path) -> None:
        try:
            if output.is_file():
                output.unlink()
        except OSError:
            pass
