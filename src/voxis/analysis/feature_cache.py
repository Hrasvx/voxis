"""Deterministic timestamped feature extraction for offline export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from ..config import VisualizationSettings
from ..models import AnalysisFrame
from ..playback.audio_decode import AudioDecoder
from .audio_analyzer import AudioAnalyzer, finalize_context_embedding


ProgressCallback = Callable[[float], None]


class AnalysisCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class FeatureCache:
    source_path: str
    duration_s: float
    fft_size: int
    frames: tuple[AnalysisFrame, ...]
    hop_size: int = 0

    def save(self, path: Path) -> None:
        """Persist a compact reproducible cache for diagnostics or reuse."""
        if not self.frames:
            raise ValueError("Cannot save an empty feature cache.")
        np.savez_compressed(
            path,
            source_path=np.asarray(self.source_path),
            duration_s=np.asarray(self.duration_s),
            fft_size=np.asarray(self.fft_size),
            hop_size=np.asarray(self.hop_size),
            timestamps=np.asarray([frame.timestamp_s for frame in self.frames]),
            durations=np.asarray([frame.duration_s for frame in self.frames]),
            spectra=np.asarray([frame.spectrum for frame in self.frames], dtype=np.float16),
            frequencies=self.frames[0].frequencies,
            bands=np.asarray([frame.band_energy for frame in self.frames], dtype=np.float16),
            scalars=np.asarray(
                [
                    (
                        frame.rms_db,
                        frame.peak_db,
                        frame.dominant_hz,
                        frame.stereo_balance,
                        frame.phase_correlation,
                        frame.spectral_centroid_hz,
                        frame.spectral_bandwidth_hz,
                        frame.spectral_rolloff_hz,
                        frame.onset_strength,
                        frame.spectral_contrast,
                    )
                    for frame in self.frames
                ],
                dtype=np.float32,
            ),
        )


def build_feature_cache(
    source_path: str,
    duration_s: float,
    settings: VisualizationSettings,
    progress: ProgressCallback | None = None,
    cancelled: Event | None = None,
) -> FeatureCache:
    decoder = AudioDecoder(source_path)
    analyzer = AudioAnalyzer(
        settings.fft_size,
        settings.audio_smoothing,
        settings.frequency_min,
        settings.frequency_max,
        minimum_peaks=settings.min_peaks_per_frame,
        maximum_peaks=settings.max_peaks_per_frame,
        peak_prominence_db=settings.peak_prominence_db,
        noise_floor_db=settings.spectral_noise_floor_db,
        silence_threshold_db=settings.silence_threshold_db,
        sensitivity=settings.audio_sensitivity,
        phrase_sensitivity=settings.phrase_detection_sensitivity,
        hop_size=settings.hop_size,
    )
    frames: list[AnalysisFrame] = []
    try:
        for chunk in decoder.chunks(0.0, settings.fft_size, settings.hop_size):
            if cancelled is not None and cancelled.is_set():
                raise AnalysisCancelled("Feature analysis cancelled.")
            frame = analyzer.analyze(chunk, generation=0)
            if frame.timestamp_s > duration_s + frame.duration_s:
                break
            frames.append(frame)
            if progress is not None and duration_s > 0.0:
                progress(min(1.0, frame.timestamp_s / duration_s))
    finally:
        decoder.close()
    if not frames:
        raise RuntimeError("No decodable audio samples were found.")
    finalize_context_embedding(frames)
    if progress is not None:
        progress(1.0)
    return FeatureCache(
        source_path,
        duration_s,
        settings.fft_size,
        tuple(frames),
        settings.hop_size,
    )
