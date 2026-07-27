"""Small dependency-light data models shared by playback and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True, frozen=True)
class MediaInfo:
    path: str
    duration_s: float
    width: int
    height: int
    frame_rate: float
    has_video: bool
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""


@dataclass(slots=True, frozen=True)
class SpectralPeak:
    """One meaningful local maximum in an STFT frame."""

    frequency_hz: float
    amplitude: float
    magnitude_db: float
    prominence_db: float
    snr_db: float
    bandwidth_hz: float
    phase: float
    stereo_balance: float = 0.0
    harmonicity: float = 0.0


@dataclass(slots=True)
class AnalysisFrame:
    """One timestamped short-time audio analysis result.

    ``timestamp_s`` is the center of the analyzed block on the media timeline.
    The renderer must compare it with the media player's clock, not wall time.
    """

    timestamp_s: float
    duration_s: float
    spectrum: np.ndarray
    frequencies: np.ndarray
    band_energy: np.ndarray
    rms_db: float
    peak_db: float
    dominant_hz: float
    stereo_balance: float
    phase_correlation: float
    spectral_centroid_hz: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    onset_strength: float = 0.0
    generation: int = 0
    peaks: tuple[SpectralPeak, ...] = ()
    context_features: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    context_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    spectral_flatness: float = 0.0
    spectral_contrast: float = 0.0
    harmonicity: float = 0.0
    phrase_centroid_hz: float = 0.0
    phrase_timbre: float = -1.0
    phrase_id: int = 0
    phrase_boundary: bool = False


@dataclass(slots=True, frozen=True)
class ClockSnapshot:
    position_s: float
    playing: bool
    generation: int
    monotonic_s: float


@dataclass(slots=True)
class AudioChunk:
    samples: np.ndarray
    timestamp_s: float
    sample_rate: int
