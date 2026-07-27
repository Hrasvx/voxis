"""Overlapped STFT analysis, adaptive peak detection, and phrase features."""

from __future__ import annotations

import math

import numpy as np

from ..models import AnalysisFrame, AudioChunk, SpectralPeak


class AudioAnalyzer:
    """Dependency-light spectral analyzer used identically by preview and export."""

    def __init__(
        self,
        fft_size: int,
        smoothing: float,
        frequency_min: float,
        frequency_max: float,
        band_count: int = 24,
        *,
        minimum_peaks: int = 6,
        maximum_peaks: int = 28,
        peak_prominence_db: float = 5.0,
        noise_floor_db: float = 7.0,
        silence_threshold_db: float = -62.0,
        sensitivity: float = 1.0,
        phrase_sensitivity: float = 1.0,
        hop_size: int | None = None,
    ) -> None:
        self.fft_size = fft_size
        self.hop_size = hop_size or fft_size
        self.smoothing = smoothing
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.band_count = band_count
        self.minimum_peaks = max(0, minimum_peaks)
        self.maximum_peaks = max(self.minimum_peaks, min(40, maximum_peaks))
        self.peak_prominence_db = peak_prominence_db
        self.noise_floor_db = noise_floor_db
        self.silence_threshold_db = silence_threshold_db
        self.sensitivity = float(np.clip(sensitivity, 0.05, 8.0))
        self.phrase_sensitivity = phrase_sensitivity
        self._window = np.hanning(fft_size).astype(np.float32)
        indices = np.arange(band_count, dtype=np.float64) + 0.5
        self._dct_basis = (
            np.cos(
                np.pi
                / band_count
                * np.outer(np.arange(13, dtype=np.float64), indices)
            )
            / max(1, band_count)
        )
        self._smoothed: np.ndarray | None = None
        self._previous_magnitude: np.ndarray | None = None
        self._previous_mfcc: np.ndarray | None = None
        self._previous_context: np.ndarray | None = None
        self._phrase_id = 0
        self._silent_frames = 0
        self._was_silent = True
        self._has_phrase = False
        self._last_boundary_s = -10.0

    def analyze(self, chunk: AudioChunk, generation: int) -> AnalysisFrame:
        stereo = _fit_samples(chunk.samples, self.fft_size)
        mono = stereo.mean(axis=0)
        windowed = mono * self._window
        complex_spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(complex_spectrum)
        magnitude /= max(1.0, self._window.sum() * 0.5)
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-9))
        normalized = np.clip((db + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)
        if self._smoothed is None or self._smoothed.shape != normalized.shape:
            self._smoothed = normalized.copy()
        else:
            attack = min(self.smoothing, 0.42)
            coefficient = np.where(normalized > self._smoothed, attack, self.smoothing)
            self._smoothed = (
                coefficient * self._smoothed + (1.0 - coefficient) * normalized
            ).astype(np.float32)

        frequencies = np.fft.rfftfreq(self.fft_size, 1.0 / chunk.sample_rate)
        maximum = min(self.frequency_max, chunk.sample_rate * 0.5)
        mask = (frequencies >= self.frequency_min) & (frequencies <= maximum)
        audible_magnitude = magnitude[mask]
        audible_frequencies = frequencies[mask]
        dominant = (
            float(audible_frequencies[np.argmax(audible_magnitude)])
            if audible_magnitude.size
            else 0.0
        )
        bands = _band_energy(
            magnitude,
            frequencies,
            self.frequency_min,
            maximum,
            self.band_count,
        )
        left_rms = float(np.sqrt(np.mean(stereo[0] ** 2) + 1e-12))
        right_rms = float(np.sqrt(np.mean(stereo[1] ** 2) + 1e-12))
        mono_rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
        peak = float(np.max(np.abs(stereo)) + 1e-12)
        rms_db = 20.0 * np.log10(max(mono_rms, 1e-8))
        balance = (right_rms - left_rms) / (right_rms + left_rms + 1e-9)
        correlation = float(
            np.sum(stereo[0] * stereo[1])
            / (
                np.sqrt(np.sum(stereo[0] ** 2) * np.sum(stereo[1] ** 2))
                + 1e-9
            )
        )
        centroid, bandwidth, rolloff = _spectral_shape(
            audible_magnitude, audible_frequencies
        )
        onset = _onset_strength(magnitude, self._previous_magnitude)
        self._previous_magnitude = magnitude.copy()
        flatness = _spectral_flatness(audible_magnitude)
        harmonicity = _harmonicity(audible_magnitude, audible_frequencies, dominant)
        contrast = _spectral_contrast(
            audible_magnitude, audible_frequencies, self.frequency_min, maximum
        )

        left_fft = np.fft.rfft(stereo[0] * self._window)
        right_fft = np.fft.rfft(stereo[1] * self._window)
        threshold_scale = 1.0 / math.sqrt(self.sensitivity)
        effective_silence_threshold = (
            self.silence_threshold_db - 6.0 * math.log2(self.sensitivity)
        )
        peaks = _detect_spectral_peaks(
            magnitude,
            db,
            np.angle(complex_spectrum),
            np.abs(left_fft),
            np.abs(right_fft),
            frequencies,
            self.frequency_min,
            maximum,
            rms_db,
            onset,
            flatness,
            self.minimum_peaks,
            self.maximum_peaks,
            max(0.5, self.peak_prominence_db * threshold_scale),
            max(0.0, self.noise_floor_db * threshold_scale),
            effective_silence_threshold,
            dominant,
        )
        compact_frequencies, compact_spectrum = _compact_spectrum(
            frequencies, self._smoothed, self.frequency_min, maximum
        )
        context = self._context_features(
            bands,
            contrast,
            centroid,
            bandwidth,
            rolloff,
            rms_db,
            onset,
            flatness,
            harmonicity,
            dominant,
            balance,
            correlation,
            maximum,
        )
        timestamp = chunk.timestamp_s + self.fft_size / chunk.sample_rate * 0.5
        phrase_id, boundary = self._update_phrase(
            timestamp, rms_db, onset, context, bool(peaks)
        )
        duration = self.hop_size / chunk.sample_rate
        return AnalysisFrame(
            timestamp_s=timestamp,
            duration_s=duration,
            spectrum=compact_spectrum,
            frequencies=compact_frequencies,
            band_energy=bands,
            rms_db=rms_db,
            peak_db=20.0 * np.log10(max(peak, 1e-8)),
            dominant_hz=dominant,
            stereo_balance=float(np.clip(balance, -1.0, 1.0)),
            phase_correlation=float(np.clip(correlation, -1.0, 1.0)),
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            spectral_rolloff_hz=rolloff,
            onset_strength=onset,
            generation=generation,
            peaks=peaks,
            context_features=context,
            spectral_flatness=flatness,
            spectral_contrast=float(np.mean(contrast)) if contrast.size else 0.0,
            harmonicity=harmonicity,
            phrase_id=phrase_id,
            phrase_boundary=boundary,
        )

    def _context_features(
        self,
        bands: np.ndarray,
        contrast: np.ndarray,
        centroid: float,
        bandwidth: float,
        rolloff: float,
        rms_db: float,
        onset: float,
        flatness: float,
        harmonicity: float,
        dominant: float,
        balance: float,
        correlation: float,
        maximum_frequency: float,
    ) -> np.ndarray:
        log_bands = np.log1p(np.maximum(bands, 0.0) * 12.0)
        mfcc = (self._dct_basis @ log_bands).astype(np.float32)
        if self._previous_mfcc is None:
            delta = np.zeros_like(mfcc)
        else:
            delta = mfcc - self._previous_mfcc
        self._previous_mfcc = mfcc.copy()
        scale = max(maximum_frequency, 1.0)
        scalars = np.asarray(
            (
                centroid / scale,
                bandwidth / scale,
                rolloff / scale,
                np.clip((rms_db + 80.0) / 80.0, 0.0, 1.0),
                onset,
                flatness,
                harmonicity,
                math.log1p(max(dominant, 0.0)) / math.log1p(scale),
                balance,
                correlation,
            ),
            dtype=np.float32,
        )
        return np.concatenate((mfcc, delta, contrast, scalars)).astype(np.float32)

    def _update_phrase(
        self,
        timestamp: float,
        rms_db: float,
        onset: float,
        context: np.ndarray,
        has_peaks: bool,
    ) -> tuple[int, bool]:
        silent = rms_db < self.silence_threshold_db or not has_peaks
        previous_silence_frames = self._silent_frames
        if silent:
            self._silent_frames += 1
        else:
            self._silent_frames = 0
        change = (
            float(np.linalg.norm(context - self._previous_context))
            / math.sqrt(max(1, context.size))
            if self._previous_context is not None
            else 0.0
        )
        enough_gap = timestamp - self._last_boundary_s > 0.12
        boundary = False
        if not silent and enough_gap:
            silence_gap_s = previous_silence_frames * self.hop_size / 48000.0
            boundary = (not self._has_phrase) or (
                self._was_silent and silence_gap_s >= 0.10
            ) or (
                onset > 0.32 / self.phrase_sensitivity
                and change > 0.07 / self.phrase_sensitivity
            )
        if boundary:
            if self._has_phrase:
                self._phrase_id += 1
            self._last_boundary_s = timestamp
            self._has_phrase = True
        self._was_silent = silent
        self._previous_context = context.copy()
        return self._phrase_id, boundary


def finalize_context_embedding(frames: list[AnalysisFrame]) -> None:
    """Apply deterministic standardized PCA in place."""
    _finalize_phrase_descriptors(frames)
    usable = [frame for frame in frames if frame.context_features.size]
    if not usable:
        return
    matrix = np.asarray([frame.context_features for frame in usable], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    standardized = (matrix - mean) / np.where(scale < 1e-7, 1.0, scale)
    if len(usable) >= 3:
        _, _, components = np.linalg.svd(standardized, full_matrices=False)
        basis = components[:3].T
        for column in range(basis.shape[1]):
            pivot = int(np.argmax(np.abs(basis[:, column])))
            if basis[pivot, column] < 0.0:
                basis[:, column] *= -1.0
        embedded = standardized @ basis
    else:
        embedded = np.pad(standardized, ((0, 0), (0, max(0, 3 - standardized.shape[1]))))[:, :3]
    spread = np.percentile(np.linalg.norm(embedded, axis=1), 90) if len(embedded) else 1.0
    embedded = embedded / max(float(spread), 1e-6) * 2.8
    for frame, position in zip(usable, embedded):
        frame.context_position = position.astype(np.float32)


def _finalize_phrase_descriptors(frames: list[AnalysisFrame]) -> None:
    phrases: dict[int, list[AnalysisFrame]] = {}
    for frame in frames:
        if frame.peaks:
            phrases.setdefault(frame.phrase_id, []).append(frame)
    for phrase_frames in phrases.values():
        weights = np.asarray(
            [
                max(
                    0.02,
                    float(np.mean([peak.amplitude for peak in frame.peaks])),
                )
                for frame in phrase_frames
            ],
            dtype=np.float64,
        )
        centroids = np.asarray(
            [frame.spectral_centroid_hz for frame in phrase_frames],
            dtype=np.float64,
        )
        timbres = np.asarray(
            [
                _timbre_descriptor(
                    frame.harmonicity,
                    frame.spectral_contrast,
                    frame.spectral_flatness,
                )
                for frame in phrase_frames
            ],
            dtype=np.float64,
        )
        centroid = float(np.average(centroids, weights=weights))
        timbre = float(np.average(timbres, weights=weights))
        for frame in phrase_frames:
            frame.phrase_centroid_hz = centroid
            frame.phrase_timbre = timbre


def _timbre_descriptor(
    harmonicity: float,
    spectral_contrast: float,
    spectral_flatness: float,
) -> float:
    return float(
        np.clip(
            harmonicity * 0.45
            + spectral_contrast * 0.35
            + (1.0 - spectral_flatness) * 0.20,
            0.0,
            1.0,
        )
    )


def _detect_spectral_peaks(
    magnitude: np.ndarray,
    db: np.ndarray,
    phase: np.ndarray,
    left_magnitude: np.ndarray,
    right_magnitude: np.ndarray,
    frequencies: np.ndarray,
    minimum_frequency: float,
    maximum_frequency: float,
    rms_db: float,
    onset: float,
    flatness: float,
    minimum_peaks: int,
    maximum_peaks: int,
    prominence_threshold: float,
    noise_floor_margin: float,
    silence_threshold: float,
    dominant_frequency: float,
) -> tuple[SpectralPeak, ...]:
    if rms_db < silence_threshold or maximum_peaks <= 0:
        return ()
    valid = np.flatnonzero(
        (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    )
    if valid.size < 3:
        return ()
    first, last = int(valid[0]), int(valid[-1])
    local = db[first : last + 1]
    noise_db = float(np.median(local))
    candidates = np.flatnonzero(
        (local[1:-1] > local[:-2]) & (local[1:-1] >= local[2:])
    ) + first + 1
    if candidates.size:
        candidates = candidates[
            db[candidates] >= noise_db + noise_floor_margin
        ]
    candidate_limit = max(24, maximum_peaks * 6)
    if candidates.size > candidate_limit:
        strongest = np.argpartition(db[candidates], -candidate_limit)[
            -candidate_limit:
        ]
        candidates = np.sort(candidates[strongest])
    results: list[tuple[float, SpectralPeak]] = []
    bin_width = frequencies[1] - frequencies[0]
    for index in candidates:
        neighborhood = db[max(first, index - 6) : min(last + 1, index + 7)]
        shoulders = np.concatenate((neighborhood[: max(0, len(neighborhood) // 2 - 1)],
                                    neighborhood[min(len(neighborhood), len(neighborhood) // 2 + 2) :]))
        shoulder_db = float(np.percentile(shoulders, 35)) if shoulders.size else noise_db
        prominence = float(db[index] - shoulder_db)
        snr = float(db[index] - noise_db)
        if prominence < prominence_threshold or snr < noise_floor_margin:
            continue
        half_level = db[index] - max(3.0, prominence * 0.5)
        left = index
        right = index
        while left > first and db[left] > half_level and index - left < 24:
            left -= 1
        while right < last and db[right] > half_level and right - index < 24:
            right += 1
        stereo = float(
            (right_magnitude[index] - left_magnitude[index])
            / (right_magnitude[index] + left_magnitude[index] + 1e-9)
        )
        frequency = float(frequencies[index])
        ratio = frequency / max(dominant_frequency, 1.0)
        nearest_harmonic = max(1, round(ratio))
        harmonicity = float(np.exp(-abs(ratio - nearest_harmonic) * 5.0))
        amplitude = float(np.clip((db[index] - max(noise_db, -80.0)) / 55.0, 0.0, 1.0))
        item = SpectralPeak(
            frequency_hz=frequency,
            amplitude=amplitude,
            magnitude_db=float(db[index]),
            prominence_db=prominence,
            snr_db=snr,
            bandwidth_hz=float(max(bin_width, frequencies[right] - frequencies[left])),
            phase=float(phase[index]),
            stereo_balance=float(np.clip(stereo, -1.0, 1.0)),
            harmonicity=harmonicity,
        )
        score = snr + prominence * 0.75 + onset * 9.0 + harmonicity * 1.5
        results.append((score, item))
    if not results:
        return ()
    complexity = float(
        np.clip(
            0.40 * flatness
            + 0.35 * onset
            + 0.25 * min(1.0, len(results) / max(1, maximum_peaks)),
            0.0,
            1.0,
        )
    )
    adaptive = int(
        round(minimum_peaks + (maximum_peaks - minimum_peaks) * complexity)
    )
    if len(results) <= 6:
        adaptive = len(results)
    target = min(maximum_peaks, max(1, adaptive), len(results))
    results.sort(key=lambda item: (-item[0], item[1].frequency_hz))
    selected = [item for _, item in results[:target]]
    selected.sort(key=lambda item: item.frequency_hz)
    return tuple(selected)


def _fit_samples(samples: np.ndarray, size: int) -> np.ndarray:
    if samples.shape[1] == size:
        return samples.astype(np.float32, copy=False)
    result = np.zeros((2, size), dtype=np.float32)
    count = min(size, samples.shape[1])
    result[:, :count] = samples[:2, :count]
    return result


def _band_energy(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    minimum: float,
    maximum: float,
    count: int,
) -> np.ndarray:
    edges = np.geomspace(max(10.0, minimum), max(minimum + 1.0, maximum), count + 1)
    result = np.zeros(count, dtype=np.float32)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frequencies >= low) & (frequencies < high)
        if np.any(mask):
            rms = float(np.sqrt(np.mean(magnitude[mask] ** 2)))
            result[index] = np.clip(
                (20.0 * np.log10(max(rms, 1e-8)) + 80.0) / 80.0, 0.0, 1.0
            )
    return result


def _spectral_shape(
    magnitude: np.ndarray, frequencies: np.ndarray
) -> tuple[float, float, float]:
    if magnitude.size == 0:
        return 0.0, 0.0, 0.0
    power = np.maximum(magnitude.astype(np.float64) ** 2, 1e-16)
    total = float(power.sum())
    centroid = float(np.sum(frequencies * power) / total)
    bandwidth = float(
        np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total)
    )
    cumulative = np.cumsum(power)
    rolloff_index = min(
        len(frequencies) - 1,
        int(np.searchsorted(cumulative, cumulative[-1] * 0.85)),
    )
    return centroid, bandwidth, float(frequencies[rolloff_index])


def _onset_strength(magnitude: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None or previous.shape != magnitude.shape:
        return 0.0
    positive_flux = np.maximum(magnitude - previous, 0.0)
    reference = float(np.mean(previous) + 1e-8)
    return float(np.clip(np.mean(positive_flux) / reference, 0.0, 4.0) / 4.0)


def _spectral_flatness(magnitude: np.ndarray) -> float:
    if not magnitude.size:
        return 0.0
    values = np.maximum(magnitude.astype(np.float64), 1e-12)
    return float(np.clip(np.exp(np.mean(np.log(values))) / np.mean(values), 0.0, 1.0))


def _harmonicity(
    magnitude: np.ndarray, frequencies: np.ndarray, fundamental: float
) -> float:
    if not magnitude.size or fundamental <= 0.0:
        return 0.0
    total = float(np.sum(magnitude) + 1e-12)
    harmonic_energy = 0.0
    for multiple in range(1, 9):
        target = fundamental * multiple
        if target > frequencies[-1]:
            break
        index = int(np.argmin(np.abs(frequencies - target)))
        harmonic_energy += float(magnitude[index])
    return float(np.clip(harmonic_energy / total * 3.0, 0.0, 1.0))


def _spectral_contrast(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    minimum: float,
    maximum: float,
    count: int = 6,
) -> np.ndarray:
    if not magnitude.size:
        return np.zeros(count, dtype=np.float32)
    edges = np.geomspace(max(10.0, minimum), max(minimum + 1.0, maximum), count + 1)
    result = np.zeros(count, dtype=np.float32)
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-9))
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        values = db[(frequencies >= low) & (frequencies < high)]
        if values.size:
            result[index] = np.clip(
                (np.percentile(values, 90) - np.percentile(values, 10)) / 60.0,
                0.0,
                1.0,
            )
    return result


def _compact_spectrum(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    minimum: float,
    maximum: float,
    bins: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.geomspace(max(10.0, minimum), max(minimum + 1.0, maximum), bins)
    values = np.interp(targets, frequencies, spectrum)
    return targets.astype(np.float32), values.astype(np.float32)
