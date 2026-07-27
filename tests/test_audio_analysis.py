import numpy as np

from voxis.analysis.audio_analyzer import AudioAnalyzer
from voxis.models import AudioChunk


def test_analyzer_finds_sine_dominant_frequency_and_timestamp() -> None:
    sample_rate = 48000
    size = 4096
    frequency = 1000.0
    time = np.arange(size) / sample_rate
    signal = (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
    samples = np.vstack((signal, signal))
    analyzer = AudioAnalyzer(size, 0.5, 30.0, 18000.0)

    result = analyzer.analyze(AudioChunk(samples, 2.0, sample_rate), generation=7)

    assert abs(result.dominant_hz - frequency) < sample_rate / size
    assert result.timestamp_s == 2.0 + size / sample_rate / 2.0
    assert result.rms_db > -12.0
    assert result.phase_correlation > 0.99
    assert abs(result.spectral_centroid_hz - frequency) < 80.0
    assert result.spectral_bandwidth_hz < 150.0
    assert abs(result.spectral_rolloff_hz - frequency) < 80.0
    assert len(result.spectrum) == 192
    assert result.generation == 7


def test_onset_strength_increases_for_a_new_transient() -> None:
    sample_rate = 48000
    size = 2048
    analyzer = AudioAnalyzer(size, 0.5, 30.0, 18000.0)
    silence = np.zeros((2, size), dtype=np.float32)
    impulse = np.zeros((2, size), dtype=np.float32)
    impulse[:, size // 2] = 1.0

    first = analyzer.analyze(AudioChunk(silence, 0.0, sample_rate), generation=0)
    second = analyzer.analyze(
        AudioChunk(impulse, size / sample_rate, sample_rate), generation=0
    )

    assert first.onset_strength == 0.0
    assert second.onset_strength > 0.5


def test_higher_sensitivity_accepts_quieter_spectral_peaks() -> None:
    sample_rate = 48000
    size = 4096
    time = np.arange(size) / sample_rate
    signal = (
        0.002 * np.sin(2.0 * np.pi * 2200.0 * time)
    ).astype(np.float32)
    samples = np.vstack((signal, signal))
    common = {
        "minimum_peaks": 0,
        "maximum_peaks": 10,
        "peak_prominence_db": 3.0,
        "noise_floor_db": 4.0,
        "silence_threshold_db": -62.0,
        "hop_size": 256,
    }
    low = AudioAnalyzer(
        size, 0.3, 300.0, 12000.0, sensitivity=0.25, **common
    )
    high = AudioAnalyzer(
        size, 0.3, 300.0, 12000.0, sensitivity=4.0, **common
    )

    low_result = low.analyze(AudioChunk(samples, 0.0, sample_rate), 0)
    high_result = high.analyze(AudioChunk(samples, 0.0, sample_rate), 0)

    assert low_result.peaks == ()
    assert len(high_result.peaks) >= 1
