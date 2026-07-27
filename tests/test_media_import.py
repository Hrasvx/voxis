import wave

import numpy as np
import pytest

from voxis.analysis.feature_cache import build_feature_cache
from voxis.config import VisualizationSettings
from voxis.media_import import MediaProbeError, inspect_audio_source


def make_wave(path, duration_s: float = 0.25) -> None:
    sample_rate = 44100
    time = np.arange(round(sample_rate * duration_s)) / sample_rate
    samples = (
        np.sin(2.0 * np.pi * 440.0 * time) * 0.5 * np.iinfo(np.int16).max
    ).astype("<i2")
    stereo = np.column_stack((samples, samples)).ravel()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(stereo.tobytes())


def test_audio_only_file_is_accepted_and_feature_cache_is_repeatable(tmp_path) -> None:
    source = tmp_path / "tone.wav"
    make_wave(source)
    media = inspect_audio_source(str(source))
    settings = VisualizationSettings(fft_size=512)

    first = build_feature_cache(str(source), media.duration_s, settings)
    second = build_feature_cache(str(source), media.duration_s, settings)

    assert media.has_audio
    assert not media.has_video
    assert media.audio_codec.startswith("pcm")
    assert len(first.frames) == len(second.frames) > 0
    np.testing.assert_array_equal(
        first.frames[2].spectrum, second.frames[2].spectrum
    )


def test_corrupt_file_reports_media_error(tmp_path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"this is not media")
    with pytest.raises(MediaProbeError):
        inspect_audio_source(str(source))
