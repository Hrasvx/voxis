"""Seekable, chunked audio decoding implemented with PyAV."""

from __future__ import annotations

from collections.abc import Iterator
import av
import numpy as np

from ..models import AudioChunk


class NoAudioStreamError(RuntimeError):
    pass


class AudioDecoder:
    """Owns one PyAV input context and yields timestamped stereo float chunks."""

    def __init__(self, path: str, sample_rate: int = 48000) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.container = av.open(path, mode="r")
        self.stream = next(
            (stream for stream in self.container.streams if stream.type == "audio"),
            None,
        )
        if self.stream is None:
            self.container.close()
            raise NoAudioStreamError("This video has no audio stream.")
        self.resampler = av.AudioResampler(
            format="fltp", layout="stereo", rate=sample_rate
        )

    def close(self) -> None:
        self.container.close()

    def chunks(
        self,
        start_s: float,
        chunk_samples: int,
        hop_samples: int | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield deterministic overlapping analysis windows from the audio stream."""
        start_s = max(0.0, start_s)
        hop_samples = max(1, min(chunk_samples, hop_samples or chunk_samples))
        self.resampler = av.AudioResampler(
            format="fltp", layout="stereo", rate=self.sample_rate
        )
        self.container.seek(
            int(start_s * av.time_base),
            backward=True,
            any_frame=False,
        )

        buffered = np.empty((2, 0), dtype=np.float32)
        buffer_start: float | None = None
        emitted_until = start_s

        for frame in self.container.decode(self.stream):
            converted = self.resampler.resample(frame)
            if converted is None:
                continue
            converted_frames = converted if isinstance(converted, list) else [converted]
            for audio_frame in converted_frames:
                array = _stereo_array(audio_frame)
                frame_time = _frame_seconds(audio_frame, frame, emitted_until)
                frame_end = frame_time + array.shape[1] / self.sample_rate
                if frame_end <= start_s:
                    continue
                if frame_time < start_s:
                    trim = int((start_s - frame_time) * self.sample_rate)
                    array = array[:, trim:]
                    frame_time += trim / self.sample_rate
                if array.shape[1] == 0:
                    continue
                if buffered.shape[1] == 0:
                    buffer_start = max(frame_time, emitted_until)
                buffered = np.concatenate((buffered, array), axis=1)
                while buffered.shape[1] >= chunk_samples:
                    assert buffer_start is not None
                    block = buffered[:, :chunk_samples].copy()
                    buffered = buffered[:, hop_samples:]
                    yield AudioChunk(block, buffer_start, self.sample_rate)
                    buffer_start += hop_samples / self.sample_rate
                    emitted_until = buffer_start
        if buffered.shape[1] and buffer_start is not None:
            block = np.zeros((2, chunk_samples), dtype=np.float32)
            block[:, : buffered.shape[1]] = buffered
            yield AudioChunk(block, buffer_start, self.sample_rate)


def _stereo_array(frame: av.AudioFrame) -> np.ndarray:
    values = frame.to_ndarray().astype(np.float32, copy=False)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] == 1:
        values = np.repeat(values, 2, axis=0)
    elif values.shape[0] > 2:
        values = values[:2]
    return np.ascontiguousarray(values)


def _frame_seconds(
    resampled: av.AudioFrame, original: av.AudioFrame, fallback: float
) -> float:
    if resampled.pts is not None and resampled.time_base is not None:
        return float(resampled.pts * resampled.time_base)
    if original.pts is not None and original.time_base is not None:
        return float(original.pts * original.time_base)
    return fallback
