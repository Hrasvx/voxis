"""Audio-source probing and validation via FFmpeg through PyAV."""

from __future__ import annotations

from pathlib import Path

import av
from av.error import FFmpegError

from ..models import MediaInfo


class MediaProbeError(RuntimeError):
    pass


def probe_media(path: str) -> MediaInfo:
    source = Path(path)
    if not source.is_file():
        raise MediaProbeError(f"File does not exist: {source}")
    try:
        with av.open(str(source), mode="r") as container:
            video = next((s for s in container.streams if s.type == "video"), None)
            audio = next((s for s in container.streams if s.type == "audio"), None)
            if audio is None:
                raise MediaProbeError(
                    "The selected file contains no audio stream to visualize."
                )
            duration_s = _duration_seconds(container, audio)
            frame_rate = (
                float(video.average_rate)
                if video is not None and video.average_rate
                else 0.0
            )
            return MediaInfo(
                path=str(source.resolve()),
                duration_s=duration_s,
                width=int(video.codec_context.width or 0) if video else 0,
                height=int(video.codec_context.height or 0) if video else 0,
                frame_rate=frame_rate,
                has_video=video is not None,
                has_audio=True,
                video_codec=video.codec_context.name or "" if video else "",
                audio_codec=audio.codec_context.name or "",
            )
    except MediaProbeError:
        raise
    except (FFmpegError, OSError, ValueError) as exc:
        raise MediaProbeError(
            f"FFmpeg could not open this media file: {exc}"
        ) from exc


def _duration_seconds(container: av.container.InputContainer, stream: object) -> float:
    if container.duration is not None:
        return max(0.0, float(container.duration) / float(av.time_base))
    duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if duration is not None and time_base is not None:
        return max(0.0, float(duration * time_base))
    return 0.0
