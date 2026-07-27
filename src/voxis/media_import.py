"""Public media-import boundary used by the UI and export system."""

from __future__ import annotations

from .models import MediaInfo
from .playback.media_probe import MediaProbeError, probe_media

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
}


def inspect_audio_source(path: str) -> MediaInfo:
    """Validate a video/audio file and return only audio-relevant metadata."""
    return probe_media(path)


__all__ = ["MediaProbeError", "SUPPORTED_EXTENSIONS", "inspect_audio_source"]
