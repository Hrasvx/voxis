"""FFmpeg discovery and source-audio mapping helpers."""

from __future__ import annotations

import shutil


class FFmpegUnavailableError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise FFmpegUnavailableError(
            "FFmpeg was not found in PATH. Install FFmpeg before exporting."
        )
    return executable

