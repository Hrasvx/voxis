"""Raw RGB frame streaming and source-audio muxing through FFmpeg."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from ..audio_extraction import find_ffmpeg


@dataclass(frozen=True, slots=True)
class ExportOptions:
    output_path: str
    width: int = 1920
    height: int = 1080
    fps: int = 60
    video_bitrate_kbps: int = 16000

    def validated(self) -> "ExportOptions":
        width = max(320, min(7680, int(self.width)))
        height = max(180, min(4320, int(self.height)))
        width -= width % 2
        height -= height % 2
        requested_fps = int(self.fps)
        fps = 60 if requested_fps >= 60 else (30 if requested_fps >= 30 else 24)
        bitrate = max(500, min(100000, int(self.video_bitrate_kbps)))
        output = str(Path(self.output_path).expanduser().resolve())
        return ExportOptions(output, width, height, fps, bitrate)


class FFmpegEncoder:
    def __init__(
        self,
        source_path: str,
        duration_s: float,
        options: ExportOptions,
    ) -> None:
        self.options = options.validated()
        command = build_ffmpeg_command(
            find_ffmpeg(), source_path, duration_s, self.options
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, rgb_frame: bytes) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg frame input is closed.")
        try:
            self.process.stdin.write(rgb_frame)
        except BrokenPipeError as exc:
            raise RuntimeError(self._error_text() or "FFmpeg stopped unexpectedly.") from exc

    def finish(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(self._error_text() or f"FFmpeg exited with {return_code}.")

    def cancel(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def _error_text(self) -> str:
        if self.process.stderr is None:
            return ""
        return self.process.stderr.read().decode("utf-8", errors="replace").strip()


def build_ffmpeg_command(
    executable: str,
    source_path: str,
    duration_s: float,
    options: ExportOptions,
) -> list[str]:
    options = options.validated()
    return [
        executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{options.width}x{options.height}",
        "-framerate",
        str(options.fps),
        "-i",
        "pipe:0",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        f"{options.video_bitrate_kbps}k",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration_s:.6f}",
        "-movflags",
        "+faststart",
        options.output_path,
    ]
