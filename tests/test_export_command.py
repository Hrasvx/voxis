from pathlib import Path

from voxis.export.ffmpeg_encoder import (
    ExportOptions,
    build_ffmpeg_command,
)
from voxis.export.worker import _analysis_frame_due


def test_export_maps_generated_video_and_source_audio_only() -> None:
    options = ExportOptions("/tmp/result.mp4", 1920, 1080, 60, 16000)
    command = build_ffmpeg_command(
        "/usr/bin/ffmpeg", "/media/source.mov", 12.5, options
    )

    maps = [
        command[index + 1]
        for index, token in enumerate(command[:-1])
        if token == "-map"
    ]
    assert maps == ["0:v:0", "1:a:0"]
    assert "1:v" not in " ".join(command)
    assert command[command.index("-t") + 1] == "12.500000"
    assert command[-1] == "/tmp/result.mp4"


def test_export_dimensions_are_even_and_fps_is_supported() -> None:
    options = ExportOptions("result.mp4", 1919, 1079, 47, 12000).validated()
    assert (options.width, options.height) == (1918, 1078)
    assert options.fps == 30


def test_24_fps_is_supported() -> None:
    assert ExportOptions("result.mp4", 1080, 1920, 24, 12000).validated().fps == 24


def test_export_never_reveals_analysis_frames_before_their_timestamp() -> None:
    assert _analysis_frame_due(0.041, 0.041)
    assert not _analysis_frame_due(0.042, 0.041)


def test_runtime_source_has_no_video_decode_or_video_widget_path() -> None:
    package = Path(__file__).parents[1] / "src" / "voxis"
    assert not (package / "playback" / "video_decode.py").exists()
    playback_source = (package / "playback" / "controller.py").read_text()
    renderer_source = (package / "visualization" / "renderer.py").read_text()
    assert "setVideoOutput" not in playback_source
    assert "QVideo" not in playback_source + renderer_source
