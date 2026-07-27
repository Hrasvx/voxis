"""Standalone ModernGL renderer used by deterministic offline export."""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np

from ..config import VisualizationSettings
from ..presets import VisualPreset
from .camera import Camera
from .grid import grid_vertices
from .points import PointManager
from .postprocessing import PostProcessor
from .overlays import overlay_rgb_bytes


class OffscreenRenderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.ctx = _standalone_context()
        shader_dir = Path(__file__).with_name("shaders")
        self.point_program = self.ctx.program(
            vertex_shader=(shader_dir / "point.vert").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "point.frag").read_text(encoding="utf-8"),
        )
        self.line_program = self.ctx.program(
            vertex_shader=(shader_dir / "line.vert").read_text(encoding="utf-8"),
            geometry_shader=(shader_dir / "line.geom").read_text(encoding="utf-8"),
            fragment_shader=(shader_dir / "line.frag").read_text(encoding="utf-8"),
        )
        self.point_buffer = self.ctx.buffer(reserve=25000 * 8 * 4, dynamic=True)
        self.line_buffer = self.ctx.buffer(reserve=100000 * 2 * 7 * 4, dynamic=True)
        self.point_vao = self.ctx.vertex_array(
            self.point_program,
            [
                (
                    self.point_buffer,
                    "3f 4f 1f",
                    "in_position",
                    "in_color",
                    "in_size",
                )
            ],
        )
        self.line_vao = self.ctx.vertex_array(
            self.line_program,
            [(self.line_buffer, "3f 4f", "in_position", "in_color")],
        )
        self.output_texture = self.ctx.texture((width, height), 4, dtype="f1")
        self.output_fbo = self.ctx.framebuffer([self.output_texture])
        self.post = PostProcessor(self.ctx, shader_dir)
        self.post.resize(width, height)
        self.camera = Camera()

    def render(
        self,
        manager: PointManager,
        settings: VisualizationSettings,
        preset: VisualPreset,
        time_s: float,
    ) -> bytes:
        point_data, line_data = manager.vertex_arrays(
            settings, quality=settings.export_quality
        )
        background = tuple(
            value * settings.background_brightness for value in preset.background
        )
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.enable(
            moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
        )
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.post.begin_scene(background)
        angles = (
            settings.rotation_speed_x * time_s if settings.automatic_rotation else 0.0,
            settings.rotation_speed_y * time_s if settings.automatic_rotation else 0.0,
            settings.rotation_speed_z * time_s if settings.automatic_rotation else 0.0,
        )
        center, radius = manager.bounds()
        focus_center, focus_weight, focus_radius = manager.camera_focus(
            settings.camera_focus_hold_duration,
            settings.camera_return_duration,
        )
        if not settings.camera_follow_new_nodes:
            focus_center = None
            focus_weight = 0.0
        if settings.grid_enabled:
            grid_data = grid_vertices(
                center,
                radius,
                settings.grid_spacing,
                settings.grid_opacity,
            )
            line_data = (
                np.vstack((grid_data, line_data))
                if line_data.size
                else grid_data
            )
        model, view, projection = self.camera.matrices(
            self.width / self.height,
            settings.camera_distance,
            settings.field_of_view,
            settings.camera_drift,
            time_s,
            angles,
            center,
            radius,
            settings.auto_fit_camera,
            settings.viewport_occupancy,
            settings.camera_smoothing,
            settings.camera_min_distance,
            settings.camera_max_distance,
            focus_center,
            focus_weight,
            focus_radius,
            settings.camera_follow_strength,
            settings.camera_follow_smoothing,
            settings.camera_follow_max_speed,
        )
        for program in (self.point_program, self.line_program):
            program["model"].write(model.tobytes())
            program["view"].write(view.tobytes())
            program["projection"].write(projection.tobytes())
            program["fog_intensity"].value = settings.fog_intensity

        if line_data.size:
            self.line_program["viewport_size"].value = (self.width, self.height)
            self.line_program["line_width"].value = settings.line_thickness
            self.line_buffer.write(line_data.tobytes())
            self.line_vao.render(mode=moderngl.LINES, vertices=len(line_data))
        if point_data.size:
            self.point_program["pixel_ratio"].value = 1.0
            self.point_program["glow_intensity"].value = settings.glow_intensity
            self.point_program["white_core_intensity"].value = (
                settings.white_core_intensity
            )
            self.point_program["background_color"].value = background
            self.point_buffer.write(point_data.tobytes())
            self.point_vao.render(mode=moderngl.POINTS, vertices=len(point_data))

        self.post.finish(
            self.output_fbo,
            settings.trail_persistence,
            settings.glow_intensity,
            settings.bloom_intensity,
            settings.grain_intensity,
            settings.flicker_intensity,
            settings.scanline_intensity,
            settings.chromatic_aberration,
            background,
            time_s,
        )
        raw = self.output_fbo.read(components=3, alignment=1)
        rgb = (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape(self.height, self.width, 3)[::-1]
            .tobytes()
        )
        return overlay_rgb_bytes(
            rgb,
            self.width,
            self.height,
            manager,
            settings,
            preset,
            model,
            view,
            projection,
        )

    def close(self) -> None:
        self.ctx.release()


def _standalone_context() -> moderngl.Context:
    try:
        return moderngl.create_standalone_context(require=330, backend="egl")
    except Exception:
        return moderngl.create_standalone_context(require=330)
