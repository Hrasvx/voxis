"""ModernGL renderer hosted in a Qt OpenGL widget."""

from __future__ import annotations

from pathlib import Path
import time

import moderngl
import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..config import SettingsState
from ..playback.clock import MediaClock
from ..playback.synchronizer import TimelineSynchronizer
from ..presets import DEFAULT_PRESET, PRESETS, VisualPreset
from .camera import Camera
from .grid import grid_vertices
from .points import PointManager
from .postprocessing import PostProcessor
from .overlays import draw_overlays


class VisualizationWidget(QOpenGLWidget):
    fullscreen_requested = Signal()
    renderer_error = Signal(str)
    performance_changed = Signal(float, int)

    def __init__(
        self,
        clock: MediaClock,
        synchronizer: TimelineSynchronizer,
        settings: SettingsState,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.clock = clock
        self.synchronizer = synchronizer
        self.settings_state = settings
        self.point_manager = PointManager(settings.snapshot().random_seed)
        self.camera = Camera()
        self.preset: VisualPreset = PRESETS[DEFAULT_PRESET]
        self._ctx: moderngl.Context | None = None
        self._post: PostProcessor | None = None
        self._point_program = None
        self._line_program = None
        self._point_buffer = None
        self._line_buffer = None
        self._point_vao = None
        self._line_vao = None
        self._last_frame_at = time.perf_counter()
        self._started_at = self._last_frame_at
        self._angles = np.zeros(3, dtype=np.float32)
        self._last_generation = clock.snapshot().generation
        self._mouse_position: QPoint | None = None
        self._mouse_button = Qt.MouseButton.NoButton
        self._density_scale = 1.0
        self._smoothed_fps = 60.0
        self._frame_counter = 0
        self._gl_failed = False
        self._clear_post_pending = True
        self._pending_graph_finalize: tuple[float, int] | None = None

        self.setMinimumSize(420, 280)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)
        self.refresh_timer()
        self._timer.start()

    def refresh_timer(self) -> None:
        fps = self.settings_state.snapshot().performance_limit_fps
        self._timer.setInterval(max(1, round(1000 / fps)))

    def set_preset(self, name: str) -> None:
        if name in PRESETS:
            self.preset = PRESETS[name]
            self.clear_timeline()

    def reset_camera(self) -> None:
        self.camera.reset()
        self._angles[:] = 0.0

    def clear_timeline(self, generation: int | None = None) -> None:
        self.point_manager.clear(self.settings_state.snapshot().random_seed)
        if generation is not None:
            self._last_generation = generation
        self._angles[:] = 0.0
        self._density_scale = 1.0
        self._clear_post_pending = True
        self._pending_graph_finalize = None
        self.camera.reset_follow()
        self.update()

    def finalize_seek_history(self, timestamp_s: float, generation: int) -> None:
        self._pending_graph_finalize = (timestamp_s, generation)
        self.update()

    def initializeGL(self) -> None:
        try:
            self._ctx = moderngl.create_context(require=330)
            shader_dir = Path(__file__).with_name("shaders")
            self._point_program = self._ctx.program(
                vertex_shader=(shader_dir / "point.vert").read_text(encoding="utf-8"),
                fragment_shader=(shader_dir / "point.frag").read_text(encoding="utf-8"),
            )
            self._line_program = self._ctx.program(
                vertex_shader=(shader_dir / "line.vert").read_text(encoding="utf-8"),
                geometry_shader=(shader_dir / "line.geom").read_text(encoding="utf-8"),
                fragment_shader=(shader_dir / "line.frag").read_text(encoding="utf-8"),
            )
            self._point_buffer = self._ctx.buffer(reserve=25000 * 8 * 4, dynamic=True)
            self._line_buffer = self._ctx.buffer(
                reserve=100000 * 2 * 7 * 4, dynamic=True
            )
            self._point_vao = self._ctx.vertex_array(
                self._point_program,
                [
                    (
                        self._point_buffer,
                        "3f 4f 1f",
                        "in_position",
                        "in_color",
                        "in_size",
                    )
                ],
            )
            self._line_vao = self._ctx.vertex_array(
                self._line_program,
                [(self._line_buffer, "3f 4f", "in_position", "in_color")],
            )
            self._post = PostProcessor(self._ctx, shader_dir)
            self._ctx.enable(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        except Exception as exc:
            self._gl_failed = True
            self.renderer_error.emit(
                "OpenGL 3.3 initialization failed. Check the GPU driver: "
                f"{exc}"
            )

    def resizeGL(self, width: int, height: int) -> None:
        if self._post is None:
            return
        ratio = self.devicePixelRatioF()
        self._post.resize(round(width * ratio), round(height * ratio))

    def paintGL(self) -> None:
        if self._gl_failed or self._ctx is None or self._post is None:
            return
        now = time.perf_counter()
        raw_delta = max(1e-4, now - self._last_frame_at)
        delta = min(raw_delta, 0.05)
        self._last_frame_at = now
        snapshot = self.clock.snapshot()
        settings = self.settings_state.snapshot()

        if snapshot.generation != self._last_generation:
            self.clear_timeline(snapshot.generation)
        if self._clear_post_pending:
            self._post.clear()
            self._clear_post_pending = False

        frames = self.synchronizer.consume_ready(
            snapshot.position_s, snapshot.generation
        )
        for frame in frames:
            self.point_manager.ingest(
                frame,
                settings,
                self.preset,
                density_scale=1.0,
            )
        self.point_manager.update_at(snapshot.position_s, settings)
        if self._pending_graph_finalize is not None:
            _, generation = self._pending_graph_finalize
            if generation == snapshot.generation:
                self.point_manager.finalize_graph(snapshot.position_s, settings)
            self._pending_graph_finalize = None
        if settings.automatic_rotation:
            media_rotation_time = snapshot.position_s
            self._angles = np.asarray(
                (
                    settings.rotation_speed_x * media_rotation_time,
                    settings.rotation_speed_y * media_rotation_time,
                    settings.rotation_speed_z * media_rotation_time,
                ),
                dtype=np.float32,
            )

        self._render(settings, now)
        self._adapt_performance(raw_delta, settings.performance_limit_fps)

    def _render(self, settings, now: float) -> None:
        assert self._ctx is not None and self._post is not None
        assert self._point_program is not None and self._line_program is not None
        point_data, line_data = self.point_manager.vertex_arrays(
            settings,
            render_fraction=self._density_scale,
            quality=settings.preview_quality,
        )
        ratio = self.devicePixelRatioF()
        width = max(1, round(self.width() * ratio))
        height = max(1, round(self.height() * ratio))
        target = self._ctx.detect_framebuffer(
            glo=int(self.defaultFramebufferObject())
        )
        self._post.resize(width, height)
        self._ctx.viewport = (0, 0, width, height)
        self._ctx.enable(
            moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
        )
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        background = tuple(
            component * settings.background_brightness
            for component in self.preset.background
        )
        self._post.begin_scene(background)

        center, radius = self.point_manager.bounds()
        focus_center, focus_weight, focus_radius = self.point_manager.camera_focus(
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
            width / height,
            settings.camera_distance,
            settings.field_of_view,
            settings.camera_drift,
            self.point_manager.visual_time,
            tuple(float(value) for value in self._angles),
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
        for program in (self._point_program, self._line_program):
            program["model"].write(model.tobytes())
            program["view"].write(view.tobytes())
            program["projection"].write(projection.tobytes())
            program["fog_intensity"].value = settings.fog_intensity

        if line_data.size:
            self._line_program["viewport_size"].value = (width, height)
            self._line_program["line_width"].value = settings.line_thickness
            self._line_buffer.write(line_data.tobytes())
            self._line_vao.render(
                mode=moderngl.LINES,
                vertices=len(line_data),
            )
        if point_data.size:
            self._point_program["pixel_ratio"].value = ratio
            self._point_program["glow_intensity"].value = settings.glow_intensity
            self._point_program["white_core_intensity"].value = (
                settings.white_core_intensity
            )
            self._point_program["background_color"].value = background
            self._point_buffer.write(point_data.tobytes())
            self._point_vao.render(
                mode=moderngl.POINTS,
                vertices=len(point_data),
            )

        self._post.finish(
            target,
            settings.trail_persistence,
            settings.glow_intensity,
            settings.bloom_intensity,
            settings.grain_intensity,
            settings.flicker_intensity,
            settings.scanline_intensity,
            settings.chromatic_aberration,
            background,
            self.point_manager.visual_time,
        )
        if settings.scientific_labels or settings.frequency_legend_preview:
            painter = QPainter(self)
            draw_overlays(
                painter,
                self.point_manager,
                settings,
                self.preset,
                self.width(),
                self.height(),
                model,
                view,
                projection,
                export=False,
            )
            painter.end()

    def _adapt_performance(self, delta: float, limit_fps: int) -> None:
        fps = 1.0 / max(delta, 1e-4)
        self._smoothed_fps = self._smoothed_fps * 0.94 + fps * 0.06
        target = float(limit_fps)
        if self._smoothed_fps < target * 0.78:
            self._density_scale = max(0.22, self._density_scale * 0.985)
        elif self._smoothed_fps > target * 0.92:
            self._density_scale = min(1.0, self._density_scale + 0.003)
        self._frame_counter += 1
        if self._frame_counter % 30 == 0:
            self.performance_changed.emit(
                self._smoothed_fps, len(self.point_manager.points)
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._mouse_position = event.position().toPoint()
            self._mouse_button = event.button()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mouse_position is not None:
            current = event.position().toPoint()
            delta = current - self._mouse_position
            if self._mouse_button == Qt.MouseButton.LeftButton:
                self.camera.rotate(delta.x(), delta.y())
            else:
                self.camera.pan(delta.x(), delta.y())
            self._mouse_position = current
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._mouse_position = None
        self._mouse_button = Qt.MouseButton.NoButton
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.fullscreen_requested.emit()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.camera.zoom(-event.angleDelta().y() / 480.0)
        self.update()
        event.accept()
