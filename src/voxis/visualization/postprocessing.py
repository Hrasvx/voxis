"""Ping-pong trail accumulation and analog final-pass shaders."""

from __future__ import annotations

from pathlib import Path

import moderngl


class PostProcessor:
    def __init__(self, context: moderngl.Context, shader_dir: Path) -> None:
        self.ctx = context
        fullscreen = (shader_dir / "fullscreen.vert").read_text(encoding="utf-8")
        self.trail_program = context.program(
            vertex_shader=fullscreen,
            fragment_shader=(shader_dir / "trail.frag").read_text(encoding="utf-8"),
        )
        self.analog_program = context.program(
            vertex_shader=fullscreen,
            fragment_shader=(shader_dir / "analog.frag").read_text(encoding="utf-8"),
        )
        self.trail_vao = context.vertex_array(self.trail_program, [])
        self.analog_vao = context.vertex_array(self.analog_program, [])
        self.size = (0, 0)
        self.scene_texture: moderngl.Texture | None = None
        self.scene_depth: moderngl.Renderbuffer | None = None
        self.scene_fbo: moderngl.Framebuffer | None = None
        self.history_textures: list[moderngl.Texture] = []
        self.history_fbos: list[moderngl.Framebuffer] = []
        self.history_index = 0

    def resize(self, width: int, height: int) -> None:
        size = (max(1, width), max(1, height))
        if size == self.size:
            return
        self.release_targets()
        self.size = size
        self.scene_texture = self.ctx.texture(size, 4, dtype="f1")
        self.scene_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_depth = self.ctx.depth_renderbuffer(size)
        self.scene_fbo = self.ctx.framebuffer(
            [self.scene_texture], self.scene_depth
        )
        for _ in range(2):
            texture = self.ctx.texture(size, 4, dtype="f1")
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            framebuffer = self.ctx.framebuffer([texture])
            framebuffer.clear(0.0, 0.0, 0.0, 1.0)
            self.history_textures.append(texture)
            self.history_fbos.append(framebuffer)
        self.history_index = 0

    def clear(self) -> None:
        for framebuffer in self.history_fbos:
            framebuffer.clear(0.0, 0.0, 0.0, 1.0)

    def begin_scene(self, background: tuple[float, float, float]) -> None:
        assert self.scene_fbo is not None
        self.scene_fbo.use()
        self.scene_fbo.clear(*background, 1.0, depth=1.0)

    def finish(
        self,
        target: moderngl.Framebuffer,
        persistence: float,
        glow: float,
        bloom: float,
        grain: float,
        flicker: float,
        scanlines: float,
        chromatic_aberration: float,
        background: tuple[float, float, float],
        time_s: float,
    ) -> None:
        assert self.scene_texture is not None and self.history_textures
        previous = self.history_index
        destination = 1 - previous

        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.history_fbos[destination].use()
        self.scene_texture.use(location=0)
        self.history_textures[previous].use(location=1)
        self.trail_program["current_frame"].value = 0
        self.trail_program["previous_frame"].value = 1
        self.trail_program["persistence"].value = persistence
        self.trail_program["glow_intensity"].value = glow + bloom * 1.4
        self.trail_program["texel"].value = (
            1.0 / self.size[0],
            1.0 / self.size[1],
        )
        self.trail_vao.render(vertices=3)
        self.history_index = destination

        target.use()
        self.ctx.viewport = (0, 0, *self.size)
        self.history_textures[destination].use(location=0)
        self.analog_program["image"].value = 0
        self.analog_program["time"].value = time_s
        self.analog_program["grain_intensity"].value = grain
        self.analog_program["flicker_intensity"].value = flicker
        self.analog_program["scanline_intensity"].value = scanlines
        self.analog_program["chromatic_aberration"].value = chromatic_aberration
        self.analog_program["background_color"].value = background
        self.analog_program["resolution"].value = self.size
        self.analog_vao.render(vertices=3)

    def release_targets(self) -> None:
        resources = [
            self.scene_fbo,
            self.scene_depth,
            self.scene_texture,
            *self.history_fbos,
            *self.history_textures,
        ]
        for resource in resources:
            if resource is not None:
                resource.release()
        self.scene_fbo = None
        self.scene_depth = None
        self.scene_texture = None
        self.history_fbos = []
        self.history_textures = []
