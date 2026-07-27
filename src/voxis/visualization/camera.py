"""Minimal perspective-camera and multi-axis tumble mathematics."""

from __future__ import annotations

import math

import numpy as np


class Camera:
    def __init__(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.distance_offset = 0.0
        self.pan_offset = np.zeros(2, dtype=np.float32)
        self._fit_distance: float | None = None
        self._fit_center = np.zeros(3, dtype=np.float32)
        self._follow_center: np.ndarray | None = None
        self._last_media_time: float | None = None

    def reset(self) -> None:
        self.yaw = self.pitch = self.roll = self.distance_offset = 0.0
        self._fit_distance = None
        self._fit_center[:] = 0.0
        self.pan_offset[:] = 0.0
        self.reset_follow()

    def reset_follow(self) -> None:
        self._follow_center = None
        self._last_media_time = None

    def rotate(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.006
        self.pitch = float(np.clip(self.pitch + dy * 0.006, -1.4, 1.4))

    def zoom(self, delta: float) -> None:
        self.distance_offset = float(
            np.clip(self.distance_offset + delta, -3.0, 8.0)
        )

    def pan(self, dx: float, dy: float) -> None:
        self.pan_offset[0] += dx * 0.006
        self.pan_offset[1] -= dy * 0.006

    def matrices(
        self,
        aspect: float,
        camera_distance: float,
        field_of_view: float,
        camera_drift: float,
        time_s: float,
        automatic_angles: tuple[float, float, float],
        geometry_center: np.ndarray | None = None,
        geometry_radius: float = 2.0,
        auto_fit: bool = False,
        occupancy: float = 0.75,
        smoothing: float = 0.12,
        minimum_distance: float = 3.0,
        maximum_distance: float = 30.0,
        focus_center: np.ndarray | None = None,
        focus_weight: float = 0.0,
        focus_radius: float = 0.55,
        follow_strength: float = 0.0,
        follow_smoothing: float = 0.48,
        follow_max_speed: float = 6.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ax, ay, az = automatic_angles
        ax += math.sin(time_s * 0.071) * 0.055
        ay += math.sin(time_s * 0.053 + 1.4) * 0.043
        az += math.sin(time_s * 0.089 + 2.1) * 0.028
        rotation = (
            _rotation_x(ax + self.pitch)
            @ _rotation_y(ay + self.yaw)
            @ _rotation_z(az + self.roll)
        )
        geometry_center_value = (
            np.asarray(geometry_center, dtype=np.float32)
            if geometry_center is not None
            else np.zeros(3, dtype=np.float32)
        )
        blend = float(np.clip(focus_weight * follow_strength, 0.0, 1.0))
        target_center = geometry_center_value
        target_radius = geometry_radius
        if focus_center is not None and blend > 0.0:
            focus = np.asarray(focus_center, dtype=np.float32)
            target_center = (
                geometry_center_value * (1.0 - blend) + focus * blend
            )
            radius_blend = blend * 0.62
            target_radius = (
                geometry_radius * (1.0 - radius_blend)
                + max(0.55, focus_radius) * radius_blend
            )
        discontinuity = (
            self._last_media_time is None
            or time_s < self._last_media_time - 1e-6
            or time_s - self._last_media_time > 1.0
        )
        elapsed = (
            0.0
            if self._last_media_time is None
            else max(0.0, time_s - self._last_media_time)
        )
        self._last_media_time = time_s
        if self._follow_center is None or discontinuity:
            self._follow_center = target_center.copy()
        elif elapsed > 0.0:
            responsiveness = 1.5 + float(np.clip(follow_smoothing, 0.01, 1.0)) * 18.5
            factor = 1.0 - math.exp(-responsiveness * elapsed)
            movement = (target_center - self._follow_center) * factor
            distance_to_target = float(np.linalg.norm(movement))
            maximum_step = max(0.1, follow_max_speed) * elapsed
            if distance_to_target > maximum_step:
                movement *= maximum_step / max(distance_to_target, 1e-9)
            self._follow_center += movement
        center = self._follow_center
        distance = camera_distance
        if auto_fit:
            vertical_half_angle = math.radians(field_of_view) * 0.5
            target = target_radius / max(
                0.12, math.tan(vertical_half_angle) * occupancy
            )
            target *= 0.74
            if aspect < 1.0:
                target /= max(0.35, aspect)
            target = float(np.clip(target, minimum_distance, maximum_distance))
            if self._fit_distance is None:
                self._fit_distance = target
                self._fit_center = center.copy()
            factor = max(
                float(np.clip(smoothing, 0.01, 1.0)),
                float(np.clip(follow_smoothing * blend, 0.0, 1.0)),
            )
            self._fit_distance += (target - self._fit_distance) * factor
            self._fit_center += (center - self._fit_center) * factor
            distance = self._fit_distance
            center = self._fit_center
        model = rotation @ _translation(-center)
        view = np.eye(4, dtype=np.float32)
        view[0, 3] = (
            float(self.pan_offset[0]) + math.sin(time_s * 0.17) * camera_drift
        )
        view[1, 3] = (
            float(self.pan_offset[1])
            + math.cos(time_s * 0.13) * camera_drift * 0.65
        )
        view[2, 3] = -(distance + self.distance_offset)
        projection = _perspective(
            math.radians(field_of_view), max(aspect, 0.01), 0.1, 100.0
        )
        return model.T.copy(), view.T.copy(), projection.T.copy()


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        ((1, 0, 0, 0), (0, c, -s, 0), (0, s, c, 0), (0, 0, 0, 1)),
        dtype=np.float32,
    )


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        ((c, 0, s, 0), (0, 1, 0, 0), (-s, 0, c, 0), (0, 0, 0, 1)),
        dtype=np.float32,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        ((c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype=np.float32,
    )


def _translation(offset: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, 3] = offset[:3]
    return result


def _perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov * 0.5)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    return result
