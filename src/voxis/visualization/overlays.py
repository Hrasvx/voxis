"""Small scientific labels and logarithmic frequency legend."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
)

from ..config import VisualizationSettings
from ..presets import VisualPreset
from .points import PointManager, _palette_color


def draw_overlays(
    painter: QPainter,
    manager: PointManager,
    settings: VisualizationSettings,
    preset: VisualPreset,
    width: int,
    height: int,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
    *,
    export: bool,
) -> None:
    labels_enabled = (
        settings.scientific_labels_export if export else settings.scientific_labels
    )
    legend_enabled = (
        settings.frequency_legend_export if export else settings.frequency_legend_preview
    )
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    if labels_enabled:
        _draw_labels(
            painter,
            manager,
            settings,
            width,
            height,
            model,
            view,
            projection,
        )
    if legend_enabled:
        _draw_legend(painter, settings, preset, width, height)


def overlay_rgb_bytes(
    rgb: bytes,
    width: int,
    height: int,
    manager: PointManager,
    settings: VisualizationSettings,
    preset: VisualPreset,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
) -> bytes:
    """Paint export overlays onto a top-down tightly packed RGB frame."""
    if not settings.scientific_labels_export and not settings.frequency_legend_export:
        return rgb
    source = QImage(
        rgb, width, height, width * 3, QImage.Format.Format_RGB888
    ).copy()
    painter = QPainter(source)
    draw_overlays(
        painter,
        manager,
        settings,
        preset,
        width,
        height,
        model,
        view,
        projection,
        export=True,
    )
    painter.end()
    result = np.empty((height, width, 3), dtype=np.uint8)
    bits = source.bits()
    row_bytes = source.bytesPerLine()
    raw = np.frombuffer(bits, dtype=np.uint8, count=row_bytes * height)
    for row in range(height):
        result[row] = raw[row * row_bytes : row * row_bytes + width * 3].reshape(
            width, 3
        )
    return result.tobytes()


def _draw_labels(
    painter: QPainter,
    manager: PointManager,
    settings: VisualizationSettings,
    width: int,
    height: int,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
) -> None:
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    font.setPixelSize(
        max(6, round(settings.label_text_size * height / 1080.0))
    )
    painter.setFont(font)
    for point in manager.important_points(
        settings.label_percentage,
        settings.label_max_count,
        settings.active_duration,
    ):
        projected = _project(
            point.position, width, height, model, view, projection
        )
        if projected is None:
            continue
        x, y, depth = projected
        if x < 4 or x > width - 120 or y < 4 or y > height - 12:
            continue
        opacity = max(
            settings.label_min_opacity,
            min(1.0, point.alpha * (1.0 - depth * 0.30)),
        )
        if opacity < 0.08:
            continue
        color = QColor.fromRgbF(
            float(point.color[0]),
            float(point.color[1]),
            float(point.color[2]),
            opacity * 0.82,
        )
        painter.setPen(QPen(color, 0.55))
        active_scale = 1.0 + 0.22 * (
            1.0
            - min(
                1.0,
                point.age / max(settings.node_settle_duration, 1e-6),
            )
        )
        outline = (
            (
                settings.label_box_size
                + point.importance * 1.1
            )
            * height
            / 1080.0
            * active_scale
        )
        painter.drawRect(QRectF(x - outline, y - outline, outline * 2, outline * 2))
        painter.setPen(QPen(QColor.fromRgbF(0.93, 0.96, 1.0, opacity), 0.5))
        frequency_label = (
            f"{point.frequency_hz / 1000.0:0.2f} kHz"
            if point.frequency_hz >= 1000.0
            else f"{point.frequency_hz:0.0f} Hz"
        )
        painter.drawText(
            QPointF(x + outline + 2, y - 1),
            frequency_label,
        )
        secondary = QFont(font)
        secondary.setPixelSize(max(4, font.pixelSize() - 2))
        painter.setFont(secondary)
        painter.setPen(
            QPen(QColor.fromRgbF(0.76, 0.80, 0.84, opacity * 0.72), 0.5)
        )
        painter.drawText(
            QPointF(x + outline + 2, y + secondary.pixelSize()),
            f"{point.magnitude_db:0.1f} dB  ·  A {point.amplitude:0.2f}",
        )
        painter.setFont(font)


def _draw_legend(
    painter: QPainter,
    settings: VisualizationSettings,
    preset: VisualPreset,
    width: int,
    height: int,
) -> None:
    legend_height = min(height * 0.58, 520.0)
    legend_width = max(8.0, width / 170.0)
    x = width - max(24.0, width * 0.027)
    y = (height - legend_height) * 0.5
    gradient = QLinearGradient(x, y, x, y + legend_height)
    for index in range(17):
        visual = index / 16.0
        frequency_position = 1.0 - visual
        if settings.reverse_palette:
            frequency_position = 1.0 - frequency_position
        color = _palette_color(preset, frequency_position)
        gradient.setColorAt(
            visual,
            QColor.fromRgbF(
                float(color[0]), float(color[1]), float(color[2]), 0.92
            ),
        )
    painter.fillRect(
        QRectF(x, y, legend_width, legend_height),
        gradient,
    )
    painter.setPen(QPen(QColor.fromRgbF(0.88, 0.92, 0.96, 0.72), 0.6))
    painter.drawRect(QRectF(x, y, legend_width, legend_height))
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    font.setPixelSize(max(7, round(height / 155)))
    painter.setFont(font)
    minimum = settings.frequency_min
    maximum = settings.frequency_max
    exponent_min = math.floor(math.log10(minimum))
    exponent_max = math.ceil(math.log10(maximum))
    ticks: list[float] = [minimum, maximum]
    for exponent in range(exponent_min, exponent_max + 1):
        for multiplier in (1.0, 2.0, 5.0):
            frequency = multiplier * 10**exponent
            if minimum < frequency < maximum:
                ticks.append(frequency)
    for frequency in sorted(set(ticks)):
        normalized = math.log(frequency / minimum) / math.log(maximum / minimum)
        tick_y = y + (1.0 - normalized) * legend_height
        painter.drawLine(
            QPointF(x - 3, tick_y), QPointF(x + legend_width + 3, tick_y)
        )
        label = (
            f"{frequency/1000.0:g} kHz"
            if frequency >= 1000.0
            else f"{frequency:g} Hz"
        )
        painter.drawText(QPointF(x - 7 - len(label) * 5.5, tick_y + 3), label)


def _project(
    position: np.ndarray,
    width: int,
    height: int,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
) -> tuple[float, float, float] | None:
    vector = np.asarray((*position, 1.0), dtype=np.float32)
    clip = projection.T @ view.T @ model.T @ vector
    if clip[3] <= 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    if abs(ndc[0]) > 1.15 or abs(ndc[1]) > 1.15 or not -1.0 <= ndc[2] <= 1.0:
        return None
    return (
        float((ndc[0] * 0.5 + 0.5) * width),
        float((1.0 - (ndc[1] * 0.5 + 0.5)) * height),
        float(ndc[2] * 0.5 + 0.5),
    )
