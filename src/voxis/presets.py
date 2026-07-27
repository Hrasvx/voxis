"""Curated archival/scientific palettes and setting overrides."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisualPreset:
    name: str
    colors: tuple[tuple[float, float, float], ...]
    background: tuple[float, float, float]
    overrides: dict[str, float]


PRESETS: dict[str, VisualPreset] = {
    "Full Spectral Rainbow": VisualPreset(
        "Full Spectral Rainbow",
        (
            (0.72, 0.08, 1.00),
            (1.00, 0.04, 0.52),
            (1.00, 0.10, 0.05),
            (1.00, 0.55, 0.03),
            (1.00, 0.94, 0.18),
            (0.18, 1.00, 0.38),
            (0.02, 0.94, 1.00),
            (0.72, 0.96, 1.00),
        ),
        (0.0015, 0.0015, 0.003),
        {"point_brightness": 1.6, "glow_intensity": 1.7},
    ),
    "Birdsong Spectral": VisualPreset(
        "Birdsong Spectral",
        (
            (0.86, 0.05, 0.88),
            (1.00, 0.08, 0.18),
            (1.00, 0.54, 0.04),
            (0.96, 0.94, 0.20),
            (0.14, 1.00, 0.50),
            (0.00, 0.92, 1.00),
            (0.86, 0.98, 1.00),
        ),
        (0.001, 0.0015, 0.002),
        {"frequency_min": 300.0, "frequency_max": 16000.0},
    ),
    "Scientific Neon": VisualPreset(
        "Scientific Neon",
        (
            (1.00, 0.02, 0.72),
            (0.42, 0.12, 1.00),
            (0.00, 0.74, 1.00),
            (0.00, 1.00, 0.58),
            (0.98, 1.00, 0.35),
            (1.00, 1.00, 1.00),
        ),
        (0.001, 0.002, 0.004),
        {"line_brightness": 1.05, "grain_intensity": 0.03},
    ),
    "Archival Gold": VisualPreset(
        "Archival Gold",
        (
            (0.55, 0.08, 0.025),
            (0.95, 0.28, 0.06),
            (1.00, 0.68, 0.20),
            (0.94, 0.94, 0.78),
        ),
        (0.010, 0.008, 0.006),
        {
            "grain_intensity": 0.14,
            "trail_persistence": 0.86,
            "glow_intensity": 0.78,
        },
    ),
    "Monochrome Laboratory": VisualPreset(
        "Monochrome Laboratory",
        (
            (0.26, 0.34, 0.29),
            (0.65, 0.76, 0.68),
            (0.93, 0.96, 0.91),
            (0.72, 0.82, 0.74),
        ),
        (0.005, 0.009, 0.006),
        {
            "grain_intensity": 0.16,
            "scanline_intensity": 0.09,
            "flicker_intensity": 0.065,
        },
    ),
    "Deep Space": VisualPreset(
        "Deep Space",
        (
            (0.20, 0.025, 0.05),
            (0.12, 0.18, 0.54),
            (0.48, 0.24, 0.83),
            (0.48, 0.76, 1.00),
            (0.90, 0.94, 1.00),
        ),
        (0.003, 0.003, 0.012),
        {
            "fog_intensity": 0.48,
            "rotation_speed_x": 0.07,
            "rotation_speed_y": 0.12,
            "rotation_speed_z": 0.045,
            "trail_persistence": 0.88,
        },
    ),
}

DEFAULT_PRESET = "Full Spectral Rainbow"
