"""Faint model-space reference grid for reading acoustic coordinates."""

from __future__ import annotations

import math

import numpy as np


def grid_vertices(
    center: np.ndarray,
    radius: float,
    spacing: float,
    opacity: float,
) -> np.ndarray:
    spacing = max(0.1, float(spacing))
    half_count = min(20, max(4, int(math.ceil(radius * 1.22 / spacing))))
    extent = half_count * spacing
    neutral = np.asarray((0.36, 0.39, 0.42), dtype=np.float32)
    rows: list[np.ndarray] = []

    def line(first, second, alpha: float) -> None:
        rows.append(
            np.concatenate(
                (
                    np.asarray(first, dtype=np.float32),
                    neutral,
                    np.asarray([alpha], dtype=np.float32),
                )
            )
        )
        rows.append(
            np.concatenate(
                (
                    np.asarray(second, dtype=np.float32),
                    neutral,
                    np.asarray([alpha], dtype=np.float32),
                )
            )
        )

    cx, cy, cz = (float(value) for value in center)
    for index in range(-half_count, half_count + 1):
        offset = index * spacing
        strength = opacity * (1.75 if index == 0 else 1.0)
        line(
            (cx - extent, cy, cz + offset),
            (cx + extent, cy, cz + offset),
            strength,
        )
        line(
            (cx + offset, cy, cz - extent),
            (cx + offset, cy, cz + extent),
            strength,
        )
        if index % 3 == 0:
            line(
                (cx + offset, cy - extent * 0.55, cz),
                (cx + offset, cy + extent * 0.55, cz),
                strength * 0.72,
            )
    return np.asarray(rows, dtype=np.float32)
