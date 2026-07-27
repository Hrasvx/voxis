"""Uniform 3D grid used for bounded-radius neighbor searches."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from typing import Protocol

import numpy as np


class Positioned(Protocol):
    id: int
    position: np.ndarray


NEIGHBOR_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


class SpatialHash:
    def __init__(self, cell_size: float) -> None:
        self.cell_size = max(1e-5, cell_size)
        self.cells: dict[tuple[int, int, int], list[Positioned]] = defaultdict(list)
        self.locations: dict[int, tuple[int, int, int]] = {}

    def rebuild(self, points) -> None:
        self.cells.clear()
        self.locations.clear()
        for point in points:
            self.insert(point)

    def insert(self, point: Positioned) -> None:
        cell = self.cell(point.position)
        self.cells[cell].append(point)
        self.locations[point.id] = cell

    def remove(self, point_id: int) -> None:
        cell = self.locations.pop(point_id, None)
        if cell is None:
            return
        values = self.cells.get(cell)
        if values is None:
            return
        self.cells[cell] = [point for point in values if point.id != point_id]
        if not self.cells[cell]:
            self.cells.pop(cell, None)

    def neighbors(self, point: Positioned):
        origin = self.cell(point.position)
        for offset in NEIGHBOR_OFFSETS:
            cell = (
                origin[0] + offset[0],
                origin[1] + offset[1],
                origin[2] + offset[2],
            )
            yield from self.cells.get(cell, ())

    def neighbors_limited(self, point: Positioned, per_cell: int = 10):
        """Yield a bounded recent sample from each adjacent spatial bucket."""
        origin = self.cell(point.position)
        count = max(1, per_cell)
        for offset in NEIGHBOR_OFFSETS:
            cell = (
                origin[0] + offset[0],
                origin[1] + offset[1],
                origin[2] + offset[2],
            )
            values = self.cells.get(cell, ())
            yield from values[-count:]

    def cell(self, position: np.ndarray) -> tuple[int, int, int]:
        values = [
            int(math.floor(float(value) / self.cell_size)) for value in position
        ]
        return values[0], values[1], values[2]
