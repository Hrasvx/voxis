from dataclasses import dataclass, field

import numpy as np

from voxis.visualization.connections import (
    edge_set,
    update_network,
    validate_network,
)


@dataclass
class Point:
    id: int
    position: np.ndarray
    frequency_norm: float
    connections: set[int] = field(default_factory=set)


def test_network_allows_branches_and_triangles() -> None:
    points = [
        Point(0, np.array([0.00, 0.00, 0.0]), 0.30),
        Point(1, np.array([0.18, 0.00, 0.0]), 0.31),
        Point(2, np.array([0.09, 0.16, 0.0]), 0.32),
        Point(3, np.array([0.09, 0.06, 0.14]), 0.33),
    ]

    update_network(points, 0.5, 3, 0.7, activity=1.0)
    edges = edge_set(points)

    assert validate_network(points, 3)
    assert any(len(point.connections) > 1 for point in points)
    assert {(0, 1), (0, 2), (1, 2)}.issubset(edges)


def test_network_never_exceeds_configured_degree() -> None:
    rng = np.random.default_rng(42)
    points = [
        Point(index, rng.normal(0.0, 0.3, 3), float(rng.random()))
        for index in range(800)
    ]

    update_network(points, 0.7, 4, 0.65, activity=1.0)

    assert validate_network(points, 4)
    assert max(len(point.connections) for point in points) <= 4


def test_edges_are_pruned_when_points_move_apart() -> None:
    points = [
        Point(0, np.array([0.0, 0.0, 0.0]), 0.4),
        Point(1, np.array([0.1, 0.0, 0.0]), 0.4),
    ]
    update_network(points, 0.5, 2, 1.0, activity=1.0)
    assert edge_set(points) == {(0, 1)}

    points[1].position = np.array([4.0, 0.0, 0.0])
    update_network(points, 0.5, 2, 1.0, activity=1.0)
    assert edge_set(points) == set()
