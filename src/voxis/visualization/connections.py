"""Deterministic, degree-limited point-network generation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .spatial_index import SpatialHash


class ConnectablePoint(Protocol):
    id: int
    position: np.ndarray
    frequency_norm: float
    connections: set[int]


def update_network(
    points,
    radius: float,
    maximum_connections: int,
    frequency_similarity: float,
    activity: float,
) -> int:
    """Prune invalid edges and add spatially/frequency-related network edges.

    Candidate discovery is O(n + local-neighborhood edges) through a uniform
    grid. Edge ordering is deterministic, and degrees are strictly capped.
    """
    point_list = list(points)
    by_id = {point.id: point for point in point_list}
    if radius <= 0.0 or maximum_connections <= 0:
        for point in point_list:
            point.connections.clear()
        return 0

    radius_sq = radius * radius
    for point in point_list:
        for other_id in tuple(point.connections):
            other = by_id.get(other_id)
            invalid = (
                other is None
                or point.id not in other.connections
                or float(np.dot(point.position - other.position, point.position - other.position))
                > radius_sq * 1.56
            )
            if invalid:
                point.connections.discard(other_id)
                if other is not None:
                    other.connections.discard(point.id)
        if len(point.connections) > maximum_connections:
            for other_id in sorted(point.connections)[maximum_connections:]:
                point.connections.discard(other_id)
                if other_id in by_id:
                    by_id[other_id].connections.discard(point.id)

    grid = SpatialHash(radius)
    grid.rebuild(point_list)
    candidates: list[tuple[float, int, int]] = []
    similarity_weight = float(np.clip(frequency_similarity, 0.0, 1.0))
    for point in sorted(point_list, key=lambda item: item.id):
        for other in grid.neighbors(point):
            if other.id <= point.id or other.id in point.connections:
                continue
            delta = point.position - other.position
            distance_sq = float(np.dot(delta, delta))
            if distance_sq > radius_sq:
                continue
            frequency_delta = abs(point.frequency_norm - other.frequency_norm)
            score = (distance_sq / radius_sq) * (1.0 - similarity_weight)
            score += frequency_delta * similarity_weight
            candidates.append((score, point.id, other.id))

    candidates.sort()
    activity = float(np.clip(activity, 0.0, 1.0))
    desired_degree = min(
        maximum_connections,
        max(1, int(round(maximum_connections * (0.30 + activity * 0.70)))),
    )
    added = 0
    for score, first_id, second_id in candidates:
        first = by_id[first_id]
        second = by_id[second_id]
        if (
            len(first.connections) >= desired_degree
            or len(second.connections) >= desired_degree
        ):
            continue
        if score > 0.34 + activity * 0.54:
            continue
        first.connections.add(second.id)
        second.connections.add(first.id)
        added += 1
    return added


def validate_network(points, maximum_connections: int) -> bool:
    by_id = {point.id: point for point in points}
    for point in by_id.values():
        if len(point.connections) > maximum_connections:
            return False
        if point.id in point.connections:
            return False
        for neighbor_id in point.connections:
            neighbor = by_id.get(neighbor_id)
            if neighbor is None or point.id not in neighbor.connections:
                return False
    return True


def edge_set(points) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for point in points:
        for other_id in point.connections:
            edges.add((min(point.id, other_id), max(point.id, other_id)))
    return edges
