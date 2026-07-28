"""Deterministic multi-peak cloud generation, tracking, and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..config import VisualizationSettings
from ..models import AnalysisFrame, SpectralPeak
from ..presets import VisualPreset
from .spatial_index import SpatialHash


EDGE_TEMPORAL = "temporal"
EDGE_INTRA = "intra"
EDGE_SIMILARITY = "similarity"
NEUTRAL_LINE_COLOR = np.asarray((0.34, 0.36, 0.38), dtype=np.float32)


@dataclass(slots=True)
class VisualEdge:
    first_id: int
    second_id: int
    kind: str
    creation_time: float
    strength: float


@dataclass(slots=True)
class VisualPoint:
    id: int
    acoustic_position: np.ndarray
    base_position: np.ndarray
    position: np.ndarray
    color: np.ndarray
    size: float
    frequency_hz: float
    frequency_norm: float
    frequency_band: int
    amplitude: float
    magnitude_db: float
    prominence_db: float
    snr_db: float
    bandwidth_hz: float
    phase: float
    stereo_balance: float
    harmonicity: float
    spectral_centroid_hz: float
    centroid_norm: float
    spectral_contrast: float
    timbre: float
    context_position: np.ndarray
    phrase_id: int
    frame_timestamp: float
    audio_energy: float
    lifetime: float
    creation_time: float
    importance: float
    age: float = 0.0
    alpha: float = 0.0
    brightness: float = 1.0
    connections: set[int] = field(default_factory=set)
    noise_phase: float = 0.0


class PointManager:
    """Converts each selected spectral peak into one stable 3D network node."""

    def __init__(self, seed: int = 24301) -> None:
        self.seed = seed
        self.points: dict[int, VisualPoint] = {}
        self.edges: dict[tuple[int, int], VisualEdge] = {}
        self._next_id = 1
        self._visual_time = 0.0
        self._last_activity = 0.0
        self._previous_frame_ids: list[int] = []
        self._previous_frame_time = -1.0
        self._previous_phrase_id = -1
        self._similarity_grid: SpatialHash | None = None
        self._grid_radius = -1.0
        self._similarity_dirty = False
        self._last_similarity_bucket = -1
        self._last_node_frame_time = -1e9
        self._last_node_bucket = -1

    @property
    def visual_time(self) -> float:
        return self._visual_time

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def clear(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.points.clear()
        self.edges.clear()
        self._next_id = 1
        self._visual_time = 0.0
        self._last_activity = 0.0
        self._previous_frame_ids = []
        self._previous_frame_time = -1.0
        self._previous_phrase_id = -1
        self._similarity_grid = None
        self._grid_radius = -1.0
        self._similarity_dirty = False
        self._last_similarity_bucket = -1
        self._last_node_frame_time = -1e9
        self._last_node_bucket = -1

    def ingest(
        self,
        frame: AnalysisFrame,
        settings: VisualizationSettings,
        preset: VisualPreset,
        density_scale: float = 1.0,
    ) -> int:
        peaks = (
            frame.peaks
            if frame.peaks or frame.context_features.size
            else _fallback_peaks(frame, settings)
        )
        if not peaks or frame.rms_db < settings.silence_threshold_db:
            self._previous_frame_ids = []
            self._previous_frame_time = frame.timestamp_s
            self._previous_phrase_id = frame.phrase_id
            return 0
        node_bucket = int(
            math.floor((frame.timestamp_s + 1e-9) / settings.node_interval_s)
        )
        if (
            not frame.phrase_boundary
            and node_bucket == self._last_node_bucket
        ):
            return 0
        self._last_node_frame_time = frame.timestamp_s
        self._last_node_bucket = node_bucket
        limit = settings.visible_point_limit
        self._evict_to_limit(max(0, limit - len(peaks)))
        available = max(0, limit - len(self.points))
        quality_fraction = 1.0 if density_scale >= 0.999 else max(0.2, density_scale)
        count = min(len(peaks), available)
        if quality_fraction < 1.0:
            count = max(1, round(count * quality_fraction))
        selected = _evenly_ranked_peaks(peaks, count)
        new_points: list[VisualPoint] = []
        for peak_index, peak in selected:
            point = self._create_point(
                peak, peak_index, frame, settings, preset
            )
            while point.id in self.points:
                point.id += 1
            self.points[point.id] = point
            new_points.append(point)

        self._last_activity = float(
            np.clip(
                0.45 * (frame.onset_strength + frame.spectral_flatness)
                + 0.55 * len(peaks) / max(1, settings.max_peaks_per_frame),
                0.0,
                1.0,
            )
        )
        contiguous = (
            self._previous_frame_ids
            and frame.phrase_id == self._previous_phrase_id
            and frame.timestamp_s - self._previous_frame_time
            <= max(settings.node_interval_s * 1.8, frame.duration_s * 3.2)
        )
        if contiguous:
            self._connect_temporal(new_points, settings, frame.timestamp_s)
        self._connect_intra(new_points, settings, frame.timestamp_s)
        self._similarity_dirty = True
        self._previous_frame_ids = [point.id for point in new_points]
        self._previous_frame_time = frame.timestamp_s
        self._previous_phrase_id = frame.phrase_id
        self._trim_edges(settings.visible_line_limit)
        return len(new_points)

    def reconstruct(
        self,
        frames,
        timestamp_s: float,
        settings: VisualizationSettings,
        preset: VisualPreset,
    ) -> None:
        """Rebuild the exact retained state after a seek."""
        self.clear(settings.random_seed)
        history_start = (
            0.0
            if settings.persistent_history
            else max(0.0, timestamp_s - settings.historical_duration)
        )
        if not settings.persistent_history:
            history_start = min(
                history_start,
                max(
                    0.0,
                    math.floor(timestamp_s / 2.0) * 2.0
                    - settings.historical_duration,
                ),
            )
            history_start = max(
                0.0, history_start - settings.node_interval_s * 1.25
            )
        for frame in frames:
            if frame.timestamp_s < history_start - 1e-6:
                continue
            if frame.timestamp_s > timestamp_s:
                break
            self.ingest(frame, settings, preset)
            self.update_at(frame.timestamp_s, settings)
        self.update_at(timestamp_s, settings)
        self.finalize_graph(timestamp_s, settings)

    def finalize_graph(
        self, timestamp_s: float, settings: VisualizationSettings
    ) -> None:
        self._similarity_dirty = True
        self._refresh_similarity_if_due(timestamp_s, settings)

    def update(self, delta_s: float, settings: VisualizationSettings) -> None:
        if delta_s > 0.0:
            self.update_at(self._visual_time + delta_s, settings)

    def update_at(
        self, media_time_s: float, settings: VisualizationSettings
    ) -> None:
        self._visual_time = max(0.0, media_time_s)
        expired: list[int] = []
        total_lifetime = (
            float("inf")
            if settings.persistent_history
            else settings.historical_duration
        )
        for point in self.points.values():
            point.age = max(0.0, self._visual_time - point.creation_time)
            if point.age > total_lifetime:
                expired.append(point.id)
                continue
            settle_duration = max(settings.node_settle_duration, 1e-6)
            birth = _settle_envelope(point.age / settle_duration)
            appear_duration = min(0.14, settle_duration * 0.22)
            appear = _smoothstep(
                0.0,
                1.0,
                point.age / max(appear_duration, 1e-6),
            )
            if point.age <= settings.active_duration:
                active_progress = point.age / max(settings.active_duration, 1e-6)
                alpha = 0.98 - active_progress * 0.18
                brightness = (
                    1.04
                    + settings.new_node_glow_boost
                    * birth
                    * (0.82 + point.audio_energy * 0.28)
                    + point.audio_energy * 0.22
                )
            else:
                history_progress = (
                    0.0
                    if settings.persistent_history
                    else (point.age - settings.active_duration)
                    / max(
                        1e-6,
                        settings.historical_duration - settings.active_duration,
                    )
                )
                fade_envelope = max(
                    0.04,
                    (1.0 - float(np.clip(history_progress, 0.0, 1.0)))
                    ** settings.fade_speed,
                )
                alpha = settings.historical_opacity * fade_envelope
                brightness = 0.92 + point.audio_energy * 0.12
            flicker = 1.0 + settings.flicker_intensity * math.sin(
                self._visual_time * 9.7 + point.noise_phase
            )
            point.alpha = appear * max(0.0, alpha)
            point.brightness = brightness * flicker
            point.base_position = _visual_position(
                point.acoustic_position,
                settings,
            )
            point.position = point.base_position
        self._remove_points(expired)
        if not settings.persistent_history:
            cutoff = self._visual_time - settings.edge_lifetime
            self._remove_edges(
                [
                    key
                    for key, edge in self.edges.items()
                    if edge.creation_time < cutoff
                ]
            )
        self._evict_phrase_limit(settings.maximum_retained_phrases)
        self._refresh_similarity_if_due(self._visual_time, settings)
        self._enforce_connection_limits(settings)
        self._trim_edges(settings.visible_line_limit)

    def bounds(self) -> tuple[np.ndarray, float]:
        if not self.points:
            return np.zeros(3, dtype=np.float32), 2.0
        positions = np.asarray(
            [point.position for point in self.points.values()], dtype=np.float32
        )
        minimum = positions.min(axis=0)
        maximum = positions.max(axis=0)
        center = (minimum + maximum) * 0.5
        radius = max(0.55, float(np.max(maximum - minimum) * 0.54))
        return center, radius

    def camera_focus(
        self,
        hold_duration: float,
        return_duration: float,
    ) -> tuple[np.ndarray | None, float, float]:
        if not self.points:
            return None, 0.0, 0.55
        eligible = [
            point
            for point in self.points.values()
            if point.creation_time <= self._visual_time + 1e-6
        ]
        if not eligible:
            return None, 0.0, 0.55
        latest_time = max(point.creation_time for point in eligible)
        latest = [
            point
            for point in eligible
            if abs(point.creation_time - latest_time) <= 1e-6
        ]
        if not latest:
            return None, 0.0, 0.55
        age = max(0.0, self._visual_time - latest_time)
        if age <= hold_duration:
            envelope = 1.0
        else:
            progress = (age - hold_duration) / max(return_duration, 1e-6)
            envelope = 1.0 - _smoothstep(0.0, 1.0, progress)
        if envelope <= 0.0:
            return None, 0.0, 0.55
        weights = np.asarray(
            [max(0.05, point.importance) ** 2 for point in latest],
            dtype=np.float32,
        )
        positions = np.asarray(
            [point.position for point in latest],
            dtype=np.float32,
        )
        center = np.average(positions, axis=0, weights=weights).astype(np.float32)
        local_radius = max(
            0.55,
            float(np.max(np.linalg.norm(positions - center, axis=1)))
            if len(positions) > 1
            else 0.55,
        )
        return center, float(envelope), local_radius

    def important_points(
        self,
        percentage: float,
        maximum_count: int,
        active_duration: float = 1.0,
    ) -> list[VisualPoint]:
        visible = [
            point
            for point in self.points.values()
            if point.alpha > 0.02
        ]
        count = min(
            maximum_count,
            max(1 if visible and percentage > 0.0 else 0, round(len(visible) * percentage)),
        )
        if count == 0:
            return []
        return sorted(
            visible,
            key=lambda point: (
                -point.importance,
                -point.magnitude_db,
                point.id,
            ),
        )[:count]

    def vertex_arrays(
        self,
        settings: VisualizationSettings,
        render_fraction: float = 1.0,
        quality: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return (
                np.empty((0, 8), dtype=np.float32),
                np.empty((0, 7), dtype=np.float32),
            )
        fraction = float(np.clip(render_fraction, 0.05, 1.0))
        candidates = sorted(
            self.points.values(),
            key=lambda point: (
                point.age > settings.active_duration,
                -point.importance,
                -point.creation_time,
                point.id,
            ),
        )
        active_quality = quality or settings.preview_quality
        render_limit = min(settings.visible_point_limit, len(candidates))
        if active_quality == "Medium":
            render_limit = min(render_limit, 4000)
        visible_points = [
            point
            for point in candidates
            if fraction >= 1.0
            or ((point.id * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF <= fraction
        ][:render_limit]
        if not visible_points:
            visible_points = [candidates[0]]
        visible_ids = {point.id for point in visible_points}
        point_rows = np.empty((len(visible_points), 8), dtype=np.float32)
        for row, point in enumerate(visible_points):
            point_rows[row, :3] = point.position
            point_rows[row, 3:6] = (
                point.color * point.brightness * settings.point_brightness
            )
            point_rows[row, 6] = point.alpha
            bloom_settle = 1.0 + settings.new_node_size_boost * _settle_envelope(
                point.age / max(settings.node_settle_duration, 1e-6)
            )
            point_rows[row, 7] = (
                point.size * settings.point_size * bloom_settle
            )

        line_limit = settings.visible_line_limit
        if active_quality == "Medium":
            line_limit = min(line_limit, 15000)
        if settings.spiderweb:
            lines = _spiderweb_lines(visible_points, line_limit, settings)
        else:
            kind_order = {
                EDGE_TEMPORAL: 0,
                EDGE_INTRA: 1,
                EDGE_SIMILARITY: 2,
            }
            visible_edges = [
                edge
                for edge in self.edges.values()
                if edge.first_id in visible_ids and edge.second_id in visible_ids
            ]
            visible_edges.sort(
                key=lambda edge: (
                    kind_order[edge.kind],
                    -edge.creation_time,
                    edge.first_id,
                    edge.second_id,
                )
            )
            lines = _typed_edge_lines(
                visible_edges[:line_limit],
                self.points,
                settings,
            )
        return point_rows, lines

    def _create_point(
        self,
        peak: SpectralPeak,
        peak_index: int,
        frame: AnalysisFrame,
        settings: VisualizationSettings,
        preset: VisualPreset,
    ) -> VisualPoint:
        frequency_norm = _log_frequency_norm(
            peak.frequency_hz, settings.frequency_min, settings.frequency_max
        )
        palette_position = 1.0 - frequency_norm if settings.reverse_palette else frequency_norm
        band_index = min(
            len(frame.band_energy) - 1,
            max(0, int(frequency_norm * len(frame.band_energy))),
        )
        stable_seed = _acoustic_seed(self.seed, peak_index, peak, frame)
        rng = np.random.default_rng(stable_seed)
        context = np.asarray(frame.context_position, dtype=np.float32)
        centroid_norm = _log_frequency_norm(
            max(frame.spectral_centroid_hz, settings.frequency_min),
            settings.frequency_min,
            settings.frequency_max,
        )
        phrase_centroid_norm = _log_frequency_norm(
            max(
                frame.phrase_centroid_hz
                if frame.phrase_centroid_hz > 0.0
                else frame.spectral_centroid_hz,
                settings.frequency_min,
            ),
            settings.frequency_min,
            settings.frequency_max,
        )
        frame_timbre = _timbre_descriptor(
            frame.harmonicity,
            frame.spectral_contrast,
            frame.spectral_flatness,
        )
        peak_timbre = float(
            np.clip(frame_timbre * 0.72 + peak.harmonicity * 0.28, 0.0, 1.0)
        )
        phrase_timbre = (
            frame.phrase_timbre
            if frame.phrase_timbre >= 0.0
            else frame_timbre
        )
        brightness_axis = centroid_norm * 0.82 + phrase_centroid_norm * 0.18
        timbre_axis = peak_timbre * 0.82 + phrase_timbre * 0.18
        position = np.asarray(
            (
                (brightness_axis - 0.5) * 4.4,
                (frequency_norm - 0.5) * 5.0,
                (timbre_axis - 0.5) * 4.2,
            ),
            dtype=np.float32,
        )
        band_influence = float(
            np.interp(
                frequency_norm,
                (0.0, 0.5, 1.0),
                (
                    settings.low_frequency_influence,
                    settings.mid_frequency_influence,
                    settings.high_frequency_influence,
                ),
            )
        )
        acoustic_position = position
        position = _visual_position(acoustic_position, settings)
        db_strength = float(
            np.clip((peak.magnitude_db + 72.0) / 66.0, 0.0, 1.0)
        )
        amplitude_strength = float(
            np.clip(peak.amplitude * 0.55 + db_strength * 0.45, 0.0, 1.0)
        )
        band_gain = float(np.clip(band_influence, 0.0, 3.0))
        importance = float(
            np.clip(
                (
                    amplitude_strength * 0.68
                    + peak.prominence_db / 40.0 * 0.20
                    + peak.snr_db / 50.0 * 0.12
                    + frame.onset_strength
                    * settings.transient_sensitivity
                    * 0.08
                )
                * (0.72 + band_gain * 0.28),
                0.0,
                1.5,
            )
        )
        point = VisualPoint(
            id=_stable_node_id(frame.timestamp_s, peak_index),
            acoustic_position=acoustic_position.copy(),
            base_position=position.copy(),
            position=position,
            color=_palette_color(preset, palette_position),
            size=float(
                settings.point_core_size
                * (0.52 + amplitude_strength * 1.32)
            ),
            frequency_hz=peak.frequency_hz,
            frequency_norm=frequency_norm,
            frequency_band=band_index,
            amplitude=peak.amplitude,
            magnitude_db=peak.magnitude_db,
            prominence_db=peak.prominence_db,
            snr_db=peak.snr_db,
            bandwidth_hz=peak.bandwidth_hz,
            phase=peak.phase,
            stereo_balance=peak.stereo_balance,
            harmonicity=peak.harmonicity,
            spectral_centroid_hz=frame.spectral_centroid_hz,
            centroid_norm=centroid_norm,
            spectral_contrast=frame.spectral_contrast,
            timbre=timbre_axis,
            context_position=context.copy(),
            phrase_id=frame.phrase_id,
            frame_timestamp=frame.timestamp_s,
            audio_energy=amplitude_strength,
            lifetime=settings.historical_duration,
            creation_time=frame.timestamp_s,
            importance=importance,
            noise_phase=float(rng.uniform(0.0, math.tau)),
        )
        self._next_id = max(self._next_id, point.id + 1)
        return point

    def _connect_temporal(
        self,
        new_points: list[VisualPoint],
        settings: VisualizationSettings,
        timestamp: float,
    ) -> None:
        previous = [
            self.points[point_id]
            for point_id in self._previous_frame_ids
            if point_id in self.points
        ]
        candidates: list[tuple[float, int, int]] = []
        for current in new_points:
            for old in previous:
                frequency_distance = abs(
                    math.log2(max(current.frequency_hz, 1.0) / max(old.frequency_hz, 1.0))
                )
                timbre_distance = abs(current.timbre - old.timbre)
                centroid_distance = abs(current.centroid_norm - old.centroid_norm)
                if (
                    frequency_distance > 0.22
                    or timbre_distance > 0.24
                    or centroid_distance > 0.20
                ):
                    continue
                phase_distance = abs(math.atan2(
                    math.sin(current.phase - old.phase),
                    math.cos(current.phase - old.phase),
                )) / math.pi
                score = (
                    frequency_distance * 4.2
                    + abs(current.amplitude - old.amplitude) * 0.52
                    + phase_distance * 0.10
                    + timbre_distance * 0.72
                    + centroid_distance * 0.48
                )
                candidates.append((score, old.id, current.id))
        candidates.sort()
        temporal_limit = 1 if settings.max_temporal_connections > 0 else 0
        per_new: dict[int, int] = {}
        per_old: dict[int, int] = {}
        for score, old_id, new_id in candidates:
            if score > 1.18:
                continue
            if per_new.get(new_id, 0) >= temporal_limit:
                continue
            if per_old.get(old_id, 0) >= temporal_limit:
                continue
            if self._add_edge(
                old_id,
                new_id,
                EDGE_TEMPORAL,
                timestamp,
                settings.temporal_edge_strength,
                settings.max_connections_per_point,
            ):
                per_new[new_id] = per_new.get(new_id, 0) + 1
                per_old[old_id] = per_old.get(old_id, 0) + 1

    def _connect_intra(
        self,
        points: list[VisualPoint],
        settings: VisualizationSettings,
        timestamp: float,
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for first_index, first in enumerate(points):
            for second in points[first_index + 1 :]:
                ratio = max(first.frequency_hz, second.frequency_hz) / max(
                    1.0, min(first.frequency_hz, second.frequency_hz)
                )
                harmonic_error = _harmonic_relation_error(ratio)
                if harmonic_error > 0.055:
                    continue
                score = harmonic_error * 5.0
                score += abs(first.amplitude - second.amplitude) * 0.18
                score += abs(first.timbre - second.timbre) * 0.24
                candidates.append((score, first.id, second.id))
        candidates.sort()
        local_degree: dict[int, int] = {}
        for _, first_id, second_id in candidates:
            if (
                local_degree.get(first_id, 0) >= settings.max_intra_connections
                or local_degree.get(second_id, 0) >= settings.max_intra_connections
            ):
                continue
            if self._add_edge(
                first_id,
                second_id,
                EDGE_INTRA,
                timestamp,
                settings.intra_frame_edge_strength,
                settings.max_connections_per_point,
            ):
                local_degree[first_id] = local_degree.get(first_id, 0) + 1
                local_degree[second_id] = local_degree.get(second_id, 0) + 1

    def _refresh_similarity_if_due(
        self,
        timestamp: float,
        settings: VisualizationSettings,
    ) -> None:
        if settings.spiderweb:
            self._similarity_dirty = True
            self._last_similarity_bucket = -1
            return
        bucket = int(max(0.0, timestamp) / 2.0)
        if not self._similarity_dirty or bucket == self._last_similarity_bucket:
            return
        self._last_similarity_bucket = bucket
        self._similarity_dirty = False
        self._remove_edges(
            [key for key, edge in self.edges.items() if edge.kind == EDGE_SIMILARITY]
        )
        if settings.max_similarity_connections <= 0 or not self.points:
            return
        if settings.connection_radius <= 0.0:
            return
        radius = min(1.6, max(0.2, settings.connection_radius))
        self._similarity_grid = None
        self._grid_radius = radius
        similarity_degree: dict[int, int] = {}
        by_phrase: dict[int, list[VisualPoint]] = {}
        for point in self.points.values():
            by_phrase.setdefault(point.phrase_id, []).append(point)
        for phrase_id in sorted(by_phrase):
            grid = SpatialHash(radius)
            phrase_points = sorted(
                by_phrase[phrase_id],
                key=lambda item: (item.creation_time, item.id),
            )
            for point in phrase_points:
                candidates: list[tuple[float, int]] = []
                for candidate_index, other in enumerate(
                    grid.neighbors_limited(point, per_cell=8)
                ):
                    if candidate_index >= 56:
                        break
                    if abs(
                        other.frame_timestamp - point.frame_timestamp
                    ) < max(0.18, settings.node_interval_s * 1.6):
                        continue
                    distance = float(
                        np.linalg.norm(
                            point.acoustic_position - other.acoustic_position
                        )
                    )
                    if distance > radius:
                        continue
                    frequency_octaves = abs(
                        math.log2(
                            max(point.frequency_hz, 1.0)
                            / max(other.frequency_hz, 1.0)
                        )
                    )
                    centroid_distance = abs(
                        point.centroid_norm - other.centroid_norm
                    )
                    timbre_distance = abs(point.timbre - other.timbre)
                    amplitude_distance = abs(
                        point.amplitude - other.amplitude
                    )
                    if (
                        frequency_octaves > 0.28
                        or centroid_distance > 0.16
                        or timbre_distance > 0.20
                        or amplitude_distance > 0.34
                    ):
                        continue
                    frequency_weight = (
                        0.35
                        + settings.frequency_similarity_influence * 0.45
                    )
                    score = (
                        frequency_octaves / 0.28 * frequency_weight
                        + centroid_distance / 0.16 * 0.16
                        + timbre_distance / 0.20 * 0.24
                        + amplitude_distance / 0.34 * 0.10
                    )
                    candidates.append((score, other.id))
                candidates.sort()
                added = 0
                for score, other_id in candidates:
                    if score > 0.78:
                        continue
                    if (
                        similarity_degree.get(other_id, 0)
                        >= settings.max_similarity_connections
                    ):
                        continue
                    if self._add_edge(
                        point.id,
                        other_id,
                        EDGE_SIMILARITY,
                        timestamp,
                        settings.similarity_edge_strength,
                        settings.max_connections_per_point,
                    ):
                        added += 1
                        similarity_degree[point.id] = (
                            similarity_degree.get(point.id, 0) + 1
                        )
                        similarity_degree[other_id] = (
                            similarity_degree.get(other_id, 0) + 1
                        )
                    if added >= settings.max_similarity_connections:
                        break
                grid.insert(point)

    def _add_edge(
        self,
        first_id: int,
        second_id: int,
        kind: str,
        timestamp: float,
        strength: float,
        maximum_degree: int,
    ) -> bool:
        if first_id == second_id:
            return False
        first = self.points.get(first_id)
        second = self.points.get(second_id)
        if first is None or second is None:
            return False
        if first.phrase_id != second.phrase_id:
            return False
        key = (min(first_id, second_id), max(first_id, second_id))
        if key in self.edges:
            return False
        if not self._make_room(first_id, kind, maximum_degree):
            return False
        if not self._make_room(second_id, kind, maximum_degree):
            return False
        self.edges[key] = VisualEdge(
            key[0], key[1], kind, timestamp, float(np.clip(strength, 0.0, 1.0))
        )
        first.connections.add(second.id)
        second.connections.add(first.id)
        return True

    def _make_room(
        self, point_id: int, new_kind: str, maximum_degree: int
    ) -> bool:
        point = self.points[point_id]
        if len(point.connections) < maximum_degree:
            return True
        replaceable = {EDGE_TEMPORAL: {EDGE_SIMILARITY, EDGE_INTRA},
                       EDGE_INTRA: {EDGE_SIMILARITY},
                       EDGE_SIMILARITY: set()}[new_kind]
        candidates = [
            (key, edge)
            for key, edge in self.edges.items()
            if edge.kind in replaceable
            and (edge.first_id == point_id or edge.second_id == point_id)
        ]
        if not candidates:
            return False
        key, _ = min(
            candidates,
            key=lambda item: (
                item[1].strength,
                item[1].creation_time,
                item[0],
            ),
        )
        self._remove_edges([key])
        return len(point.connections) < maximum_degree

    def _trim_edges(self, limit: int) -> None:
        if len(self.edges) <= limit:
            return
        order = {EDGE_SIMILARITY: 0, EDGE_INTRA: 1, EDGE_TEMPORAL: 2}
        excess = len(self.edges) - limit
        victims = sorted(
            self.edges,
            key=lambda key: (
                order[self.edges[key].kind],
                self.edges[key].creation_time,
                key,
            ),
        )[:excess]
        self._remove_edges(victims)

    def _enforce_connection_limits(
        self, settings: VisualizationSettings
    ) -> None:
        kind_limits = {
            EDGE_TEMPORAL: settings.max_temporal_connections,
            EDGE_INTRA: settings.max_intra_connections,
            EDGE_SIMILARITY: settings.max_similarity_connections,
        }
        retain_order = {
            EDGE_TEMPORAL: 0,
            EDGE_INTRA: 1,
            EDGE_SIMILARITY: 2,
        }
        victims: set[tuple[int, int]] = set()
        incident_by_point: dict[int, list[tuple[tuple[int, int], VisualEdge]]] = {
            point_id: [] for point_id in self.points
        }
        for key, edge in self.edges.items():
            incident_by_point.get(edge.first_id, []).append((key, edge))
            incident_by_point.get(edge.second_id, []).append((key, edge))
        for point in sorted(self.points.values(), key=lambda item: item.id):
            incident = incident_by_point.get(point.id, [])
            for kind, limit in kind_limits.items():
                matching = sorted(
                    (
                        (key, edge)
                        for key, edge in incident
                        if edge.kind == kind and key not in victims
                    ),
                    key=lambda item: (
                        -item[1].strength,
                        -item[1].creation_time,
                        item[0],
                    ),
                )
                victims.update(key for key, _ in matching[limit:])
            retained = sorted(
                (
                    (key, edge)
                    for key, edge in incident
                    if key not in victims
                ),
                key=lambda item: (
                    retain_order[item[1].kind],
                    -item[1].strength,
                    -item[1].creation_time,
                    item[0],
                ),
            )
            victims.update(
                key
                for key, _ in retained[settings.max_connections_per_point :]
            )
        self._remove_edges(victims)

    def _evict_to_limit(self, target_size: int) -> None:
        excess = len(self.points) - max(0, target_size)
        if excess <= 0:
            return
        victims = list(self.points)[:excess]
        self._remove_points(victims)

    def _evict_phrase_limit(self, maximum_phrases: int) -> None:
        phrases = sorted({point.phrase_id for point in self.points.values()})
        if len(phrases) <= maximum_phrases:
            return
        rejected = set(phrases[: len(phrases) - maximum_phrases])
        self._remove_points(
            [point.id for point in self.points.values() if point.phrase_id in rejected]
        )

    def _expire_at(
        self, timestamp: float, settings: VisualizationSettings
    ) -> None:
        if settings.persistent_history:
            return
        cutoff = timestamp - settings.historical_duration
        if cutoff <= 0.0:
            return
        expired: list[int] = []
        for point in self.points.values():
            if point.creation_time >= cutoff:
                break
            expired.append(point.id)
        self._remove_points(expired)

    def _remove_points(self, point_ids) -> None:
        rejected = set(point_ids)
        if not rejected:
            return
        self._similarity_dirty = True
        edge_keys: set[tuple[int, int]] = set()
        for point_id in rejected:
            point = self.points.get(point_id)
            if point is None:
                continue
            edge_keys.update(
                (min(point_id, other_id), max(point_id, other_id))
                for other_id in point.connections
            )
        self._remove_edges(edge_keys)
        for point_id in rejected:
            if self._similarity_grid is not None:
                self._similarity_grid.remove(point_id)
            self.points.pop(point_id, None)

    def _remove_edges(self, edge_keys) -> None:
        for key in edge_keys:
            edge = self.edges.pop(key, None)
            if edge is None:
                continue
            if edge.first_id in self.points:
                self.points[edge.first_id].connections.discard(edge.second_id)
            if edge.second_id in self.points:
                self.points[edge.second_id].connections.discard(edge.first_id)


def _typed_edge_lines(
    edges: list[VisualEdge],
    points: dict[int, VisualPoint],
    settings: VisualizationSettings,
) -> np.ndarray:
    rows = np.empty((len(edges) * 2, 7), dtype=np.float32)
    written = 0
    for edge in edges:
        first = points.get(edge.first_id)
        second = points.get(edge.second_id)
        if first is None or second is None:
            continue
        alpha = (
            min(first.alpha, second.alpha)
            * settings.connection_opacity
            * edge.strength
        )
        _write_line(rows, written, first, second, alpha, settings.line_brightness)
        written += 2
    return rows[:written]


def _spiderweb_lines(
    points: list[VisualPoint],
    limit: int,
    settings: VisualizationSettings,
) -> np.ndarray:
    point_count = len(points)
    if point_count < 2 or limit <= 0:
        return np.empty((0, 7), dtype=np.float32)
    edge_count = min(limit, point_count * (point_count - 1) // 2)
    rows = np.empty((edge_count * 2, 7), dtype=np.float32)
    strength = float(np.clip(1.4 / math.sqrt(point_count), 0.035, 0.22))
    written_edges = 0
    for gap in range(1, point_count):
        for first_index in range(point_count - gap):
            first = points[first_index]
            second = points[first_index + gap]
            alpha = (
                min(first.alpha, second.alpha)
                * settings.connection_opacity
                * strength
            )
            _write_line(
                rows,
                written_edges * 2,
                first,
                second,
                alpha,
                settings.line_brightness,
            )
            written_edges += 1
            if written_edges >= edge_count:
                return rows
    return rows[: written_edges * 2]


def _write_line(
    rows: np.ndarray,
    row: int,
    first: VisualPoint,
    second: VisualPoint,
    alpha: float,
    brightness: float,
) -> None:
    rows[row, :3] = first.position
    rows[row, 3:6] = (
        NEUTRAL_LINE_COLOR * 0.82 + first.color * 0.18
    ) * brightness
    rows[row, 6] = alpha
    rows[row + 1, :3] = second.position
    rows[row + 1, 3:6] = (
        NEUTRAL_LINE_COLOR * 0.82 + second.color * 0.18
    ) * brightness
    rows[row + 1, 6] = alpha


def _fallback_peaks(
    frame: AnalysisFrame, settings: VisualizationSettings
) -> tuple[SpectralPeak, ...]:
    """Compatibility path for cached/tests frames created before peak metadata."""
    if not frame.spectrum.size:
        return ()
    candidates = np.flatnonzero(
        (frame.spectrum[1:-1] > frame.spectrum[:-2])
        & (frame.spectrum[1:-1] >= frame.spectrum[2:])
    ) + 1
    if not candidates.size:
        candidates = np.asarray([int(np.argmax(frame.spectrum))])
    ranked = sorted(
        candidates,
        key=lambda index: (-float(frame.spectrum[index]), int(index)),
    )[: settings.max_peaks_per_frame]
    return tuple(
        SpectralPeak(
            frequency_hz=float(frame.frequencies[index]),
            amplitude=float(frame.spectrum[index]),
            magnitude_db=float(frame.spectrum[index] * 80.0 - 80.0),
            prominence_db=8.0,
            snr_db=12.0,
            bandwidth_hz=max(10.0, float(frame.frequencies[index]) * 0.02),
            phase=float(index * 1.61803398875),
            stereo_balance=frame.stereo_balance,
            harmonicity=frame.harmonicity,
        )
        for index in sorted(ranked)
    )


def _evenly_ranked_peaks(
    peaks: tuple[SpectralPeak, ...], count: int
) -> list[tuple[int, SpectralPeak]]:
    ranked = sorted(
        enumerate(peaks),
        key=lambda item: (
            -(item[1].snr_db + item[1].prominence_db * 0.5),
            item[1].frequency_hz,
        ),
    )[:count]
    return sorted(ranked, key=lambda item: item[0])


def _palette_color(preset: VisualPreset, position: float) -> np.ndarray:
    colors = preset.colors
    scaled = float(np.clip(position, 0.0, 1.0)) * (len(colors) - 1)
    lower = min(len(colors) - 1, int(math.floor(scaled)))
    upper = min(len(colors) - 1, lower + 1)
    mix = scaled - lower
    return (
        np.asarray(colors[lower], dtype=np.float32) * (1.0 - mix)
        + np.asarray(colors[upper], dtype=np.float32) * mix
    )


def _log_frequency_norm(frequency: float, minimum: float, maximum: float) -> float:
    return float(
        np.clip(
            math.log(max(frequency, minimum) / minimum)
            / max(1e-9, math.log(maximum / minimum)),
            0.0,
            1.0,
        )
    )


def _visual_position(
    acoustic_position: np.ndarray,
    settings: VisualizationSettings,
) -> np.ndarray:
    position = np.asarray(acoustic_position, dtype=np.float32).copy()
    position *= settings.spatial_spread / 1.55
    position[0] *= settings.horizontal_spacing
    position[1] *= settings.frequency_spacing
    return position


def _timbre_descriptor(
    harmonicity: float,
    spectral_contrast: float,
    spectral_flatness: float,
) -> float:
    return float(
        np.clip(
            harmonicity * 0.45
            + spectral_contrast * 0.35
            + (1.0 - spectral_flatness) * 0.20,
            0.0,
            1.0,
        )
    )


def _harmonic_relation_error(ratio: float) -> float:
    relationships = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
    return min(abs(math.log2(max(ratio, 1e-9) / target)) for target in relationships)


def _acoustic_seed(
    seed: int,
    peak_index: int,
    peak: SpectralPeak,
    frame: AnalysisFrame,
) -> int:
    values = (
        round(peak.frequency_hz * 100.0),
        round(peak.amplitude * 100_000.0),
        round(peak.bandwidth_hz * 100.0),
        round(peak.phase * 100_000.0),
        round(peak.stereo_balance * 100_000.0),
        round(peak.harmonicity * 100_000.0),
        round(frame.rms_db * 1_000.0),
        round(frame.dominant_hz * 100.0),
        round(frame.spectral_centroid_hz * 100.0),
        *(
            round(float(component) * 100_000.0)
            for component in np.asarray(frame.context_position).ravel()
        ),
    )
    value = int(seed) * 0x9E3779B1 ^ peak_index * 0x27D4EB2F
    for component in values:
        value ^= int(component) * 0x85EBCA77
        value = (value * 0xC2B2AE3D) & 0xFFFFFFFFFFFFFFFF
    return value & 0xFFFFFFFFFFFFFFFF


def _stable_node_id(timestamp: float, peak_index: int) -> int:
    return max(1, round(timestamp * 1_000_000) * 64 + peak_index + 1)


def _smoothstep(low: float, high: float, value: float) -> float:
    x = float(np.clip((value - low) / max(high - low, 1e-9), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _settle_envelope(progress: float) -> float:
    x = float(np.clip(progress, 0.0, 1.0))
    smoother = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    return 1.0 - smoother
