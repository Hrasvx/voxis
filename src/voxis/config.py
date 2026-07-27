"""Runtime settings, validation, migration, and JSON persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import threading
from typing import Any


@dataclass(slots=True)
class VisualizationSettings:
    audio_sensitivity: float = 1.20
    frequency_min: float = 300.0
    frequency_max: float = 16000.0
    fft_size: int = 4096
    hop_size: int = 256
    audio_smoothing: float = 0.48
    transient_sensitivity: float = 1.15
    min_peaks_per_frame: int = 1
    max_peaks_per_frame: int = 8
    peak_prominence_db: float = 8.0
    spectral_noise_floor_db: float = 11.0
    silence_threshold_db: float = -62.0
    phrase_detection_sensitivity: float = 1.0
    low_frequency_influence: float = 1.20
    mid_frequency_influence: float = 1.00
    high_frequency_influence: float = 1.15
    analysis_lookahead_ms: int = 350

    point_size: float = 1.0
    point_core_size: float = 1.3
    white_core_intensity: float = 1.2
    new_node_size_boost: float = 1.3
    new_node_glow_boost: float = 2.4
    node_settle_duration: float = 1.0
    max_active_points: int = 2500
    visible_point_limit: int = 2500
    point_lifetime: float = 45.0
    active_duration: float = 1.0
    historical_duration: float = 45.0
    historical_opacity: float = 0.5
    persistent_history: bool = False
    maximum_retained_phrases: int = 32
    node_interval_s: float = 0.12
    fade_speed: float = 1.0
    spatial_spread: float = 1.55
    point_brightness: float = 1.6

    connection_radius: float = 1.4
    max_connections_per_point: int = 6
    max_temporal_connections: int = 1
    max_intra_connections: int = 2
    max_similarity_connections: int = 2
    visible_line_limit: int = 20000
    edge_lifetime: float = 45.0
    line_thickness: float = 0.75
    line_brightness: float = 1.85
    connection_opacity: float = 0.92
    temporal_edge_strength: float = 0.55
    intra_frame_edge_strength: float = 0.38
    similarity_edge_strength: float = 0.26
    frequency_similarity_influence: float = 0.72

    rotation_speed_x: float = 0.052
    rotation_speed_y: float = 0.094
    rotation_speed_z: float = 0.031
    camera_distance: float = 9.0
    camera_min_distance: float = 3.0
    camera_max_distance: float = 30.0
    field_of_view: float = 47.0
    camera_drift: float = 0.04
    fog_intensity: float = 0.26
    automatic_rotation: bool = True
    auto_fit_camera: bool = True
    viewport_occupancy: float = 1.4
    camera_smoothing: float = 0.12
    camera_follow_new_nodes: bool = True
    camera_follow_strength: float = 0.72
    camera_follow_smoothing: float = 0.48
    camera_follow_max_speed: float = 6.0
    camera_focus_hold_duration: float = 0.45
    camera_return_duration: float = 1.2
    grid_enabled: bool = True
    grid_spacing: float = 1.5
    grid_opacity: float = 0.07

    glow_intensity: float = 1.7
    bloom_intensity: float = 0.55
    grain_intensity: float = 0.045
    flicker_intensity: float = 0.025
    trail_persistence: float = 0.72
    scanline_intensity: float = 0.02
    chromatic_aberration: float = 0.04
    background_brightness: float = 0.32

    palette: str = "Full Spectral Rainbow"
    reverse_palette: bool = False
    scientific_labels: bool = True
    scientific_labels_export: bool = True
    label_percentage: float = 0.08
    label_max_count: int = 200
    label_box_size: float = 7.0
    label_text_size: int = 9
    label_min_opacity: float = 0.68
    frequency_legend_preview: bool = True
    frequency_legend_export: bool = False
    preview_quality: str = "High"
    export_quality: str = "High"

    random_seed: int = 24301
    performance_limit_fps: int = 60
    schema_version: int = 8

    def validated(self) -> "VisualizationSettings":
        self.audio_sensitivity = _clamp(self.audio_sensitivity, 0.05, 8.0)
        self.frequency_min = _clamp(self.frequency_min, 10.0, 18000.0)
        self.frequency_max = _clamp(
            self.frequency_max, self.frequency_min + 10.0, 24000.0
        )
        allowed_fft = (512, 1024, 2048, 4096, 8192)
        self.fft_size = min(allowed_fft, key=lambda x: abs(x - int(self.fft_size)))
        allowed_hop = (64, 128, 256, 512, 1024, 2048)
        self.hop_size = min(
            (value for value in allowed_hop if value <= self.fft_size),
            key=lambda x: abs(x - int(self.hop_size)),
        )
        self.audio_smoothing = _clamp(self.audio_smoothing, 0.0, 0.98)
        self.transient_sensitivity = _clamp(self.transient_sensitivity, 0.0, 5.0)
        self.min_peaks_per_frame = int(_clamp(self.min_peaks_per_frame, 0, 20))
        self.max_peaks_per_frame = int(
            _clamp(self.max_peaks_per_frame, self.min_peaks_per_frame, 40)
        )
        self.peak_prominence_db = _clamp(self.peak_prominence_db, 0.5, 30.0)
        self.spectral_noise_floor_db = _clamp(
            self.spectral_noise_floor_db, 0.0, 40.0
        )
        self.silence_threshold_db = _clamp(
            self.silence_threshold_db, -100.0, -20.0
        )
        self.phrase_detection_sensitivity = _clamp(
            self.phrase_detection_sensitivity, 0.1, 4.0
        )
        self.low_frequency_influence = _clamp(
            self.low_frequency_influence, 0.0, 3.0
        )
        self.mid_frequency_influence = _clamp(
            self.mid_frequency_influence, 0.0, 3.0
        )
        self.high_frequency_influence = _clamp(
            self.high_frequency_influence, 0.0, 3.0
        )
        self.analysis_lookahead_ms = int(
            _clamp(self.analysis_lookahead_ms, 50, 500)
        )

        self.point_size = _clamp(self.point_size, 0.25, 3.0)
        self.point_core_size = _clamp(self.point_core_size, 1.0, 6.0)
        self.white_core_intensity = _clamp(
            self.white_core_intensity, 0.0, 2.5
        )
        self.new_node_size_boost = _clamp(
            self.new_node_size_boost, 0.0, 2.0
        )
        self.new_node_glow_boost = _clamp(
            self.new_node_glow_boost, 0.0, 3.0
        )
        self.node_settle_duration = _clamp(
            self.node_settle_duration, 0.1, 5.0
        )
        self.max_active_points = int(_clamp(self.max_active_points, 64, 25000))
        self.visible_point_limit = int(
            _clamp(self.visible_point_limit, 256, 25000)
        )
        self.point_lifetime = _clamp(self.point_lifetime, 0.25, 60.0)
        self.active_duration = _clamp(self.active_duration, 0.25, 8.0)
        self.historical_duration = _clamp(
            self.historical_duration, self.active_duration, 60.0
        )
        self.historical_opacity = _clamp(self.historical_opacity, 0.0, 0.8)
        self.maximum_retained_phrases = int(
            _clamp(self.maximum_retained_phrases, 1, 128)
        )
        self.node_interval_s = _clamp(self.node_interval_s, 0.02, 1.0)
        self.fade_speed = _clamp(self.fade_speed, 0.1, 5.0)
        self.spatial_spread = _clamp(self.spatial_spread, 0.2, 4.0)
        self.point_brightness = _clamp(self.point_brightness, 0.1, 3.0)

        self.connection_radius = _clamp(self.connection_radius, 0.0, 1.6)
        self.max_connections_per_point = int(
            _clamp(self.max_connections_per_point, 0, 10)
        )
        self.max_temporal_connections = int(
            _clamp(self.max_temporal_connections, 0, 1)
        )
        self.max_intra_connections = int(
            _clamp(self.max_intra_connections, 0, 4)
        )
        self.max_similarity_connections = int(
            _clamp(self.max_similarity_connections, 0, 4)
        )
        self.visible_line_limit = int(
            _clamp(self.visible_line_limit, 256, 100000)
        )
        self.edge_lifetime = _clamp(
            self.edge_lifetime, self.active_duration, 60.0
        )
        self.line_thickness = _clamp(self.line_thickness, 0.5, 3.0)
        self.line_brightness = _clamp(self.line_brightness, 0.0, 2.0)
        self.connection_opacity = _clamp(self.connection_opacity, 0.0, 1.0)
        self.temporal_edge_strength = _clamp(
            self.temporal_edge_strength, 0.0, 1.0
        )
        self.intra_frame_edge_strength = _clamp(
            self.intra_frame_edge_strength, 0.0, 1.0
        )
        self.similarity_edge_strength = _clamp(
            self.similarity_edge_strength, 0.0, 1.0
        )
        self.frequency_similarity_influence = _clamp(
            self.frequency_similarity_influence, 0.0, 1.0
        )

        self.rotation_speed_x = _clamp(self.rotation_speed_x, -1.5, 1.5)
        self.rotation_speed_y = _clamp(self.rotation_speed_y, -1.5, 1.5)
        self.rotation_speed_z = _clamp(self.rotation_speed_z, -1.5, 1.5)
        self.camera_distance = _clamp(self.camera_distance, 3.0, 30.0)
        self.camera_min_distance = _clamp(self.camera_min_distance, 1.5, 20.0)
        self.camera_max_distance = _clamp(
            self.camera_max_distance, self.camera_min_distance + 1.0, 60.0
        )
        self.field_of_view = _clamp(self.field_of_view, 25.0, 90.0)
        self.camera_drift = _clamp(self.camera_drift, 0.0, 1.0)
        self.fog_intensity = _clamp(self.fog_intensity, 0.0, 1.0)
        self.viewport_occupancy = _clamp(self.viewport_occupancy, 0.50, 1.60)
        self.camera_smoothing = _clamp(self.camera_smoothing, 0.01, 1.0)
        self.camera_follow_strength = _clamp(
            self.camera_follow_strength, 0.0, 1.0
        )
        self.camera_follow_smoothing = _clamp(
            self.camera_follow_smoothing, 0.01, 1.0
        )
        self.camera_follow_max_speed = _clamp(
            self.camera_follow_max_speed, 0.1, 30.0
        )
        self.camera_focus_hold_duration = _clamp(
            self.camera_focus_hold_duration, 0.0, 5.0
        )
        self.camera_return_duration = _clamp(
            self.camera_return_duration, 0.1, 10.0
        )
        self.grid_spacing = _clamp(self.grid_spacing, 0.25, 5.0)
        self.grid_opacity = _clamp(self.grid_opacity, 0.0, 0.30)

        self.glow_intensity = _clamp(self.glow_intensity, 0.0, 2.5)
        self.bloom_intensity = _clamp(self.bloom_intensity, 0.0, 1.5)
        self.grain_intensity = _clamp(self.grain_intensity, 0.0, 0.5)
        self.flicker_intensity = _clamp(self.flicker_intensity, 0.0, 0.3)
        self.trail_persistence = _clamp(self.trail_persistence, 0.0, 0.96)
        self.scanline_intensity = _clamp(self.scanline_intensity, 0.0, 0.3)
        self.chromatic_aberration = _clamp(
            self.chromatic_aberration, 0.0, 1.0
        )
        self.background_brightness = _clamp(
            self.background_brightness, 0.0, 1.5
        )
        self.label_percentage = _clamp(self.label_percentage, 0.0, 1.0)
        self.label_max_count = int(_clamp(self.label_max_count, 1, 2000))
        self.label_box_size = _clamp(self.label_box_size, 2.0, 24.0)
        self.label_text_size = int(_clamp(self.label_text_size, 6, 24))
        self.label_min_opacity = _clamp(self.label_min_opacity, 0.05, 1.0)
        palettes = {
            "Full Spectral Rainbow",
            "Birdsong Spectral",
            "Scientific Neon",
            "Archival Gold",
            "Monochrome Laboratory",
            "Deep Space",
        }
        if self.palette not in palettes:
            self.palette = "Full Spectral Rainbow"
        if self.preview_quality not in {"Medium", "High"}:
            self.preview_quality = "High"
        if self.export_quality not in {"Medium", "High"}:
            self.export_quality = "High"
        self.random_seed = int(_clamp(self.random_seed, 0, 2_147_483_647))
        self.performance_limit_fps = int(
            _clamp(self.performance_limit_fps, 24, 144)
        )
        self.schema_version = 8
        return self

    def update(self, name: str, value: Any) -> None:
        known = {item.name for item in fields(self)}
        if name not in known:
            raise KeyError(f"Unknown setting: {name}")
        current = getattr(self, name)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, str):
            value = str(value)
        elif isinstance(current, int):
            value = int(value)
        else:
            value = float(value)
        setattr(self, name, value)
        self.validated()

    def copy(self) -> "VisualizationSettings":
        return VisualizationSettings(**asdict(self))


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "voxis" / "config.json"
LEGACY_CONFIG_PATHS = (
    Path.home() / ".config" / "auralattice" / "config.json",
    Path.home() / ".config" / "resonant-tumbler" / "config.json",
)

LEGACY_KEYS = {
    "connection_distance": "connection_radius",
    "rotation_speed": "rotation_speed_y",
}

V1_FACTORY_MIGRATIONS = {
    "audio_sensitivity": (1.30, 1.20),
    "frequency_min": (35.0, 300.0),
    "fft_size": (2048, 4096),
    "audio_smoothing": (0.72, 0.48),
    "high_frequency_influence": (0.90, 1.15),
    "analysis_lookahead_ms": (180, 350),
    "max_active_points": (3200, 6000),
    "point_lifetime": (6.5, 18.0),
    "spatial_spread": (1.0, 1.35),
    "point_brightness": (1.0, 1.45),
    "connection_radius": (0.78, 1.05),
    "max_connections_per_point": (3, 8),
    "line_thickness": (1.0, 0.75),
    "line_brightness": (0.75, 0.90),
    "connection_opacity": (0.32, 0.72),
    "rotation_speed_x": (0.13, 0.052),
    "rotation_speed_y": (0.20, 0.094),
    "rotation_speed_z": (0.08, 0.031),
    "camera_distance": (8.0, 9.0),
    "camera_drift": (0.08, 0.04),
    "fog_intensity": (0.32, 0.26),
    "glow_intensity": (0.72, 0.95),
    "bloom_intensity": (0.24, 0.34),
    "grain_intensity": (0.10, 0.045),
    "flicker_intensity": (0.05, 0.025),
    "trail_persistence": (0.83, 0.72),
    "scanline_intensity": (0.05, 0.02),
    "chromatic_aberration": (0.12, 0.04),
    "background_brightness": (0.55, 0.32),
}

V2_FACTORY_MIGRATIONS = {
    "min_peaks_per_frame": (6, 1),
    "max_peaks_per_frame": (28, 8),
    "peak_prominence_db": (5.0, 8.0),
    "spectral_noise_floor_db": (7.0, 11.0),
    "phrase_spatial_separation": (2.4, 1.6),
    "point_core_size": (2.0, 2.4),
    "max_active_points": (6000, 2500),
    "visible_point_limit": (6000, 2500),
    "point_lifetime": (18.0, 30.0),
    "active_duration": (1.5, 1.0),
    "historical_duration": (18.0, 30.0),
    "historical_opacity": (0.16, 0.48),
    "max_connections_per_point": (8, 4),
    "max_temporal_connections": (2, 1),
    "max_intra_connections": (4, 2),
    "max_similarity_connections": (2, 1),
    "visible_line_limit": (25000, 8000),
    "edge_lifetime": (18.0, 30.0),
    "connection_opacity": (0.72, 0.58),
    "temporal_edge_strength": (0.52, 0.62),
    "intra_frame_edge_strength": (0.26, 0.22),
    "similarity_edge_strength": (0.12, 0.10),
}

V3_FACTORY_MIGRATIONS = {
    "line_thickness": (0.75, 1.10),
    "glow_intensity": (0.95, 1.25),
    "scientific_labels_export": (False, True),
}

V4_FACTORY_MIGRATIONS = {
    "phrase_spatial_separation": (1.6, 0.9),
    "point_core_size": (2.4, 1.3),
    "white_core_intensity": (1.35, 1.2),
    "new_node_size_boost": (0.75, 1.3),
    "new_node_glow_boost": (0.95, 2.4),
    "point_lifetime": (30.0, 45.0),
    "historical_duration": (30.0, 45.0),
    "historical_opacity": (0.48, 0.5),
    "maximum_retained_phrases": (24, 32),
    "spatial_spread": (1.35, 1.55),
    "point_brightness": (1.45, 1.6),
    "connection_radius": (1.05, 3.5),
    "max_connections_per_point": (4, 10),
    "max_intra_connections": (2, 4),
    "max_similarity_connections": (1, 4),
    "visible_line_limit": (8000, 20000),
    "edge_lifetime": (30.0, 45.0),
    "line_thickness": (1.10, 0.75),
    "line_brightness": (0.90, 1.85),
    "connection_opacity": (0.58, 0.92),
    "temporal_edge_strength": (0.62, 0.55),
    "intra_frame_edge_strength": (0.22, 0.38),
    "similarity_edge_strength": (0.10, 0.26),
    "viewport_occupancy": (0.75, 1.4),
    "grid_spacing": (1.0, 1.5),
    "grid_opacity": (0.07, 0.07),
    "glow_intensity": (1.25, 1.7),
    "bloom_intensity": (0.34, 0.55),
    "label_box_size": (12.0, 7.0),
    "label_min_opacity": (0.72, 0.68),
}

V7_FACTORY_MIGRATIONS = {
    "connection_radius": (3.5, 1.4),
    "max_connections_per_point": (10, 6),
    "max_intra_connections": (4, 2),
    "max_similarity_connections": (4, 2),
}


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> VisualizationSettings:
    settings = VisualizationSettings()
    if path == DEFAULT_CONFIG_PATH and not path.is_file():
        path = next(
            (
                candidate
                for candidate in LEGACY_CONFIG_PATHS
                if candidate.is_file()
            ),
            path,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return settings
    except (OSError, json.JSONDecodeError):
        return settings
    for old, new in LEGACY_KEYS.items():
        if old in payload and new not in payload:
            payload[new] = payload[old]
    try:
        schema_version = int(payload.get("schema_version", 1))
    except (TypeError, ValueError):
        schema_version = 1
    if schema_version < 2:
        for name, (old_value, new_value) in V1_FACTORY_MIGRATIONS.items():
            if payload.get(name) == old_value:
                payload[name] = new_value
        schema_version = 2
    if schema_version < 3:
        for name, (old_value, new_value) in V2_FACTORY_MIGRATIONS.items():
            if payload.get(name) == old_value:
                payload[name] = new_value
        schema_version = 3
    if schema_version < 4:
        for name, (old_value, new_value) in V3_FACTORY_MIGRATIONS.items():
            if payload.get(name) == old_value:
                payload[name] = new_value
        schema_version = 4
    if schema_version < 5:
        for name, (_, new_value) in V4_FACTORY_MIGRATIONS.items():
            payload[name] = new_value
        payload["schema_version"] = 5
        schema_version = 5
    if schema_version < 6:
        payload.pop("phrase_spatial_separation", None)
        payload["schema_version"] = 6
        schema_version = 6
    if schema_version < 8:
        for name, (old_value, new_value) in V7_FACTORY_MIGRATIONS.items():
            if payload.get(name, old_value) == old_value:
                payload[name] = new_value
        payload.pop("movement_speed", None)
        payload.pop("point_generation_rate", None)
        payload["schema_version"] = 8
    if "max_active_points" in payload and "visible_point_limit" not in payload:
        payload["visible_point_limit"] = max(6000, int(payload["max_active_points"]))
    for field in fields(settings):
        if field.name in payload:
            try:
                settings.update(field.name, payload[field.name])
            except (TypeError, ValueError):
                continue
    return settings.validated()


def save_settings(
    settings: VisualizationSettings, path: Path = DEFAULT_CONFIG_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(settings.validated()), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SettingsState:
    """Thread-safe mutable settings shared by preview and export setup."""

    def __init__(self, settings: VisualizationSettings | None = None) -> None:
        self._settings = settings or VisualizationSettings()
        self._lock = threading.RLock()

    def snapshot(self) -> VisualizationSettings:
        with self._lock:
            return self._settings.copy()

    def update(self, name: str, value: Any) -> VisualizationSettings:
        with self._lock:
            self._settings.update(name, value)
            return self._settings.copy()

    def replace(self, settings: VisualizationSettings) -> None:
        with self._lock:
            self._settings = settings.validated()
