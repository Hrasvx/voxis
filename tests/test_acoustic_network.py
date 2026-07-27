import json
from dataclasses import replace

import numpy as np

from voxis.analysis.audio_analyzer import AudioAnalyzer, finalize_context_embedding
from voxis.config import VisualizationSettings, load_settings
from voxis.models import AnalysisFrame, AudioChunk, SpectralPeak
from voxis.presets import PRESETS
from voxis.visualization.points import (
    EDGE_INTRA,
    EDGE_SIMILARITY,
    EDGE_TEMPORAL,
    PointManager,
)
from voxis.ui.settings_panel import SECTIONS
from voxis.visualization.camera import Camera


def peak(frequency: float, phase: float = 0.0, amplitude: float = 0.8) -> SpectralPeak:
    return SpectralPeak(
        frequency_hz=frequency,
        amplitude=amplitude,
        magnitude_db=-12.0,
        prominence_db=14.0,
        snr_db=24.0,
        bandwidth_hz=45.0,
        phase=phase,
        stereo_balance=0.1,
        harmonicity=0.8,
    )


def frame(
    timestamp: float,
    phrase_id: int = 0,
    frequencies: tuple[float, ...] = (800.0, 1200.0, 1600.0, 2400.0),
) -> AnalysisFrame:
    return AnalysisFrame(
        timestamp_s=timestamp,
        duration_s=256 / 48000,
        spectrum=np.linspace(0.1, 0.8, 32, dtype=np.float32),
        frequencies=np.geomspace(300.0, 16000.0, 32).astype(np.float32),
        band_energy=np.linspace(0.1, 0.8, 24, dtype=np.float32),
        rms_db=-18.0,
        peak_db=-6.0,
        dominant_hz=frequencies[0],
        stereo_balance=0.1,
        phase_correlation=0.7,
        spectral_centroid_hz=2800.0,
        spectral_bandwidth_hz=1700.0,
        spectral_rolloff_hz=7200.0,
        onset_strength=0.35,
        peaks=tuple(
            peak(frequency, index * 0.31)
            for index, frequency in enumerate(frequencies)
        ),
        context_features=np.linspace(-0.5, 0.5, 36, dtype=np.float32),
        context_position=np.asarray((0.15, -0.2, 0.25), dtype=np.float32),
        spectral_flatness=0.25,
        harmonicity=0.75,
        phrase_id=phrase_id,
        phrase_boundary=timestamp == 0.0,
    )


def test_stft_detects_multiple_peaks_and_silence_creates_none() -> None:
    sample_rate = 48000
    size = 4096
    time = np.arange(size) / sample_rate
    signal = sum(
        np.sin(2.0 * np.pi * frequency * time) * 0.18
        for frequency in (700.0, 1200.0, 2300.0, 5100.0, 8700.0)
    ).astype(np.float32)
    analyzer = AudioAnalyzer(
        size,
        0.3,
        300.0,
        12000.0,
        minimum_peaks=0,
        maximum_peaks=20,
        peak_prominence_db=3.0,
        noise_floor_db=4.0,
        hop_size=256,
    )

    active = analyzer.analyze(
        AudioChunk(np.vstack((signal, signal)), 0.0, sample_rate), 0
    )
    silent = analyzer.analyze(
        AudioChunk(np.zeros((2, size), dtype=np.float32), 0.1, sample_rate), 0
    )

    detected = [item.frequency_hz for item in active.peaks]
    assert len(detected) >= 4
    assert all(any(abs(value - expected) < 30.0 for value in detected) for expected in (700, 1200, 2300, 5100))
    assert silent.peaks == ()


def test_phrase_detector_starts_a_new_structure_after_sustained_silence() -> None:
    sample_rate = 48000
    size = 4096
    hop = 256
    time = np.arange(size) / sample_rate
    tone = (0.4 * np.sin(2.0 * np.pi * 2200.0 * time)).astype(np.float32)
    active = np.vstack((tone, tone))
    silence = np.zeros((2, size), dtype=np.float32)
    analyzer = AudioAnalyzer(
        size,
        0.3,
        300.0,
        12000.0,
        hop_size=hop,
        phrase_sensitivity=1.0,
    )

    first = analyzer.analyze(AudioChunk(active, 0.0, sample_rate), 0)
    for index in range(24):
        analyzer.analyze(
            AudioChunk(silence, (index + 1) * hop / sample_rate, sample_rate),
            0,
        )
    resumed = analyzer.analyze(
        AudioChunk(active, 25 * hop / sample_rate, sample_rate), 0
    )

    assert first.phrase_boundary
    assert resumed.phrase_boundary
    assert resumed.phrase_id == first.phrase_id + 1


def test_peak_tracking_creates_temporal_and_intra_frame_edges() -> None:
    settings = VisualizationSettings(
        visible_point_limit=1000,
        visible_line_limit=5000,
        node_interval_s=0.001,
    )
    manager = PointManager(42)
    preset = PRESETS["Full Spectral Rainbow"]
    manager.ingest(frame(0.10), settings, preset)
    manager.ingest(frame(0.105, frequencies=(805.0, 1194.0, 1610.0, 2388.0)), settings, preset)

    kinds = {edge.kind for edge in manager.edges.values()}
    assert EDGE_TEMPORAL in kinds
    assert EDGE_INTRA in kinds


def test_phrase_and_timestamp_do_not_move_equal_sounds() -> None:
    settings = VisualizationSettings()
    first = PointManager(17)
    second = PointManager(17)
    preset = PRESETS["Full Spectral Rainbow"]
    first.ingest(frame(0.0, phrase_id=0), settings, preset)
    second.ingest(frame(18.0, phrase_id=9), settings, preset)

    first_positions = np.asarray(
        [point.base_position for point in first.points.values()]
    )
    second_positions = np.asarray(
        [point.base_position for point in second.points.values()]
    )

    np.testing.assert_array_equal(first_positions, second_positions)


def test_fixed_acoustic_axes_move_only_for_their_descriptors() -> None:
    settings = VisualizationSettings(
        low_frequency_influence=1.0,
        mid_frequency_influence=1.0,
        high_frequency_influence=1.0,
    )
    preset = PRESETS[settings.palette]
    baseline = PointManager(21)
    brighter = PointManager(21)
    higher_frequency = PointManager(21)
    different_timbre = PointManager(21)
    baseline.ingest(
        replace(
            frame(0.0, frequencies=(1200.0,)),
            spectral_centroid_hz=1800.0,
            harmonicity=0.2,
            spectral_contrast=0.2,
            spectral_flatness=0.7,
        ),
        settings,
        preset,
    )
    brighter.ingest(
        replace(
            frame(2.0, frequencies=(1200.0,)),
            spectral_centroid_hz=7200.0,
            harmonicity=0.2,
            spectral_contrast=0.2,
            spectral_flatness=0.7,
        ),
        settings,
        preset,
    )
    higher_frequency.ingest(
        replace(
            frame(4.0, frequencies=(4800.0,)),
            spectral_centroid_hz=1800.0,
            harmonicity=0.2,
            spectral_contrast=0.2,
            spectral_flatness=0.7,
        ),
        settings,
        preset,
    )
    different_timbre.ingest(
        replace(
            frame(6.0, frequencies=(1200.0,)),
            spectral_centroid_hz=1800.0,
            harmonicity=0.95,
            spectral_contrast=0.9,
            spectral_flatness=0.05,
        ),
        settings,
        preset,
    )

    base = next(iter(baseline.points.values())).base_position
    bright = next(iter(brighter.points.values())).base_position
    high = next(iter(higher_frequency.points.values())).base_position
    timbre = next(iter(different_timbre.points.values())).base_position

    assert bright[0] > base[0]
    np.testing.assert_array_equal(bright[1:], base[1:])
    assert high[1] > base[1]
    np.testing.assert_array_equal(high[[0, 2]], base[[0, 2]])
    assert timbre[2] > base[2]
    np.testing.assert_array_equal(timbre[:2], base[:2])


def test_visual_points_retain_peak_and_frame_descriptors() -> None:
    settings = VisualizationSettings()
    source = replace(
        frame(0.4, frequencies=(3200.0,)),
        spectral_centroid_hz=4100.0,
        spectral_contrast=0.63,
    )
    manager = PointManager(25)
    manager.ingest(source, settings, PRESETS[settings.palette])
    point = next(iter(manager.points.values()))
    source_peak = source.peaks[0]

    assert point.frame_timestamp == source.timestamp_s
    assert point.frequency_hz == source_peak.frequency_hz
    assert point.amplitude == source_peak.amplitude
    assert point.magnitude_db == source_peak.magnitude_db
    assert point.prominence_db == source_peak.prominence_db
    assert point.snr_db == source_peak.snr_db
    assert point.bandwidth_hz == source_peak.bandwidth_hz
    assert point.phase == source_peak.phase
    assert point.harmonicity == source_peak.harmonicity
    assert point.spectral_centroid_hz == source.spectral_centroid_hz
    assert point.spectral_contrast == source.spectral_contrast


def test_temporal_peak_tracking_is_one_to_one() -> None:
    settings = VisualizationSettings(
        node_interval_s=0.02,
        max_temporal_connections=2,
        max_connections_per_point=8,
    )
    manager = PointManager(27)
    preset = PRESETS[settings.palette]
    manager.ingest(frame(0.0), settings, preset)
    manager.ingest(
        frame(
            0.03,
            frequencies=(804.0, 1198.0, 1607.0, 2392.0),
        ),
        settings,
        preset,
    )

    temporal_degree: dict[int, int] = {}
    for edge in manager.edges.values():
        if edge.kind != EDGE_TEMPORAL:
            continue
        temporal_degree[edge.first_id] = temporal_degree.get(edge.first_id, 0) + 1
        temporal_degree[edge.second_id] = temporal_degree.get(edge.second_id, 0) + 1

    assert temporal_degree
    assert max(temporal_degree.values()) == 1


def test_intra_frame_edges_require_harmonic_relationships() -> None:
    settings = VisualizationSettings(
        node_interval_s=0.02,
        max_intra_connections=4,
        max_connections_per_point=8,
    )
    manager = PointManager(29)
    manager.ingest(
        frame(0.0, frequencies=(1000.0, 1100.0, 1500.0, 2000.0)),
        settings,
        PRESETS[settings.palette],
    )

    intra = [
        edge for edge in manager.edges.values() if edge.kind == EDGE_INTRA
    ]
    assert intra
    for edge in intra:
        first = manager.points[edge.first_id]
        second = manager.points[edge.second_id]
        ratio = max(first.frequency_hz, second.frequency_hz) / min(
            first.frequency_hz, second.frequency_hz
        )
        assert abs(ratio - 1.1) > 0.02


def test_loudness_changes_appearance_without_moving_acoustic_axes() -> None:
    settings = VisualizationSettings()
    preset = PRESETS[settings.palette]
    quiet = PointManager(31)
    loud = PointManager(31)
    quiet_peaks = tuple(
        replace(item, amplitude=0.2, magnitude_db=-48.0)
        for item in frame(0.0).peaks
    )
    loud_peaks = tuple(
        replace(item, amplitude=0.95, magnitude_db=-4.0)
        for item in frame(0.0).peaks
    )
    quiet.ingest(
        replace(frame(0.0), rms_db=-42.0, peaks=quiet_peaks),
        settings,
        preset,
    )
    loud.ingest(
        replace(frame(12.0, phrase_id=4), rms_db=-8.0, peaks=loud_peaks),
        settings,
        preset,
    )
    quiet_positions = np.asarray(
        [point.base_position for point in quiet.points.values()]
    )
    loud_positions = np.asarray(
        [point.base_position for point in loud.points.values()]
    )

    np.testing.assert_array_equal(quiet_positions, loud_positions)
    assert min(point.size for point in loud.points.values()) > max(
        point.size for point in quiet.points.values()
    )
    assert min(point.audio_energy for point in loud.points.values()) > max(
        point.audio_energy for point in quiet.points.values()
    )


def test_context_embedding_has_no_timeline_smoothing() -> None:
    first_features = np.linspace(-1.0, 1.0, 36, dtype=np.float32)
    frames = [
        replace(frame(0.0), context_features=first_features.copy()),
        replace(
            frame(1.0),
            context_features=np.cos(
                np.linspace(0.0, np.pi, 36)
            ).astype(np.float32),
        ),
        replace(
            frame(2.0),
            context_features=np.sin(
                np.linspace(0.0, np.pi * 2.0, 36)
            ).astype(np.float32),
        ),
        replace(
            frame(18.0, phrase_id=8),
            context_features=first_features.copy(),
        ),
    ]

    finalize_context_embedding(frames)

    np.testing.assert_array_equal(
        frames[0].context_position,
        frames[3].context_position,
    )


def test_phrase_descriptors_create_acoustic_cluster_centers() -> None:
    frames = [
        replace(frame(0.0, phrase_id=0), spectral_centroid_hz=1800.0),
        replace(frame(0.2, phrase_id=0), spectral_centroid_hz=2200.0),
        replace(
            frame(1.0, phrase_id=1),
            spectral_centroid_hz=7600.0,
            harmonicity=0.15,
            spectral_contrast=0.1,
            spectral_flatness=0.8,
        ),
        replace(
            frame(1.2, phrase_id=1),
            spectral_centroid_hz=8400.0,
            harmonicity=0.15,
            spectral_contrast=0.1,
            spectral_flatness=0.8,
        ),
    ]

    finalize_context_embedding(frames)

    assert frames[0].phrase_centroid_hz == frames[1].phrase_centroid_hz
    assert frames[2].phrase_centroid_hz == frames[3].phrase_centroid_hz
    assert frames[2].phrase_centroid_hz > frames[0].phrase_centroid_hz
    assert frames[0].phrase_timbre == frames[1].phrase_timbre
    assert frames[2].phrase_timbre == frames[3].phrase_timbre


def test_node_placement_is_deterministic_and_independent_of_seed() -> None:
    settings = VisualizationSettings(random_seed=991)
    preset = PRESETS["Full Spectral Rainbow"]
    first = PointManager(991)
    second = PointManager(991)
    other = PointManager(992)
    for manager in (first, second, other):
        manager.ingest(frame(0.25), settings, preset)

    first_positions = np.asarray([point.base_position for point in first.points.values()])
    second_positions = np.asarray([point.base_position for point in second.points.values()])
    other_positions = np.asarray([point.base_position for point in other.points.values()])
    np.testing.assert_array_equal(first_positions, second_positions)
    np.testing.assert_array_equal(first_positions, other_positions)
    assert {
        point.noise_phase for point in first.points.values()
    } != {
        point.noise_phase for point in other.points.values()
    }


def test_duplicate_edges_and_count_limits_are_prevented() -> None:
    settings = VisualizationSettings(
        visible_point_limit=24,
        visible_line_limit=18,
        max_connections_per_point=6,
        connection_radius=6.0,
    )
    manager = PointManager(3)
    preset = PRESETS["Full Spectral Rainbow"]
    dense_frequencies = tuple(np.geomspace(500.0, 9000.0, 20))
    for index in range(5):
        manager.ingest(frame(index * 0.01, frequencies=dense_frequencies), settings, preset)

    assert len(manager.points) <= 24
    assert len(manager.edges) <= 18
    assert len(manager.edges) == len(set(manager.edges))
    assert max(len(point.connections) for point in manager.points.values()) <= 6


def test_seeking_reconstruction_matches_uninterrupted_state() -> None:
    settings = VisualizationSettings(
        random_seed=71,
        visible_point_limit=200,
        visible_line_limit=800,
        historical_duration=1.0,
    )
    preset = PRESETS["Full Spectral Rainbow"]
    frames = [frame(index * 0.02, phrase_id=index // 8) for index in range(30)]
    uninterrupted = PointManager(71)
    for item in frames:
        uninterrupted.ingest(item, settings, preset)
        uninterrupted.update_at(item.timestamp_s, settings)
    uninterrupted.update_at(frames[-1].timestamp_s, settings)

    rebuilt = PointManager(71)
    rebuilt.reconstruct(frames, frames[-1].timestamp_s, settings, preset)

    first_points, first_lines = uninterrupted.vertex_arrays(settings)
    second_points, second_lines = rebuilt.vertex_arrays(settings)
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_lines, second_lines)
    assert set(uninterrupted.edges) == set(rebuilt.edges)


def test_seeking_reconstruction_matches_after_history_has_expired() -> None:
    settings = VisualizationSettings(
        random_seed=72,
        visible_point_limit=240,
        visible_line_limit=900,
        historical_duration=0.8,
        edge_lifetime=0.8,
    )
    preset = PRESETS["Full Spectral Rainbow"]
    frames = [frame(index * 0.02, phrase_id=index // 20) for index in range(205)]
    timestamp = frames[-1].timestamp_s
    uninterrupted = PointManager(72)
    for item in frames:
        uninterrupted.ingest(item, settings, preset)
        uninterrupted.update_at(item.timestamp_s, settings)

    rebuilt = PointManager(72)
    rebuilt.reconstruct(frames, timestamp, settings, preset)

    first_points, first_lines = uninterrupted.vertex_arrays(settings)
    second_points, second_lines = rebuilt.vertex_arrays(settings)
    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_lines, second_lines)


def test_similarity_edges_stay_within_a_phrase() -> None:
    settings = VisualizationSettings(
        connection_radius=8.0,
        max_similarity_connections=2,
        max_connections_per_point=8,
    )
    manager = PointManager(9)
    preset = PRESETS["Full Spectral Rainbow"]
    manager.ingest(frame(0.0, phrase_id=0), settings, preset)
    manager.ingest(frame(0.5, phrase_id=0), settings, preset)
    manager.ingest(frame(1.0, phrase_id=1), settings, preset)
    manager.update_at(1.0, settings)

    assert EDGE_SIMILARITY in {edge.kind for edge in manager.edges.values()}
    assert all(
        manager.points[edge.first_id].phrase_id
        == manager.points[edge.second_id].phrase_id
        for edge in manager.edges.values()
    )
    node_count = len(manager.points)
    assert len(manager.edges) < node_count * (node_count - 1) // 2


def test_legacy_factory_defaults_migrate_to_dense_bright_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "fft_size": 2048,
                "frequency_min": 35.0,
                "point_brightness": 1.0,
                "connection_opacity": 0.32,
                "scientific_labels_export": False,
            }
        )
    )

    settings = load_settings(path)

    assert settings.schema_version == 8
    assert settings.fft_size == 4096
    assert settings.frequency_min == 300.0
    assert settings.point_brightness == 1.6
    assert settings.connection_opacity == 0.92
    assert settings.scientific_labels_export


def test_default_node_cadence_prevents_stft_point_flooding() -> None:
    settings = VisualizationSettings()
    manager = PointManager(settings.random_seed)
    preset = PRESETS[settings.palette]
    created = 0
    for index in range(100):
        created += manager.ingest(frame(index * 0.01), settings, preset)

    assert created <= 40


def test_schema_four_visual_profile_is_replaced_once(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "line_thickness": 2.8,
                "point_core_size": 5.5,
                "viewport_occupancy": 0.5,
                "max_peaks_per_frame": 5,
            }
        )
    )

    settings = load_settings(path)

    assert settings.schema_version == 8
    assert settings.line_thickness == VisualizationSettings().line_thickness
    assert settings.point_core_size == VisualizationSettings().point_core_size
    assert settings.viewport_occupancy == VisualizationSettings().viewport_occupancy
    assert settings.max_peaks_per_frame == 5


def test_explicit_frequency_axis_places_higher_peaks_higher() -> None:
    settings = VisualizationSettings(node_interval_s=0.08)
    manager = PointManager(11)
    manager.ingest(
        frame(0.1, frequencies=(500.0, 8000.0)),
        settings,
        PRESETS[settings.palette],
    )
    by_frequency = sorted(manager.points.values(), key=lambda item: item.frequency_hz)

    assert by_frequency[1].base_position[1] > by_frequency[0].base_position[1]


def test_temporal_tracking_is_one_to_one_and_lines_are_neutral() -> None:
    settings = VisualizationSettings(node_interval_s=0.08)
    manager = PointManager(15)
    preset = PRESETS[settings.palette]
    manager.ingest(frame(0.1), settings, preset)
    manager.ingest(frame(0.2), settings, preset)
    manager.update_at(0.2, settings)

    temporal_degree: dict[int, int] = {}
    for edge in manager.edges.values():
        if edge.kind == EDGE_TEMPORAL:
            temporal_degree[edge.first_id] = temporal_degree.get(edge.first_id, 0) + 1
            temporal_degree[edge.second_id] = temporal_degree.get(edge.second_id, 0) + 1
    _, lines = manager.vertex_arrays(settings)
    assert temporal_degree and max(temporal_degree.values()) <= 1
    assert float(np.max(np.ptp(lines[:, 3:6], axis=1))) < 0.50
    assert float(np.max(lines[:, 3:6])) < 1.0


def test_historical_points_keep_their_labels() -> None:
    settings = VisualizationSettings(
        historical_duration=30.0,
        historical_opacity=0.48,
        label_percentage=1.0,
        label_max_count=20,
    )
    manager = PointManager(19)
    manager.ingest(frame(0.0), settings, PRESETS[settings.palette])
    original_ids = set(manager.points)

    manager.update_at(12.0, settings)

    assert {point.id for point in manager.important_points(1.0, 20)} == original_ids
    assert all(point.age == 12.0 for point in manager.points.values())


def test_labels_select_only_the_strongest_retained_points() -> None:
    settings = VisualizationSettings(
        node_interval_s=0.02,
        historical_duration=30.0,
        label_percentage=0.2,
        label_max_count=40,
    )
    manager = PointManager(37)
    preset = PRESETS[settings.palette]
    for index in range(20):
        manager.ingest(frame(index * 0.12), settings, preset)
        manager.update_at(index * 0.12, settings)

    labels = manager.important_points(
        settings.label_percentage,
        settings.label_max_count,
        settings.active_duration,
    )

    expected = sorted(
        manager.points.values(),
        key=lambda point: (
            -point.importance,
            -point.magnitude_db,
            point.id,
        ),
    )[: len(labels)]

    assert [point.id for point in labels] == [point.id for point in expected]


def test_new_node_size_and_glow_settle_are_configurable() -> None:
    settings = VisualizationSettings(
        active_duration=1.0,
        historical_duration=30.0,
        new_node_size_boost=1.5,
        new_node_glow_boost=2.0,
        node_settle_duration=1.0,
    )
    manager = PointManager(23)
    manager.ingest(frame(0.0), settings, PRESETS[settings.palette])
    manager.update_at(0.05, settings)
    new_vertices, _ = manager.vertex_arrays(settings)
    new_brightness = max(point.brightness for point in manager.points.values())

    manager.update_at(1.05, settings)
    settled_vertices, _ = manager.vertex_arrays(settings)
    settled_brightness = max(point.brightness for point in manager.points.values())

    assert np.all(new_vertices[:, 7] > settled_vertices[:, 7])
    assert new_brightness > settled_brightness


def test_requested_visual_controls_are_available() -> None:
    names = {
        item[0]
        for _, section in SECTIONS
        for item in section
    }

    assert {
        "max_connections_per_point",
        "line_thickness",
        "glow_intensity",
        "new_node_size_boost",
        "new_node_glow_boost",
        "node_settle_duration",
        "label_percentage",
        "label_max_count",
        "label_box_size",
        "label_text_size",
        "label_min_opacity",
        "camera_follow_strength",
        "camera_follow_smoothing",
        "camera_follow_max_speed",
        "camera_focus_hold_duration",
        "camera_return_duration",
    } <= names
    assert "point_generation_rate" not in names


def test_lowering_max_lines_per_point_prunes_existing_edges() -> None:
    settings = VisualizationSettings(
        node_interval_s=0.02,
        max_connections_per_point=6,
        max_temporal_connections=2,
        max_intra_connections=4,
        max_similarity_connections=4,
    )
    manager = PointManager(29)
    preset = PRESETS[settings.palette]
    for index in range(4):
        manager.ingest(frame(index * 0.03), settings, preset)
        manager.update_at(index * 0.03, settings)

    settings.max_connections_per_point = 1
    manager.update_at(0.2, settings)

    assert max(len(point.connections) for point in manager.points.values()) <= 1


def test_open_source_defaults_keep_labels_visible() -> None:
    settings = VisualizationSettings()

    assert settings.scientific_labels
    assert settings.scientific_labels_export
    assert 0.5 <= settings.line_thickness < 1.0
    assert settings.glow_intensity > 1.0
    assert settings.point_core_size < 1.5
    assert settings.max_connections_per_point == 6
    assert settings.max_similarity_connections == 2
    assert 0.45 < settings.historical_opacity < 0.55


def test_camera_follow_targets_new_activity_and_limits_pan_speed() -> None:
    camera = Camera()
    zero = np.zeros(3, dtype=np.float32)
    first_focus = np.asarray((4.0, 0.0, 0.0), dtype=np.float32)
    camera.matrices(
        1.0,
        9.0,
        47.0,
        0.0,
        0.0,
        (0.0, 0.0, 0.0),
        zero,
        4.0,
        True,
        1.0,
        0.12,
        3.0,
        30.0,
        first_focus,
        1.0,
        0.55,
        1.0,
        1.0,
        1.0,
    )
    np.testing.assert_allclose(camera._follow_center, first_focus)

    camera.matrices(
        1.0,
        9.0,
        47.0,
        0.0,
        0.1,
        (0.0, 0.0, 0.0),
        zero,
        4.0,
        True,
        1.0,
        0.12,
        3.0,
        30.0,
        -first_focus,
        1.0,
        0.55,
        1.0,
        1.0,
        1.0,
    )

    assert 0.09 <= float(np.linalg.norm(camera._follow_center - first_focus)) <= 0.101


def test_camera_focus_holds_then_returns_to_geometry() -> None:
    settings = VisualizationSettings()
    manager = PointManager(53)
    manager.ingest(frame(0.0), settings, PRESETS[settings.palette])
    manager.update_at(0.1, settings)

    focus, weight, radius = manager.camera_focus(0.25, 0.75)

    assert focus is not None
    assert weight == 1.0
    assert radius >= 0.55

    manager.update_at(0.7, settings)
    _, returning_weight, _ = manager.camera_focus(0.25, 0.75)
    assert 0.0 < returning_weight < 1.0

    manager.update_at(1.1, settings)
    focus, weight, _ = manager.camera_focus(0.25, 0.75)
    assert focus is None
    assert weight == 0.0


def test_camera_focus_ignores_future_only_nodes() -> None:
    settings = VisualizationSettings()
    manager = PointManager(59)
    manager.ingest(frame(1.0), settings, PRESETS[settings.palette])
    manager.update_at(0.5, settings)

    focus, weight, radius = manager.camera_focus(0.25, 0.75)

    assert focus is None
    assert weight == 0.0
    assert radius == 0.55


def test_band_influence_changes_importance_without_distorting_axes() -> None:
    quiet_bands = VisualizationSettings(
        low_frequency_influence=0.0,
        mid_frequency_influence=0.0,
        high_frequency_influence=0.0,
    )
    strong_bands = VisualizationSettings(
        low_frequency_influence=2.0,
        mid_frequency_influence=2.0,
        high_frequency_influence=2.0,
    )
    quiet = PointManager(61)
    strong = PointManager(61)
    preset = PRESETS[quiet_bands.palette]
    quiet.ingest(frame(0.0), quiet_bands, preset)
    strong.ingest(frame(0.0), strong_bands, preset)

    quiet_positions = np.asarray(
        [point.base_position for point in quiet.points.values()]
    )
    strong_positions = np.asarray(
        [point.base_position for point in strong.points.values()]
    )

    np.testing.assert_array_equal(quiet_positions, strong_positions)
    assert min(point.importance for point in strong.points.values()) > max(
        point.importance for point in quiet.points.values()
    )


def test_fade_speed_controls_historical_decay() -> None:
    slow_settings = VisualizationSettings(
        active_duration=1.0,
        historical_duration=10.0,
        fade_speed=0.5,
    )
    fast_settings = VisualizationSettings(
        active_duration=1.0,
        historical_duration=10.0,
        fade_speed=3.0,
    )
    slow = PointManager(67)
    fast = PointManager(67)
    preset = PRESETS[slow_settings.palette]
    slow.ingest(frame(0.0), slow_settings, preset)
    fast.ingest(frame(0.0), fast_settings, preset)
    slow.update_at(5.0, slow_settings)
    fast.update_at(5.0, fast_settings)

    assert min(point.alpha for point in slow.points.values()) > max(
        point.alpha for point in fast.points.values()
    )
