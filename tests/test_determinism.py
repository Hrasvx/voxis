import numpy as np

from voxis.config import VisualizationSettings
from voxis.models import AnalysisFrame
from voxis.presets import PRESETS
from voxis.visualization.connections import edge_set
from voxis.visualization.points import PointManager


def analysis_frame(timestamp: float) -> AnalysisFrame:
    frequencies = np.geomspace(35.0, 16000.0, 64).astype(np.float32)
    spectrum = (
        0.2 + 0.6 * np.abs(np.sin(np.linspace(0.0, 8.0, 64) + timestamp))
    ).astype(np.float32)
    return AnalysisFrame(
        timestamp_s=timestamp,
        duration_s=2048 / 48000,
        spectrum=spectrum,
        frequencies=frequencies,
        band_energy=np.linspace(0.2, 0.8, 12, dtype=np.float32),
        rms_db=-16.0,
        peak_db=-8.0,
        dominant_hz=440.0,
        stereo_balance=0.12,
        phase_correlation=0.8,
        spectral_centroid_hz=2400.0,
        spectral_bandwidth_hz=1800.0,
        spectral_rolloff_hz=6400.0,
        onset_strength=0.35,
    )


def simulate(seed: int):
    settings = VisualizationSettings(
        random_seed=seed,
        max_active_points=1000,
    )
    manager = PointManager(seed)
    preset = PRESETS["Archival Gold"]
    for timestamp in np.arange(0.05, 1.0, 0.05):
        manager.ingest(analysis_frame(float(timestamp)), settings, preset)
        manager.update_at(float(timestamp), settings)
    points, lines = manager.vertex_arrays(settings)
    return points, lines, edge_set(manager.points.values())


def test_same_seed_features_and_timestamps_are_repeatable() -> None:
    first_points, first_lines, first_edges = simulate(12345)
    second_points, second_lines, second_edges = simulate(12345)

    np.testing.assert_array_equal(first_points, second_points)
    np.testing.assert_array_equal(first_lines, second_lines)
    assert first_edges == second_edges


def test_different_seed_does_not_change_acoustic_structure() -> None:
    first_points, first_lines, first_edges = simulate(1)
    second_points, second_lines, second_edges = simulate(2)
    assert first_points.shape == second_points.shape
    np.testing.assert_array_equal(first_points[:, :3], second_points[:, :3])
    np.testing.assert_array_equal(first_points[:, 7], second_points[:, 7])
    np.testing.assert_array_equal(first_lines, second_lines)
    assert first_edges == second_edges
