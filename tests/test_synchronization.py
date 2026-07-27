from unittest.mock import patch

import numpy as np

from voxis.models import AnalysisFrame
from voxis.playback.clock import MediaClock
from voxis.playback.synchronizer import TimelineSynchronizer


def frame(timestamp: float, generation: int) -> AnalysisFrame:
    return AnalysisFrame(
        timestamp_s=timestamp,
        duration_s=0.02,
        spectrum=np.zeros(8, dtype=np.float32),
        frequencies=np.arange(8, dtype=np.float32),
        band_energy=np.zeros(3, dtype=np.float32),
        rms_db=-30.0,
        peak_db=-20.0,
        dominant_hz=100.0,
        stereo_balance=0.0,
        phase_correlation=1.0,
        generation=generation,
    )


def test_only_due_analysis_is_consumed() -> None:
    sync = TimelineSynchronizer(capacity=8, stale_after_s=0.25)
    sync.reset(4)
    for timestamp in (1.00, 1.02, 1.04, 1.30):
        assert sync.push(frame(timestamp, 4))

    ready = sync.consume_ready(1.035, 4, future_tolerance_s=0.0)

    assert [item.timestamp_s for item in ready] == [1.00, 1.02]
    assert sync.buffered_count() == 2


def test_seek_generation_rejects_previous_timeline() -> None:
    sync = TimelineSynchronizer(capacity=8)
    sync.reset(2)
    assert sync.push(frame(9.0, 2))
    sync.reset(3)

    assert not sync.push(frame(9.02, 2))
    assert sync.push(frame(2.0, 3))
    assert sync.consume_ready(2.0, 2) == []
    assert [x.timestamp_s for x in sync.consume_ready(2.0, 3)] == [2.0]


def test_stale_frames_are_dropped_instead_of_accumulating_lag() -> None:
    sync = TimelineSynchronizer(capacity=4, stale_after_s=0.10)
    sync.reset(1)
    for timestamp in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        sync.push(frame(timestamp, 1))

    ready = sync.consume_ready(1.5, 1)

    assert [item.timestamp_s for item in ready] == [1.4, 1.5]
    assert sync.buffered_count() == 0


def test_media_clock_interpolates_playback_and_freezes_on_pause() -> None:
    with patch(
        "voxis.playback.clock.time.monotonic",
        side_effect=[10.0, 10.0, 10.1, 10.2, 10.3, 11.0],
    ):
        clock = MediaClock()
        clock.update_position(3.0)
        clock.set_playing(True)
        moving = clock.snapshot()
        clock.set_playing(False)
        paused = clock.snapshot()

    assert 3.09 <= moving.position_s <= 3.11
    assert abs(paused.position_s - 3.2) < 1e-9
    assert not paused.playing
