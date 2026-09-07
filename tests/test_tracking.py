from drone4rf.config import TrackingConfig
from drone4rf.detectors.base import Detection
from drone4rf.tracking import SignalTracker


def _det(freq: float, ts: float) -> Detection:
    return Detection(
        detector="energy",
        timestamp=ts,
        center_hz=freq,
        bandwidth_hz=100e3,
        peak_db=-55.0,
        avg_db=-60.0,
        snr_db=25.0,
        duration_s=0.02,
        score=0.8,
    )


def test_promotion_after_repeated_hits() -> None:
    cfg = TrackingConfig(promote_hits=3, drop_after_misses=5)
    tracker = SignalTracker(cfg)
    events = []
    for i in range(4):
        events += tracker.update([_det(2.4e9, float(i))], now=float(i))
    promoted = [e for e in events if e.kind == "track_persistent"]
    assert len(promoted) == 1
    assert promoted[0].track.hits >= 3
    assert "persisted" in promoted[0].explanation
    assert "not a confirmed drone" in promoted[0].explanation


def test_track_closes_after_misses_with_burst_stats() -> None:
    cfg = TrackingConfig(promote_hits=3, drop_after_misses=2)
    tracker = SignalTracker(cfg)
    events = []
    # Intermittent signal: present 2 of every 3 frames.
    for i in range(9):
        dets = [_det(2.4e9, float(i))] if i % 3 != 2 else []
        events += tracker.update(dets, now=float(i))
    for i in range(9, 12):  # signal gone
        events += tracker.update([], now=float(i))
    closed = [e for e in events if e.kind == "track_closed"]
    assert len(closed) == 1
    assert closed[0].track.occupancy < 0.8  # recognized as intermittent
    assert "burst" in closed[0].explanation


def test_nearby_detections_join_one_track() -> None:
    cfg = TrackingConfig(match_tolerance_hz=200e3, promote_hits=3, drop_after_misses=5)
    tracker = SignalTracker(cfg)
    for i, off in enumerate((0.0, 50e3, -80e3, 30e3)):
        tracker.update([_det(2.4e9 + off, float(i))], now=float(i))
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].hits == 4


def test_out_of_window_tracks_accrue_no_misses() -> None:
    """Sweep mode: a 2.4 GHz track is untouched while observing 5.8 GHz."""
    cfg = TrackingConfig(promote_hits=3, drop_after_misses=2)
    tracker = SignalTracker(cfg)
    win_24 = (2.395e9, 2.405e9)
    win_58 = (5.795e9, 5.805e9)
    tracker.update([_det(2.4e9, 0.0)], now=0.0, window=win_24)
    # Many frames observing a different band: would exceed
    # drop_after_misses if the window were ignored.
    for i in range(1, 8):
        tracker.update([], now=float(i), window=win_58)
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert track.consecutive_misses == 0
    assert track.frames_observed == 1  # only the frame that saw its band
    # Back on 2.4 GHz: the track continues accumulating hits.
    tracker.update([_det(2.4e9, 8.0)], now=8.0, window=win_24)
    assert tracker.active_tracks[0].hits == 2


def test_misses_do_accrue_inside_window() -> None:
    cfg = TrackingConfig(promote_hits=3, drop_after_misses=2)
    tracker = SignalTracker(cfg)
    win = (2.395e9, 2.405e9)
    tracker.update([_det(2.4e9, 0.0)], now=0.0, window=win)
    tracker.update([], now=1.0, window=win)
    tracker.update([], now=2.0, window=win)
    assert tracker.active_tracks == []  # closed after 2 in-window misses


def test_distant_detections_make_separate_tracks() -> None:
    cfg = TrackingConfig(match_tolerance_hz=200e3, promote_hits=3, drop_after_misses=5)
    tracker = SignalTracker(cfg)
    tracker.update([_det(2.40e9, 0.0), _det(2.45e9, 0.0)], now=0.0)
    assert len(tracker.active_tracks) == 2
