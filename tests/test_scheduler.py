from drone4rf.config import AdaptiveSweepConfig, BandConfig, SweepConfig
from drone4rf.scheduler import SweepScheduler

FS = 10e6
CHUNK = 65_536


def _cfg(bands, adaptive=None) -> SweepConfig:
    cfg = SweepConfig(
        bands=tuple(bands),
        adaptive=adaptive or AdaptiveSweepConfig(),
    )
    cfg.validate()
    return cfg


def test_steps_cover_band() -> None:
    band = BandConfig(name="ism", start_hz=2400e6, stop_hz=2483.5e6)
    sched = SweepScheduler(_cfg([band]), FS, CHUNK)
    usable = FS * 0.75
    steps = sched.steps
    # Windows must jointly cover the whole band.
    lo = min(s.center_hz for s in steps) - usable / 2
    hi = max(s.center_hz for s in steps) + usable / 2
    assert lo <= band.start_hz
    assert hi >= band.stop_hz
    # And no window may extend beyond the band by more than the overlap.
    assert all(band.start_hz - usable <= s.center_hz <= band.stop_hz for s in steps)


def test_narrow_band_gets_single_centered_step() -> None:
    band = BandConfig(name="rc433", start_hz=433.05e6, stop_hz=434.79e6)
    sched = SweepScheduler(_cfg([band]), FS, CHUNK)
    assert len(sched.steps) == 1
    assert abs(sched.steps[0].center_hz - 433.92e6) < 1e4


def test_disabled_band_is_skipped() -> None:
    bands = [
        BandConfig(name="a", start_hz=2400e6, stop_hz=2405e6),
        BandConfig(name="b", start_hz=5725e6, stop_hz=5730e6, enabled=False),
    ]
    sched = SweepScheduler(_cfg(bands), FS, CHUNK)
    assert {s.band for s in sched.steps} == {"a"}


def test_priority_breaks_ties_when_both_due() -> None:
    bands = [
        BandConfig(name="low", start_hz=2400e6, stop_hz=2405e6, priority=1),
        BandConfig(name="high", start_hz=5725e6, stop_hz=5730e6, priority=5),
    ]
    sched = SweepScheduler(_cfg(bands), FS, CHUNK)
    assert sched.next_step(now=0.0).band == "high"


def test_round_robin_between_equal_priority_bands() -> None:
    bands = [
        BandConfig(name="a", start_hz=2400e6, stop_hz=2405e6),
        BandConfig(name="b", start_hz=5725e6, stop_hz=5730e6),
    ]
    sched = SweepScheduler(_cfg(bands), FS, CHUNK)
    visited = []
    now = 0.0
    for _ in range(6):
        step = sched.next_step(now)
        visited.append(step.band)
        now += 1.0
        sched.complete(step.step_id, now)
    # Oldest-due-first selection alternates strictly between the bands.
    assert visited == ["a", "b", "a", "b", "a", "b"]


def test_revisit_interval_defers_step() -> None:
    bands = [
        BandConfig(name="slow", start_hz=2400e6, stop_hz=2405e6, revisit_s=100.0),
        BandConfig(name="fast", start_hz=5725e6, stop_hz=5730e6, revisit_s=0.0),
    ]
    sched = SweepScheduler(_cfg(bands), FS, CHUNK)
    visits = {"slow": 0, "fast": 0}
    now = 0.0
    for _ in range(20):
        step = sched.next_step(now)
        visits[step.band] += 1
        now += 1.0
        sched.complete(step.step_id, now)
    # 20 s of sweeping: 'slow' (100 s revisit) is visited once, then its
    # timer keeps it out; 'fast' (revisit 0) takes every other slot.
    assert visits["slow"] == 1
    assert visits["fast"] == 19


def test_adaptive_activity_shortens_revisit() -> None:
    band = BandConfig(name="a", start_hz=2400e6, stop_hz=2405e6, revisit_s=60.0)
    adaptive = AdaptiveSweepConfig(enabled=True, hot_revisit_s=1.0, hot_visits=2)
    sched = SweepScheduler(_cfg([band], adaptive), FS, CHUNK)
    step = sched.next_step(0.0)
    sched.complete(step.step_id, 0.0)
    assert sched.steps[0].next_due == 60.0  # normal revisit
    sched.report_activity(step.center_hz, detections=3, now=0.0)
    assert sched.steps[0].next_due <= 1.0  # pulled forward
    # Hot visits use the short interval, then decay back to normal.
    sched.complete(step.step_id, 10.0)
    assert sched.steps[0].next_due == 11.0
    sched.complete(step.step_id, 20.0)
    assert sched.steps[0].next_due == 21.0
    sched.complete(step.step_id, 30.0)
    assert sched.steps[0].next_due == 90.0


def test_dwell_chunks_from_dwell_seconds() -> None:
    band = BandConfig(name="a", start_hz=2400e6, stop_hz=2405e6, dwell_s=0.5)
    sched = SweepScheduler(_cfg([band]), FS, CHUNK)
    expected = max(1, round(0.5 * FS / CHUNK))
    assert sched.steps[0].dwell_chunks == expected
