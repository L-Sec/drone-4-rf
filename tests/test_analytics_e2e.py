"""End-to-end analytics: simulated emitters -> pipeline -> assessments."""

from pathlib import Path

from drone4rf.analytics.fusion import CATEGORY_POSSIBLE, category_rank
from drone4rf.config import (
    AnalyticsConfig,
    AppConfig,
    DeviceConfig,
    DSPConfig,
    TrackingConfig,
)
from drone4rf.events import EventStore
from drone4rf.pipeline import ScannerPipeline
from drone4rf.sdr.simulated import (
    HopperEmitter,
    SimScenario,
    SimulatedSource,
    ToneEmitter,
)

FS = 10e6

ELEVATED = ("possible_drone_activity", "probable_drone_activity",
            "high_confidence_drone_activity")


def _config() -> AppConfig:
    cfg = AppConfig(
        device=DeviceConfig(sample_rate=FS, center_freq_hz=2.437e9),
        dsp=DSPConfig(fft_size=1024, chunk_samples=65_536),
        tracking=TrackingConfig(promote_hits=4, drop_after_misses=8),
        analytics=AnalyticsConfig(assessment_interval_s=0.05),
    )
    cfg.validate()
    return cfg


def _run(scenario: SimScenario, tmp_path: Path, seconds: float = 2.0):
    cfg = _config()
    source = SimulatedSource(
        scenario, center_hz=cfg.device.center_freq_hz,
        sample_rate=FS, duration_s=seconds,
    )
    store = EventStore(tmp_path / "events.db")
    pipeline = ScannerPipeline(cfg, source, store=store)
    with source:
        pipeline.run()
    rows = store.recent(limit=300)
    store.close()
    return rows


def test_hopper_reaches_elevated_assessment(tmp_path) -> None:
    scenario = SimScenario(
        noise_std=0.005,
        emitters=[
            HopperEmitter(
                channel_offsets_hz=tuple(-3e6 + i * 1e6 for i in range(6)),
                amplitude=0.1,
                hop_period_s=0.002,
                duty=1.0,
            )
        ],
    )
    rows = _run(scenario, tmp_path)
    assessments = [r for r in rows if r["kind"] == "assessment"]
    assert assessments, "hopper should have produced assessments"
    elevated = [a for a in assessments if a["confidence_label"] in ELEVATED]
    assert elevated, (
        "hopper should reach at least 'possible': "
        + str([(a["confidence_label"], a["score"]) for a in assessments])
    )
    best = max(elevated, key=lambda a: a["score"])
    assert "channels" in best["explanation"]
    assert "not a confirmed drone" in best["explanation"]


def test_plain_tone_never_elevated(tmp_path) -> None:
    scenario = SimScenario(
        noise_std=0.005,
        emitters=[ToneEmitter(offset_hz=1.5e6, amplitude=0.1)],
    )
    rows = _run(scenario, tmp_path)
    assessments = [r for r in rows if r["kind"] == "assessment"]
    # Emission policy: quiet/unclassified entities produce no assessment
    # chatter at all - and certainly nothing elevated.
    assert not [a for a in assessments if a["confidence_label"] in ELEVATED]


def test_noise_only_produces_no_assessments(tmp_path) -> None:
    rows = _run(SimScenario(noise_std=0.005, emitters=[]), tmp_path, seconds=1.0)
    assert not [r for r in rows if r["kind"] == "assessment"]


def test_assessment_row_shape(tmp_path) -> None:
    scenario = SimScenario(
        noise_std=0.005,
        emitters=[
            HopperEmitter(
                channel_offsets_hz=tuple(-3e6 + i * 1e6 for i in range(6)),
                amplitude=0.1,
                hop_period_s=0.002,
            )
        ],
    )
    rows = _run(scenario, tmp_path)
    assessments = [r for r in rows if r["kind"] == "assessment"]
    assert assessments
    a = assessments[0]
    assert a["detector"] == "fusion"
    assert 0.0 <= a["score"] <= 1.0
    assert a["features_json"]
    assert category_rank(a["confidence_label"]) >= category_rank(
        "background_activity"
    ) or a["confidence_label"] == CATEGORY_POSSIBLE
