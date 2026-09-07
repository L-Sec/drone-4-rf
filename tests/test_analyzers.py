"""Unit tests for the hopping, EMI, and background analyzers."""

from drone4rf.analytics import background
from drone4rf.analytics.emi import EMIAnalyzer
from drone4rf.analytics.hopping import HopAnalyzer
from drone4rf.config import BackgroundConfig, EMIConfig, HoppingConfig
from drone4rf.detectors.base import Detection
from drone4rf.tracking import SignalTrack


def _det(freq: float, ts: float, bw: float = 50e3, duty: float = 0.3) -> Detection:
    return Detection(
        detector="cfar",
        timestamp=ts,
        center_hz=freq,
        bandwidth_hz=bw,
        peak_db=-50.0,
        avg_db=-55.0,
        snr_db=25.0,
        duration_s=0.02,
        score=0.8,
        features={"duty": duty},
    )


# -- hopping ---------------------------------------------------------------


def test_hopper_pattern_scores_high() -> None:
    analyzer = HopAnalyzer(HoppingConfig())
    channels = [2.405e9 + i * 1e6 for i in range(6)]
    t = 0.0
    for rep in range(4):  # each channel seen repeatedly, low duty
        for ch in channels:
            analyzer.ingest([_det(ch, t, duty=0.3)])
            t += 0.05
    cands = analyzer.candidates(now=t)
    assert len(cands) == 1
    cand = cands[0]
    assert cand.n_channels == 6
    assert cand.score > 0.6
    assert cand.spacing_regularity > 0.9
    assert "6 distinct narrowband channels" in cand.explanation


def test_single_channel_is_not_hopping() -> None:
    analyzer = HopAnalyzer(HoppingConfig())
    for i in range(20):
        analyzer.ingest([_det(2.41e9, i * 0.05)])
    assert analyzer.candidates(now=1.0) == []


def test_continuous_multitone_is_downgraded() -> None:
    """Six always-on tones (duty 1.0) must score far below a hopper."""
    analyzer = HopAnalyzer(HoppingConfig())
    channels = [2.405e9 + i * 1e6 for i in range(6)]
    t = 0.0
    for rep in range(4):
        for ch in channels:
            analyzer.ingest([_det(ch, t, duty=1.0)])
            t += 0.05
    cands = analyzer.candidates(now=t)
    assert cands and cands[0].score < 0.25  # continuous-duty gate applied


def test_wideband_detections_ignored_by_hop_analyzer() -> None:
    analyzer = HopAnalyzer(HoppingConfig())
    for i in range(20):
        analyzer.ingest([_det(2.41e9 + i * 1e6, i * 0.05, bw=15e6)])
    assert analyzer.candidates(now=1.0) == []


def test_old_events_pruned_by_window() -> None:
    analyzer = HopAnalyzer(HoppingConfig(window_s=3.0))
    channels = [2.405e9 + i * 1e6 for i in range(6)]
    for ch in channels:
        analyzer.ingest([_det(ch, 0.0)])
    assert analyzer.candidates(now=10.0) == []  # all events aged out


# -- EMI ---------------------------------------------------------------------


def test_emi_comb_detected() -> None:
    analyzer = EMIAnalyzer(EMIConfig())
    peaks = [_det(2.400e9 + k * 450e3, 1.0, bw=100e3) for k in range(6)]
    obs = analyzer.analyze(1.0, peaks)
    assert obs is not None
    assert obs.n_harmonics == 6
    assert obs.score > 0.6
    assert abs(obs.spacing_hz - 450e3) < 1e3
    assert "power supply" in " ".join(obs.competing)


def test_irregular_peaks_are_not_a_comb() -> None:
    analyzer = EMIAnalyzer(EMIConfig())
    freqs = [2.4001e9, 2.4009e9, 2.4022e9, 2.4051e9, 2.4093e9]  # spacing grows
    obs = analyzer.analyze(1.0, [_det(f, 1.0, bw=100e3) for f in freqs])
    assert obs is None


def test_too_few_peaks_is_not_a_comb() -> None:
    analyzer = EMIAnalyzer(EMIConfig())
    peaks = [_det(2.400e9 + k * 450e3, 1.0, bw=100e3) for k in range(3)]
    assert analyzer.analyze(1.0, peaks) is None


# -- background -----------------------------------------------------------


def test_wifi_similarity_on_channel_grid() -> None:
    cfg = BackgroundConfig()
    assert background.wifi_similarity(cfg, 2.437e9, 18e6, 0.4) == 1.0
    # Narrowband signal on a Wi-Fi channel is not Wi-Fi.
    assert background.wifi_similarity(cfg, 2.437e9, 100e3, 0.4) == 0.0
    # Wide signal far off any grid center is not Wi-Fi.
    assert background.wifi_similarity(cfg, 2.6e9, 18e6, 0.4) == 0.0


def test_bt_similarity_needs_bt_band_span_and_spacing() -> None:
    high = background.bt_similarity(20, 2.405e9, 2.475e9, 1e6, 0.1)
    assert high >= 0.9  # wide span, many channels, BT raster, irregular
    # Narrow-span hop set is not Bluetooth-like.
    low = background.bt_similarity(6, 2.434e9, 2.439e9, 1e6, 1.0)
    assert low < 0.5
    # Outside the BT band entirely.
    assert background.bt_similarity(20, 5.7e9, 5.8e9, 1e6, 0.1) == 0.0
    # Regular channel grid (drone-typical) scores lower than irregular.
    regular = background.bt_similarity(20, 2.405e9, 2.475e9, 1e6, 1.0)
    irregular = background.bt_similarity(20, 2.405e9, 2.475e9, 1e6, 0.0)
    assert irregular > regular


def test_fixed_emitter_requires_age_and_occupancy() -> None:
    cfg = BackgroundConfig()
    old = SignalTrack(
        track_id=1, center_hz=2.45e9, bandwidth_hz=1e6,
        first_seen=0.0, last_seen=100.0, hits=100, frames_observed=100,
    )
    assert background.fixed_emitter_score(cfg, old, now=100.0) > 0.5
    young = SignalTrack(
        track_id=2, center_hz=2.45e9, bandwidth_hz=1e6,
        first_seen=95.0, last_seen=100.0, hits=5, frames_observed=5,
    )
    assert background.fixed_emitter_score(cfg, young, now=100.0) == 0.0
    intermittent = SignalTrack(
        track_id=3, center_hz=2.45e9, bandwidth_hz=1e6,
        first_seen=0.0, last_seen=100.0, hits=30, frames_observed=100,
    )
    assert background.fixed_emitter_score(cfg, intermittent, now=100.0) == 0.0
