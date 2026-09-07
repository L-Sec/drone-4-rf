"""Unit tests for the confidence fusion engine."""

import math

from drone4rf.analytics.emi import EMIObservation
from drone4rf.analytics.fusion import (
    CATEGORY_BACKGROUND,
    CATEGORY_POSSIBLE,
    CATEGORY_UNCLASSIFIED,
    DroneAssessment,
    FusionEngine,
    category_rank,
)
from drone4rf.analytics.hopping import HopCandidate
from drone4rf.config import AnalyticsConfig, FusionWeights
from drone4rf.tracking import SignalTrack

NOW = 10.0


def _track(
    tid: int,
    freq: float,
    bw: float = 100e3,
    hits: int = 15,
    snr: float = 30.0,
    duty: float = 1.0,
    n_bursts: int = 1,
    cv: float = math.nan,
    flags: tuple = (),
) -> SignalTrack:
    tr = SignalTrack(
        track_id=tid,
        center_hz=freq,
        bandwidth_hz=bw,
        first_seen=0.0,
        last_seen=NOW,
        hits=hits,
        frames_observed=hits,
        peak_db=-50.0,
        max_snr_db=snr,
    )
    tr.quality_flags = set(flags)
    for i in range(min(hits, 20)):
        tr.history.append(
            {"t": float(i), "snr_db": snr, "bandwidth_hz": bw,
             "duty": duty, "n_bursts": n_bursts, "period_cv": cv}
        )
    return tr


def _hop_candidate(score: float = 0.9) -> HopCandidate:
    centers = tuple(2.405e9 + i * 1e6 for i in range(6))
    return HopCandidate(
        timestamp=NOW,
        freq_lo_hz=centers[0],
        freq_hi_hz=centers[-1],
        n_channels=6,
        channel_centers_hz=centers,
        mean_duty=0.3,
        spacing_regularity=1.0,
        score=score,
        explanation="6 distinct narrowband channels",
    )


def _engine(**overrides) -> FusionEngine:
    cfg = AnalyticsConfig(**overrides) if overrides else AnalyticsConfig()
    cfg.validate()
    return FusionEngine(cfg)


def test_continuous_tone_is_unclassified() -> None:
    out = _engine().assess(NOW, [_track(1, 2.42e9)], [], [])
    assert len(out) == 1
    assert out[0].category == CATEGORY_UNCLASSIFIED
    assert "not a confirmed drone" in out[0].explanation


def test_regular_bursts_raise_category() -> None:
    bursty = _track(1, 2.42e9, duty=0.3, n_bursts=3, cv=0.1)
    out = _engine().assess(NOW, [bursty], [], [])
    assert category_rank(out[0].category) >= category_rank(CATEGORY_POSSIBLE)
    assert out[0].evidence["burst"] > 0.9
    assert "burst structure" in out[0].explanation


def test_wifi_like_signal_is_penalized_to_background() -> None:
    wifi = _track(1, 2.437e9, bw=18e6, duty=0.4)
    out = _engine().assess(NOW, [wifi], [], [])
    assert out[0].category == CATEGORY_BACKGROUND
    assert out[0].penalties["wifi_similarity"] == 1.0
    assert "Wi-Fi" in out[0].explanation


def test_hop_candidate_consumes_channel_tracks() -> None:
    tracks = [_track(i, 2.405e9 + i * 1e6, hits=5, snr=25.0) for i in range(6)]
    out = _engine().assess(NOW, tracks, [_hop_candidate()], [])
    hoppers = [a for a in out if a.entity == "hopper"]
    assert len(hoppers) == 1
    assert len(out) == 1  # channel tracks folded in, not double-counted
    a = hoppers[0]
    assert a.evidence["hopping"] > 0.8
    assert "anomaly" in a.evidence
    assert category_rank(a.category) >= category_rank(CATEGORY_POSSIBLE)


def test_multiband_correlation_boosts_both_entities() -> None:
    tracks = [_track(i, 2.405e9 + i * 1e6, hits=5, snr=25.0) for i in range(6)]
    video = _track(99, 5.80e9, bw=6e6, duty=0.4, hits=15, snr=30.0)
    out = _engine().assess(NOW, tracks + [video], [_hop_candidate()], [])
    with_mb = [a for a in out if "multiband" in a.evidence]
    assert len(with_mb) == 2  # both the hopper and the video-band entity
    assert all(
        category_rank(a.category) >= category_rank(CATEGORY_POSSIBLE)
        for a in with_mb
    )
    assert any("time-correlated activity" in a.explanation for a in with_mb)


def test_emi_alone_is_capped_at_possible() -> None:
    # Inflated EMI weight would push confidence sky-high without the cap.
    engine = _engine(weights=FusionWeights(emi=10.0, bias=0.0))
    obs = EMIObservation(
        timestamp=NOW, center_hz=2.44e9, spacing_hz=450e3,
        n_harmonics=8, score=1.0, explanation="harmonic comb",
    )
    out = engine.assess(NOW, [], [], [obs])
    assert len(out) == 1
    a = out[0]
    assert a.entity == "emi"
    assert category_rank(a.category) <= category_rank(CATEGORY_POSSIBLE)
    assert "Capped at 'possible'" in a.explanation
    assert a.confidence <= engine.cfg.possible_max


def test_clipping_caps_at_possible() -> None:
    engine = _engine(
        weights=FusionWeights(burst=8.0, clipping_penalty=0.0, bias=0.0)
    )
    hot = _track(1, 2.42e9, duty=0.3, n_bursts=3, cv=0.1, flags=("clipping",))
    out = engine.assess(NOW, [hot], [], [])
    a = out[0]
    assert category_rank(a.category) <= category_rank(CATEGORY_POSSIBLE)
    assert "clipping" in a.explanation


def test_fixed_infrastructure_penalized() -> None:
    old = _track(1, 2.45e9, hits=100, snr=30.0, duty=1.0)
    old.first_seen = -100.0  # 110 s old at NOW, occupancy 1.0
    out = _engine().assess(NOW, [old], [], [])
    assert out[0].penalties.get("fixed_emitter", 0) > 0
    assert out[0].category == CATEGORY_BACKGROUND


def test_vocabulary_never_contains_confirmed() -> None:
    from drone4rf.analytics.fusion import CATEGORY_ORDER

    assert not any("confirmed" in c for c in CATEGORY_ORDER)


def test_assessment_is_serializable_shape() -> None:
    out = _engine().assess(NOW, [_track(1, 2.42e9)], [], [])
    a = out[0]
    assert isinstance(a, DroneAssessment)
    assert 0.0 <= a.confidence <= 1.0
    assert isinstance(a.evidence, dict) and isinstance(a.penalties, dict)
