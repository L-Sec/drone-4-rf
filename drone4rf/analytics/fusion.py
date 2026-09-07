"""Explainable confidence fusion.

Evidence terms (each in [0, 1]) are combined as weighted log-odds:

    confidence = sigmoid( sum(w_i * s_i) - sum(p_j * b_j) - bias )

and mapped onto a capped category vocabulary. Hard rules that no weight
tuning can override:

- 'confirmed drone' does not exist in the vocabulary;
- entities with fewer than 3 observations are capped at 'possible';
- entities whose evidence is EMI-only are capped at 'possible'
  (appliances produce identical signatures);
- entities observed during receiver clipping are capped at 'possible'.

Every assessment carries its evidence terms, penalty terms, and a
human-readable explanation naming both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from drone4rf.analytics import background
from drone4rf.analytics.emi import EMIObservation
from drone4rf.analytics.hopping import HopCandidate
from drone4rf.config import AnalyticsConfig
from drone4rf.tracking import SignalTrack

CATEGORY_BACKGROUND = "background_activity"
CATEGORY_UNCLASSIFIED = "unclassified_rf_activity"
CATEGORY_POSSIBLE = "possible_drone_activity"
CATEGORY_PROBABLE = "probable_drone_activity"
CATEGORY_HIGH = "high_confidence_drone_activity"

CATEGORY_ORDER = (
    CATEGORY_BACKGROUND,
    CATEGORY_UNCLASSIFIED,
    CATEGORY_POSSIBLE,
    CATEGORY_PROBABLE,
    CATEGORY_HIGH,
)


def category_rank(category: str) -> int:
    return CATEGORY_ORDER.index(category)


@dataclass(frozen=True)
class DroneAssessment:
    timestamp: float
    entity: str  # "track" | "hopper" | "emi"
    center_hz: float
    freq_span_hz: float
    confidence: float
    category: str
    evidence: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)
    observations: int = 0
    explanation: str = ""


@dataclass
class _Entity:
    kind: str
    center_hz: float
    span_hz: float
    observations: int
    evidence: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    clipping: bool = False


class FusionEngine:
    def __init__(self, cfg: AnalyticsConfig, ml_scorer=None) -> None:
        """ml_scorer: optional drone4rf.ml.models.MLScorer. Kept as an
        injected dependency so this module never imports scikit-learn."""
        self.cfg = cfg
        self.ml_scorer = ml_scorer

    # -- public -----------------------------------------------------------

    def assess(
        self,
        now: float,
        tracks: list[SignalTrack],
        hop_candidates: list[HopCandidate],
        emi_observations: list[EMIObservation],
    ) -> list[DroneAssessment]:
        entities = self._build_entities(now, tracks, hop_candidates)
        self._attach_emi(entities, emi_observations, now)
        self._apply_multiband(entities)
        return [self._finalize(now, e) for e in entities]

    # -- entity construction ----------------------------------------------

    def _build_entities(
        self,
        now: float,
        tracks: list[SignalTrack],
        hop_candidates: list[HopCandidate],
    ) -> list[_Entity]:
        cfg = self.cfg
        entities: list[_Entity] = []
        consumed: set[int] = set()

        for cand in hop_candidates:
            ent = _Entity(
                kind="hopper",
                center_hz=cand.center_hz,
                span_hz=cand.freq_hi_hz - cand.freq_lo_hz,
                observations=cand.n_channels,
            )
            ent.evidence["hopping"] = cand.score
            ent.notes.append(cand.explanation)
            # Tracks living on the candidate's channels are part of this
            # entity, not independent anomalies.
            snrs, hits, clip = [], 0, False
            spacing = (
                (cand.freq_hi_hz - cand.freq_lo_hz) / max(1, cand.n_channels - 1)
            )
            for tr in tracks:
                if cand.freq_lo_hz - 1e6 <= tr.center_hz <= cand.freq_hi_hz + 1e6:
                    consumed.add(tr.track_id)
                    snrs.append(tr.max_snr_db)
                    hits += tr.hits
                    clip = clip or ("clipping" in tr.quality_flags)
            if snrs:
                ent.evidence["anomaly"] = self._anomaly_score(max(snrs), hits)
                ent.observations = max(ent.observations, hits)
                ent.clipping = clip
                ent.notes.append(
                    f"peak SNR {max(snrs):.0f} dB across {hits} channel detections"
                )
            bt = background.bt_similarity(
                cand.n_channels,
                cand.freq_lo_hz,
                cand.freq_hi_hz,
                spacing,
                cand.spacing_regularity,
            )
            if bt > 0:
                ent.penalties["bt_similarity"] = bt
                ent.notes.append(
                    "hop pattern partially matches Bluetooth/BLE behavior"
                )
            entities.append(ent)

        for tr in tracks:
            if tr.track_id in consumed or tr.hits < cfg.min_track_hits:
                continue
            ent = _Entity(
                kind="track",
                center_hz=tr.center_hz,
                span_hz=tr.bandwidth_hz,
                observations=tr.hits,
                clipping="clipping" in tr.quality_flags,
            )
            ent.evidence["anomaly"] = self._anomaly_score(tr.max_snr_db, tr.hits)
            ent.notes.append(
                f"signal {tr.max_snr_db:.0f} dB over baseline across "
                f"{tr.hits} detections (occupancy {tr.occupancy:.2f})"
            )
            burst, duty = self._burst_score(tr)
            if burst > 0:
                ent.evidence["burst"] = burst
                ent.notes.append(
                    f"repetitive sub-chunk burst structure (score {burst:.2f}"
                    + (f", duty {duty:.2f})" if duty is not None else ")")
                )
            wifi = background.wifi_similarity(
                cfg.background, tr.center_hz, tr.bandwidth_hz, duty
            )
            if wifi > 0:
                ent.penalties["wifi_similarity"] = wifi
                ent.notes.append("bandwidth and channel position match Wi-Fi")
            fixed = background.fixed_emitter_score(cfg.background, tr, now)
            if fixed > 0:
                ent.penalties["fixed_emitter"] = fixed
                ent.notes.append(
                    "long-lived continuous occupancy suggests fixed infrastructure"
                )
            entities.append(ent)
        return entities

    def _attach_emi(
        self,
        entities: list[_Entity],
        emi_observations: list[EMIObservation],
        now: float,
    ) -> None:
        recent = [
            o for o in emi_observations if now - o.timestamp <= self.cfg.multiband_window_s
        ]
        if not recent:
            return
        best = max(recent, key=lambda o: o.score)
        # EMI supports only the single entity closest to the comb - a
        # nearby switching supply must not inflate every track in the
        # band. 5 MHz vicinity: motor EMI rides on/close to the link.
        nearest = min(
            entities,
            key=lambda e: abs(e.center_hz - best.center_hz),
            default=None,
        )
        matched = False
        if nearest is not None and abs(nearest.center_hz - best.center_hz) < 5e6:
            nearest.evidence["emi"] = best.score
            nearest.notes.append(best.explanation)
            matched = True
        if not matched:
            ent = _Entity(
                kind="emi",
                center_hz=best.center_hz,
                span_hz=best.spacing_hz * best.n_harmonics,
                observations=1,
            )
            ent.evidence["emi"] = best.score
            ent.notes.append(best.explanation)
            entities.append(ent)

    def _apply_multiband(self, entities: list[_Entity]) -> None:
        """Simultaneous strong activity in well-separated bands reinforces
        both entities (e.g. control-band hopping + video-band wideband)."""
        w = self.cfg.weights

        def base_strength(e: _Entity) -> float:
            return sum(
                getattr(w, name, 1.0) * s for name, s in e.evidence.items()
            )

        strong = [e for e in entities if base_strength(e) >= 0.8 and e.kind != "emi"]
        for e in strong:
            for other in strong:
                if (
                    other is not e
                    and abs(other.center_hz - e.center_hz)
                    >= self.cfg.multiband_min_separation_hz
                ):
                    e.evidence["multiband"] = 1.0
                    e.notes.append(
                        f"time-correlated activity also present near "
                        f"{other.center_hz / 1e9:.3f} GHz"
                    )
                    break

    # -- scoring ----------------------------------------------------------

    @staticmethod
    def _anomaly_score(snr_db: float, hits: int) -> float:
        snr_part = min(1.0, max(0.0, snr_db) / 30.0)
        persistence_part = min(1.0, hits / 15.0)
        return 0.5 * snr_part + 0.5 * persistence_part

    def _burst_score(self, track: SignalTrack) -> tuple[float, float | None]:
        cfg = self.cfg.burst
        recent = [h for h in track.history if h.get("duty") is not None][-20:]
        if not recent:
            return 0.0, None
        duties = [h["duty"] for h in recent]
        mean_duty = sum(duties) / len(duties)
        n_bursts = [h["n_bursts"] for h in recent if h.get("n_bursts") is not None]
        cvs = [
            h["period_cv"]
            for h in recent
            if h.get("period_cv") is not None and not math.isnan(h["period_cv"])
        ]
        score = 0.0
        if cfg.duty_min <= mean_duty <= cfg.duty_max:
            score += 0.4
        if n_bursts and sorted(n_bursts)[len(n_bursts) // 2] >= cfg.min_bursts:
            score += 0.3
        if cvs and sorted(cvs)[len(cvs) // 2] <= cfg.period_cv_max:
            score += 0.3
        return score, mean_duty

    # -- finalization -----------------------------------------------------

    def _finalize(self, now: float, ent: _Entity) -> DroneAssessment:
        w = self.cfg.weights
        self._apply_ml(ent)
        raw = sum(getattr(w, name, 1.0) * s for name, s in ent.evidence.items())
        penalty_map = {
            "wifi_similarity": w.wifi_penalty,
            "bt_similarity": w.bt_penalty,
            "fixed_emitter": w.fixed_penalty,
            "ml_background": w.ml,
        }
        raw -= sum(penalty_map.get(n, 1.0) * b for n, b in ent.penalties.items())
        if ent.clipping:
            ent.penalties["clipping"] = 1.0
            raw -= w.clipping_penalty
        if ent.observations < 3:
            ent.penalties["single_observation"] = 1.0
            raw -= w.single_obs_penalty
        raw -= w.bias
        confidence = 1.0 / (1.0 + math.exp(-raw))
        category = self._category(confidence)

        # Hard caps (see module docstring).
        cap_reasons = []
        if ent.observations < 3:
            cap_reasons.append("too few independent observations")
        if ent.clipping:
            cap_reasons.append("receiver was clipping during observation")
        non_emi = [n for n in ent.evidence if n != "emi"]
        if not non_emi:
            cap_reasons.append(
                "EMI signature alone (many appliances produce the same)"
            )
        if cap_reasons and category_rank(category) > category_rank(CATEGORY_POSSIBLE):
            category = CATEGORY_POSSIBLE
            confidence = min(confidence, self.cfg.possible_max)

        explanation = "; ".join(ent.notes)
        if ent.penalties:
            names = {
                "wifi_similarity": "Wi-Fi-like signature",
                "bt_similarity": "Bluetooth-like hopping",
                "fixed_emitter": "fixed-infrastructure behavior",
                "clipping": "receiver overload during observation",
                "single_observation": "too few observations",
            }
            explanation += ". Confidence reduced by: " + ", ".join(
                names.get(n, n) for n in ent.penalties
            )
        if cap_reasons:
            explanation += ". Capped at 'possible': " + "; ".join(cap_reasons)
        explanation += ". Passive RF indicators only - not a confirmed drone."

        return DroneAssessment(
            timestamp=now,
            entity=ent.kind,
            center_hz=ent.center_hz,
            freq_span_hz=ent.span_hz,
            confidence=confidence,
            category=category,
            evidence=dict(ent.evidence),
            penalties=dict(ent.penalties),
            observations=ent.observations,
            explanation=explanation,
        )

    def _apply_ml(self, ent: _Entity) -> None:
        """Add classifier/novelty terms from the optional ML subsystem.

        The feature vector is built from the deterministic scores BEFORE
        any ml terms exist, so the model never sees its own output. A
        rejected (uncertain) classification contributes nothing.
        """
        if self.ml_scorer is None:
            return
        from drone4rf.ml.features import vector_from_scores

        fv = vector_from_scores(
            ent.evidence, ent.penalties, ent.observations, ent.span_hz
        )
        result = self.ml_scorer.score(fv)
        if result.rejected:
            ent.notes.append(
                "ML classifier abstained (below confidence threshold)"
            )
        elif result.p_drone is not None:
            if result.p_drone >= 0.5:
                ent.evidence["ml"] = result.p_drone
            else:
                ent.penalties["ml_background"] = 1.0 - result.p_drone
            calib = "calibrated " if self.ml_scorer.calibrated else "UNCALIBRATED "
            ent.notes.append(
                f"{calib}ML classifier: P(drone)={result.p_drone:.2f} "
                f"(model {result.model_version[:19]})"
            )
        if result.anomaly01 is not None and result.anomaly01 > 0.5:
            ent.evidence["ml_anomaly"] = result.anomaly01
            ent.notes.append(
                f"behavioral profile unusual for this site "
                f"(novelty {result.anomaly01:.2f})"
            )

    def _category(self, confidence: float) -> str:
        cfg = self.cfg
        if confidence < cfg.background_max:
            return CATEGORY_BACKGROUND
        if confidence < cfg.unclassified_max:
            return CATEGORY_UNCLASSIFIED
        if confidence < cfg.possible_max:
            return CATEGORY_POSSIBLE
        if confidence < cfg.probable_max:
            return CATEGORY_PROBABLE
        return CATEGORY_HIGH
