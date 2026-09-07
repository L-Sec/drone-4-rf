"""Background-signal similarity scoring: Wi-Fi, Bluetooth, fixed emitters.

These scores become confidence PENALTIES in fusion - an emitter that
looks like ordinary neighborhood RF must clear a higher evidence bar
before being called drone-related. They never suppress an event outright
(that is what operator exclusion ranges are for).
"""

from __future__ import annotations

from drone4rf.config import BackgroundConfig
from drone4rf.tracking import SignalTrack

# Wi-Fi channel center grids (Hz). 2.4 GHz: ch 1-13 + ch 14; 5 GHz:
# UNII-1/2/2e/3 20 MHz raster. Coarse by design - the tolerance handles
# 40/80 MHz bonding well enough for a penalty heuristic.
_WIFI_CENTERS = tuple(
    [2412e6 + 5e6 * k for k in range(13)]
    + [2484e6]
    + [5180e6 + 20e6 * k for k in range(33)]
)

_BT_LO, _BT_HI = 2402e6, 2480e6


def wifi_similarity(
    cfg: BackgroundConfig, center_hz: float, bandwidth_hz: float, duty: float | None
) -> float:
    """1.0 = textbook Wi-Fi: wide OFDM burst on a channel-grid center."""
    if not (cfg.wifi_bw_min_hz <= bandwidth_hz <= cfg.wifi_bw_max_hz):
        return 0.0
    grid_dist = min(abs(center_hz - c) for c in _WIFI_CENTERS)
    if grid_dist > cfg.wifi_grid_tolerance_hz:
        return 0.0
    score = 0.6
    if duty is not None and 0.05 <= duty <= 0.95:
        score += 0.4  # bursty occupancy, like traffic-driven Wi-Fi
    return score


def bt_similarity(
    n_channels: int,
    freq_lo_hz: float,
    freq_hi_hz: float,
    channel_spacing_hz: float,
    spacing_regularity: float | None = None,
) -> float:
    """Similarity of a hop pattern to Bluetooth/BLE.

    BT/BLE: many channels confined to 2402-2480 MHz, ~1-2 MHz raster -
    and, crucially, *irregular* apparent spacing when undersampled by a
    scanner (pseudorandom 79/40-channel sequences). Drone FHSS links
    typically use small regular channel grids, so irregular spacing
    inside the BT band is a strong background indicator.
    """
    if freq_lo_hz < _BT_LO - 5e6 or freq_hi_hz > _BT_HI + 5e6:
        return 0.0
    span = freq_hi_hz - freq_lo_hz
    score = 0.0
    if span >= 40e6:
        score += 0.3
    if n_channels >= 10:
        score += 0.2
    if 0.8e6 <= channel_spacing_hz <= 2.5e6:
        score += 0.2
    if spacing_regularity is not None and spacing_regularity < 0.3:
        score += 0.3
    return score


def fixed_emitter_score(
    cfg: BackgroundConfig, track: SignalTrack, now: float
) -> float:
    """Persistent, always-on emitters are infrastructure, not aircraft."""
    age = now - track.first_seen
    if age < cfg.fixed_min_age_s:
        return 0.0
    if track.occupancy < cfg.fixed_min_occupancy:
        return 0.0
    return min(1.0, 0.5 + age / (4 * cfg.fixed_min_age_s))
