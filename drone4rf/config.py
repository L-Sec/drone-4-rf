"""Validated application configuration loaded from YAML.

Every config value used anywhere in the pipeline lives in these frozen
dataclasses; modules never read YAML directly. Validation errors name the
offending key path so operators can fix the file quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


# HackRF One hardware constraints.
HACKRF_SAMPLE_RATE_RANGE = (2_000_000.0, 20_000_000.0)
HACKRF_LNA_STEP = 8
HACKRF_LNA_MAX = 40
HACKRF_VGA_STEP = 2
HACKRF_VGA_MAX = 62


def _require(cond: bool, key: str, msg: str) -> None:
    if not cond:
        raise ConfigError(f"config error at '{key}': {msg}")


@dataclass(frozen=True)
class DeviceConfig:
    driver: str = "hackrf"
    sample_rate: float = 10e6
    center_freq_hz: float = 2_437e6
    lna_gain_db: int = 16
    vga_gain_db: int = 20
    amp_enabled: bool = False
    freq_correction_ppm: float = 0.0

    def validate(self) -> None:
        _require(self.sample_rate > 0, "device.sample_rate", "must be positive")
        _require(self.center_freq_hz > 0, "device.center_freq_hz", "must be positive")
        if self.driver == "hackrf":
            lo, hi = HACKRF_SAMPLE_RATE_RANGE
            _require(
                lo <= self.sample_rate <= hi,
                "device.sample_rate",
                f"HackRF supports {lo/1e6:.0f}-{hi/1e6:.0f} MS/s",
            )
            _require(
                0 <= self.lna_gain_db <= HACKRF_LNA_MAX
                and self.lna_gain_db % HACKRF_LNA_STEP == 0,
                "device.lna_gain_db",
                f"HackRF LNA gain must be 0-{HACKRF_LNA_MAX} in {HACKRF_LNA_STEP} dB steps",
            )
            _require(
                0 <= self.vga_gain_db <= HACKRF_VGA_MAX
                and self.vga_gain_db % HACKRF_VGA_STEP == 0,
                "device.vga_gain_db",
                f"HackRF VGA gain must be 0-{HACKRF_VGA_MAX} in {HACKRF_VGA_STEP} dB steps",
            )
        _require(
            abs(self.freq_correction_ppm) < 200,
            "device.freq_correction_ppm",
            "implausibly large ppm correction",
        )


@dataclass(frozen=True)
class DSPConfig:
    fft_size: int = 4096
    overlap: float = 0.5
    window: str = "hann"
    chunk_samples: int = 262_144
    dc_mask_bins: int = 3

    def validate(self) -> None:
        _require(
            self.fft_size >= 64 and (self.fft_size & (self.fft_size - 1)) == 0,
            "dsp.fft_size",
            "must be a power of two >= 64",
        )
        _require(0.0 <= self.overlap <= 0.9, "dsp.overlap", "must be in [0.0, 0.9]")
        _require(
            self.chunk_samples >= self.fft_size,
            "dsp.chunk_samples",
            "must be >= fft_size",
        )
        _require(
            0 <= self.dc_mask_bins < self.fft_size // 4,
            "dsp.dc_mask_bins",
            "must be small relative to fft_size",
        )


@dataclass(frozen=True)
class EnergyDetectorConfig:
    enabled: bool = True
    min_bins: int = 2
    merge_gap_bins: int = 2

    def validate(self) -> None:
        _require(self.min_bins >= 1, "detection.energy.min_bins", "must be >= 1")
        _require(self.merge_gap_bins >= 0, "detection.energy.merge_gap_bins", "must be >= 0")


@dataclass(frozen=True)
class CFARDetectorConfig:
    enabled: bool = True
    guard_bins: int = 4
    train_bins: int = 24
    quantile: float = 0.75
    offset_db: float = 9.0
    min_bins: int = 2
    merge_gap_bins: int = 2

    def validate(self) -> None:
        _require(self.guard_bins >= 0, "detection.cfar.guard_bins", "must be >= 0")
        _require(self.train_bins >= 4, "detection.cfar.train_bins", "must be >= 4")
        _require(0.0 < self.quantile < 1.0, "detection.cfar.quantile", "must be in (0, 1)")
        _require(self.offset_db > 0, "detection.cfar.offset_db", "must be positive")
        _require(self.min_bins >= 1, "detection.cfar.min_bins", "must be >= 1")


@dataclass(frozen=True)
class DetectionConfig:
    energy: EnergyDetectorConfig = field(default_factory=EnergyDetectorConfig)
    cfar: CFARDetectorConfig = field(default_factory=CFARDetectorConfig)
    clip_fraction_warn: float = 1e-3

    def validate(self) -> None:
        self.energy.validate()
        self.cfar.validate()
        _require(
            0.0 < self.clip_fraction_warn <= 1.0,
            "detection.clip_fraction_warn",
            "must be in (0, 1]",
        )


@dataclass(frozen=True)
class BaselineConfig:
    alpha_up: float = 0.02
    alpha_down: float = 0.2
    spread_alpha: float = 0.05
    k_mad: float = 6.0
    min_delta_db: float = 6.0
    hot_bin_damping: float = 0.1
    path: str = "data/baseline.npz"

    def validate(self) -> None:
        for name in ("alpha_up", "alpha_down", "spread_alpha"):
            v = getattr(self, name)
            _require(0.0 < v <= 1.0, f"baseline.{name}", "must be in (0, 1]")
        _require(self.k_mad > 0, "baseline.k_mad", "must be positive")
        _require(self.min_delta_db > 0, "baseline.min_delta_db", "must be positive")
        _require(
            0.0 < self.hot_bin_damping <= 1.0,
            "baseline.hot_bin_damping",
            "must be in (0, 1]",
        )


@dataclass(frozen=True)
class BandConfig:
    """One frequency region in the sweep plan."""

    name: str = "band"
    start_hz: float = 0.0
    stop_hz: float = 0.0
    enabled: bool = True
    # Priority breaks ties between simultaneously due steps; use
    # revisit_s to control how often a band is actually rescanned.
    priority: int = 1
    dwell_s: float = 0.5
    revisit_s: float = 0.0

    def validate(self) -> None:
        _require(bool(self.name), "sweep.bands[].name", "must be non-empty")
        _require(
            0 < self.start_hz < self.stop_hz,
            f"sweep.bands[{self.name}]",
            "requires 0 < start_hz < stop_hz",
        )
        _require(self.dwell_s > 0, f"sweep.bands[{self.name}].dwell_s", "must be positive")
        _require(
            self.revisit_s >= 0, f"sweep.bands[{self.name}].revisit_s", "must be >= 0"
        )
        _require(self.priority >= 1, f"sweep.bands[{self.name}].priority", "must be >= 1")


@dataclass(frozen=True)
class ExclusionRange:
    """Frequency range whose detections are suppressed (known emitters)."""

    start_hz: float = 0.0
    stop_hz: float = 0.0

    def validate(self) -> None:
        _require(
            0 < self.start_hz < self.stop_hz,
            "sweep.exclusions[]",
            "requires 0 < start_hz < stop_hz",
        )

    def contains(self, freq_hz: float) -> bool:
        return self.start_hz <= freq_hz <= self.stop_hz


@dataclass(frozen=True)
class AdaptiveSweepConfig:
    """Adaptive rescanning: steps with recent activity are revisited sooner."""

    enabled: bool = True
    hot_revisit_s: float = 1.0
    hot_visits: int = 3

    def validate(self) -> None:
        _require(
            self.hot_revisit_s >= 0, "sweep.adaptive.hot_revisit_s", "must be >= 0"
        )
        _require(self.hot_visits >= 1, "sweep.adaptive.hot_visits", "must be >= 1")


@dataclass(frozen=True)
class CaptureConfig:
    """Short triggered IQ captures. Disabled by default (privacy/storage)."""

    enabled: bool = False
    dir: str = "data/captures"
    max_files: int = 20
    chunks_per_event: int = 2

    def validate(self) -> None:
        _require(self.max_files >= 1, "sweep.capture.max_files", "must be >= 1")
        _require(
            self.chunks_per_event >= 1,
            "sweep.capture.chunks_per_event",
            "must be >= 1",
        )


@dataclass(frozen=True)
class SweepConfig:
    # Fraction of the sample rate treated as usable analysis bandwidth
    # (HackRF baseband filter edges roll off; steps overlap accordingly).
    usable_fraction: float = 0.75
    # Chunks discarded after each retune (LO settling / filter transient).
    settle_chunks: int = 1
    environments_dir: str = "data/environments"
    bands: tuple[BandConfig, ...] = ()
    exclusions: tuple[ExclusionRange, ...] = ()
    adaptive: AdaptiveSweepConfig = field(default_factory=AdaptiveSweepConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    def validate(self) -> None:
        _require(
            0.1 < self.usable_fraction <= 1.0,
            "sweep.usable_fraction",
            "must be in (0.1, 1.0]",
        )
        _require(self.settle_chunks >= 0, "sweep.settle_chunks", "must be >= 0")
        names = [b.name for b in self.bands]
        _require(
            len(names) == len(set(names)), "sweep.bands", "band names must be unique"
        )
        for band in self.bands:
            band.validate()
        for excl in self.exclusions:
            excl.validate()
        self.adaptive.validate()
        self.capture.validate()

    def enabled_bands(self) -> tuple[BandConfig, ...]:
        return tuple(b for b in self.bands if b.enabled)

    def is_excluded(self, freq_hz: float) -> bool:
        return any(e.contains(freq_hz) for e in self.exclusions)


@dataclass(frozen=True)
class FusionWeights:
    """Evidence weights and penalties for the confidence fusion engine.

    confidence = sigmoid(sum(w_i * s_i) - sum(p_j * b_j) - bias)
    """

    anomaly: float = 1.0
    burst: float = 1.5
    hopping: float = 2.0
    emi: float = 1.0  # supporting evidence only; standalone EMI is capped
    multiband: float = 2.0
    ml: float = 1.5  # calibrated classifier P(drone) / P(background)
    ml_anomaly: float = 0.5  # Isolation Forest feature-space novelty
    wifi_penalty: float = 1.5
    bt_penalty: float = 1.2
    fixed_penalty: float = 1.5
    clipping_penalty: float = 2.0
    single_obs_penalty: float = 1.0
    bias: float = 2.0

    def validate(self) -> None:
        for name in (
            "anomaly", "burst", "hopping", "emi", "multiband",
            "ml", "ml_anomaly",
            "wifi_penalty", "bt_penalty", "fixed_penalty",
            "clipping_penalty", "single_obs_penalty",
        ):
            _require(getattr(self, name) >= 0, f"analytics.weights.{name}", "must be >= 0")


@dataclass(frozen=True)
class HoppingConfig:
    window_s: float = 3.0
    min_channels: int = 5
    max_channel_bw_hz: float = 2_000_000.0
    channel_tolerance_hz: float = 500_000.0

    def validate(self) -> None:
        _require(self.window_s > 0, "analytics.hopping.window_s", "must be positive")
        _require(self.min_channels >= 3, "analytics.hopping.min_channels", "must be >= 3")
        _require(
            self.max_channel_bw_hz > 0,
            "analytics.hopping.max_channel_bw_hz",
            "must be positive",
        )


@dataclass(frozen=True)
class EMIConfig:
    min_harmonics: int = 4
    spacing_tolerance: float = 0.15  # relative deviation allowed between spacings
    max_peak_bw_hz: float = 500_000.0

    def validate(self) -> None:
        _require(self.min_harmonics >= 3, "analytics.emi.min_harmonics", "must be >= 3")
        _require(
            0 < self.spacing_tolerance < 1,
            "analytics.emi.spacing_tolerance",
            "must be in (0, 1)",
        )


@dataclass(frozen=True)
class BackgroundConfig:
    wifi_bw_min_hz: float = 8_000_000.0
    wifi_bw_max_hz: float = 25_000_000.0
    wifi_grid_tolerance_hz: float = 3_000_000.0
    fixed_min_age_s: float = 60.0
    fixed_min_occupancy: float = 0.95

    def validate(self) -> None:
        _require(
            0 < self.wifi_bw_min_hz < self.wifi_bw_max_hz,
            "analytics.background.wifi_bw_min_hz",
            "requires 0 < min < max",
        )
        _require(
            0 < self.fixed_min_occupancy <= 1.0,
            "analytics.background.fixed_min_occupancy",
            "must be in (0, 1]",
        )


@dataclass(frozen=True)
class BurstConfig:
    min_bursts: int = 2  # per chunk, from the sub-chunk temporal profile
    duty_min: float = 0.02
    duty_max: float = 0.7
    period_cv_max: float = 0.5

    def validate(self) -> None:
        _require(self.min_bursts >= 1, "analytics.burst.min_bursts", "must be >= 1")
        _require(
            0 <= self.duty_min < self.duty_max <= 1.0,
            "analytics.burst.duty_min",
            "requires 0 <= duty_min < duty_max <= 1",
        )


@dataclass(frozen=True)
class MLConfig:
    """Optional machine-learning subsystem (Stage 6).

    Disabled by default and stays disabled until the operator trains a
    model on their own labeled data (`drone4rf ml train`). Inference
    is CPU-only (scikit-learn).
    """

    enabled: bool = False
    model_path: str = "data/ml/model.joblib"
    # Optional Isolation Forest novelty model ("" = none).
    anomaly_model_path: str = ""
    # Unknown-class rejection: below this calibrated max-probability the
    # classifier contributes nothing to fusion.
    min_calibrated_confidence: float = 0.6

    def validate(self) -> None:
        _require(
            0.5 <= self.min_calibrated_confidence < 1.0,
            "ml.min_calibrated_confidence",
            "must be in [0.5, 1.0)",
        )
        _require(bool(self.model_path), "ml.model_path", "must be non-empty")


@dataclass(frozen=True)
class AnalyticsConfig:
    enabled: bool = True
    assessment_interval_s: float = 2.0
    # Re-emission hysteresis: an entity's category changes are announced
    # at most this often unless the category ESCALATES (rank increases).
    emit_cooldown_s: float = 30.0
    min_track_hits: int = 3
    multiband_window_s: float = 5.0
    multiband_min_separation_hz: float = 200_000_000.0
    # Category thresholds on the fused confidence in [0, 1].
    background_max: float = 0.20
    unclassified_max: float = 0.45
    possible_max: float = 0.65
    probable_max: float = 0.85
    weights: FusionWeights = field(default_factory=FusionWeights)
    hopping: HoppingConfig = field(default_factory=HoppingConfig)
    emi: EMIConfig = field(default_factory=EMIConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    burst: BurstConfig = field(default_factory=BurstConfig)

    def validate(self) -> None:
        _require(
            self.assessment_interval_s > 0,
            "analytics.assessment_interval_s",
            "must be positive",
        )
        _require(self.min_track_hits >= 1, "analytics.min_track_hits", "must be >= 1")
        _require(self.emit_cooldown_s >= 0, "analytics.emit_cooldown_s", "must be >= 0")
        _require(
            0
            < self.background_max
            < self.unclassified_max
            < self.possible_max
            < self.probable_max
            < 1.0,
            "analytics.background_max",
            "category thresholds must be strictly increasing within (0, 1)",
        )
        self.weights.validate()
        self.hopping.validate()
        self.emi.validate()
        self.background.validate()
        self.burst.validate()


@dataclass(frozen=True)
class TrackingConfig:
    match_tolerance_hz: float = 200_000.0
    promote_hits: int = 5
    drop_after_misses: int = 10

    def validate(self) -> None:
        _require(self.match_tolerance_hz > 0, "tracking.match_tolerance_hz", "must be positive")
        _require(self.promote_hits >= 2, "tracking.promote_hits", "must be >= 2")
        _require(self.drop_after_misses >= 1, "tracking.drop_after_misses", "must be >= 1")


@dataclass(frozen=True)
class StorageConfig:
    database_path: str = "data/events.db"
    log_level: str = "INFO"

    def validate(self) -> None:
        _require(
            self.log_level.upper() in {"DEBUG", "INFO", "WARNING", "ERROR"},
            "storage.log_level",
            "must be DEBUG, INFO, WARNING, or ERROR",
        )


@dataclass(frozen=True)
class AppConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    def validate(self) -> None:
        self.device.validate()
        self.dsp.validate()
        self.detection.validate()
        self.baseline.validate()
        self.tracking.validate()
        self.sweep.validate()
        self.analytics.validate()
        self.ml.validate()
        self.storage.validate()
        for band in self.sweep.enabled_bands():
            # Every sweep window must be tunable by the configured device.
            _require(
                band.stop_hz - band.start_hz >= 0
                and band.start_hz > self.device.sample_rate / 2,
                f"sweep.bands[{band.name}]",
                "band must sit above half the sample rate",
            )


def _build(cls: type, data: Any, key: str) -> Any:
    """Construct a (possibly nested) dataclass from a YAML mapping."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"config error at '{key}': expected a mapping")
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"config error at '{key}': unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = fields[name].type
        subkey = f"{key}.{name}"
        if isinstance(value, dict):
            subcls = _NESTED.get((cls, name))
            if subcls is None:
                raise ConfigError(f"config error at '{subkey}': unexpected mapping")
            kwargs[name] = _build(subcls, value, subkey)
        elif isinstance(value, list):
            elemcls = _LIST_ELEMS.get((cls, name))
            if elemcls is None:
                raise ConfigError(f"config error at '{subkey}': unexpected list")
            items = []
            for i, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ConfigError(
                        f"config error at '{subkey}[{i}]': expected a mapping"
                    )
                items.append(_build(elemcls, item, f"{subkey}[{i}]"))
            kwargs[name] = tuple(items)
        else:
            kwargs[name] = _coerce(value, str(ftype), subkey)
    return cls(**kwargs)


_NESTED: dict[tuple[type, str], type] = {
    (AppConfig, "device"): DeviceConfig,
    (AppConfig, "dsp"): DSPConfig,
    (AppConfig, "detection"): DetectionConfig,
    (AppConfig, "baseline"): BaselineConfig,
    (AppConfig, "tracking"): TrackingConfig,
    (AppConfig, "sweep"): SweepConfig,
    (AppConfig, "analytics"): AnalyticsConfig,
    (AppConfig, "ml"): MLConfig,
    (AppConfig, "storage"): StorageConfig,
    (DetectionConfig, "energy"): EnergyDetectorConfig,
    (DetectionConfig, "cfar"): CFARDetectorConfig,
    (SweepConfig, "adaptive"): AdaptiveSweepConfig,
    (SweepConfig, "capture"): CaptureConfig,
    (AnalyticsConfig, "weights"): FusionWeights,
    (AnalyticsConfig, "hopping"): HoppingConfig,
    (AnalyticsConfig, "emi"): EMIConfig,
    (AnalyticsConfig, "background"): BackgroundConfig,
    (AnalyticsConfig, "burst"): BurstConfig,
}

_LIST_ELEMS: dict[tuple[type, str], type] = {
    (SweepConfig, "bands"): BandConfig,
    (SweepConfig, "exclusions"): ExclusionRange,
}


def _coerce(value: Any, type_name: str, key: str) -> Any:
    """Coerce YAML scalars to the annotated field type with clear errors."""
    try:
        if "bool" in type_name:
            if not isinstance(value, bool):
                raise TypeError("expected true/false")
            return value
        if "int" in type_name:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expected an integer")
            if isinstance(value, float) and not value.is_integer():
                raise TypeError("expected an integer")
            return int(value)
        if "float" in type_name:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expected a number")
            return float(value)
        if "str" in type_name:
            if not isinstance(value, str):
                raise TypeError("expected a string")
            return value
    except TypeError as exc:
        raise ConfigError(f"config error at '{key}': {exc} (got {value!r})") from None
    return value


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration; defaults are used when path is None."""
    if path is None:
        cfg = AppConfig()
    else:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"configuration file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = _build(AppConfig, raw, "root")
    cfg.validate()
    return cfg
