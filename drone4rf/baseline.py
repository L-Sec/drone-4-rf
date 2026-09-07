"""Adaptive per-bin RF background model.

Asymmetric exponentially weighted tracker per PSD bin:

- level rises slowly (alpha_up) and falls quickly (alpha_down), so a
  disappearing signal is forgotten fast but a NEW persistent emitter is
  absorbed only over many frames;
- bins currently ABOVE the detection threshold adapt with an extra
  damping factor - the anti-poisoning guard: an anomaly cannot promote
  itself into the background at normal speed;
- spread is an exponentially weighted mean absolute deviation;
- threshold = level + max(k_mad * spread * MAD_TO_SIGMA, min_delta_db).

The model can be frozen (calibrated operation) and saved/loaded as .npz
with tuning metadata; a mismatched baseline is refused at load.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from drone4rf.config import BaselineConfig
from drone4rf.dsp.noise_floor import MAD_TO_SIGMA

log = logging.getLogger(__name__)


class BaselineMismatchError(ValueError):
    """Raised when a stored baseline does not match the current tuning."""


class BaselineModel:
    def __init__(
        self,
        n_bins: int,
        cfg: BaselineConfig,
        center_hz: float,
        sample_rate: float,
    ) -> None:
        self.n_bins = n_bins
        self.cfg = cfg
        self.center_hz = center_hz
        self.sample_rate = sample_rate
        self.frozen = False
        self.frames_seen = 0
        self._level: np.ndarray | None = None  # per-bin level (dB)
        self._spread: np.ndarray | None = None  # per-bin deviation (dB)

    @property
    def ready(self) -> bool:
        """True once enough frames have been absorbed to trust thresholds."""
        return self.frames_seen >= 5

    def update(self, psd_db: np.ndarray) -> None:
        if len(psd_db) != self.n_bins:
            raise ValueError(f"expected {self.n_bins} bins, got {len(psd_db)}")
        if self.frozen:
            return
        if self._level is None:
            self._level = psd_db.astype(np.float64).copy()
            self._spread = np.full(self.n_bins, 1.0)
            self.frames_seen = 1
            return
        assert self._spread is not None
        cfg = self.cfg
        # Anti-poisoning: bins currently above the detection threshold
        # ("hot") adapt with extra damping applied to BOTH level and
        # spread. Damping only the level is not enough - an anomaly would
        # inflate the spread, raise the threshold past itself, lose hot
        # status, and then be absorbed at full speed.
        hot = psd_db > self.threshold_db()
        lvl_alpha = np.where(psd_db > self._level, cfg.alpha_up, cfg.alpha_down)
        lvl_alpha = np.where(hot, lvl_alpha * cfg.hot_bin_damping, lvl_alpha)
        self._level += lvl_alpha * (psd_db - self._level)
        # Winsorize deviations so a single outlier frame cannot blow up
        # the spread estimate (and thereby desensitize the threshold).
        dev = np.minimum(np.abs(psd_db - self._level), 3.0 * self._spread)
        sp_alpha = np.where(hot, cfg.spread_alpha * cfg.hot_bin_damping, cfg.spread_alpha)
        self._spread += sp_alpha * (dev - self._spread)
        # Keep spread away from zero so thresholds never collapse.
        np.maximum(self._spread, 0.1, out=self._spread)
        self.frames_seen += 1

    def threshold_db(self) -> np.ndarray:
        """Per-bin detection threshold in dB."""
        if self._level is None or self._spread is None:
            raise RuntimeError("baseline has no data yet")
        margin = np.maximum(
            self.cfg.k_mad * self._spread * MAD_TO_SIGMA, self.cfg.min_delta_db
        )
        return self._level + margin

    def level_db(self) -> np.ndarray:
        if self._level is None:
            raise RuntimeError("baseline has no data yet")
        return self._level.copy()

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self._level is None or self._spread is None:
            raise RuntimeError("cannot save an empty baseline")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            level=self._level,
            spread=self._spread,
            n_bins=self.n_bins,
            center_hz=self.center_hz,
            sample_rate=self.sample_rate,
            frames_seen=self.frames_seen,
        )
        log.info("baseline saved to %s (%d frames)", p, self.frames_seen)

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: BaselineConfig,
        center_hz: float,
        sample_rate: float,
        n_bins: int,
    ) -> "BaselineModel":
        with np.load(Path(path)) as data:
            if (
                int(data["n_bins"]) != n_bins
                or float(data["center_hz"]) != center_hz
                or float(data["sample_rate"]) != sample_rate
            ):
                raise BaselineMismatchError(
                    f"stored baseline ({float(data['center_hz'])/1e6:.3f} MHz, "
                    f"{float(data['sample_rate'])/1e6:.1f} MS/s, "
                    f"{int(data['n_bins'])} bins) does not match current tuning "
                    f"({center_hz/1e6:.3f} MHz, {sample_rate/1e6:.1f} MS/s, "
                    f"{n_bins} bins)"
                )
            model = cls(n_bins, cfg, center_hz, sample_rate)
            model._level = np.asarray(data["level"], dtype=np.float64)
            model._spread = np.asarray(data["spread"], dtype=np.float64)
            model.frames_seen = int(data["frames_seen"])
        return model


class BaselineBank:
    """A BaselineModel per sweep step, keyed by (center, sample_rate).

    Banks are the unit of a *named environment*: save() writes one .npz
    per step plus a manifest.json into a directory; load() restores the
    whole set. Models for steps never visited before are created lazily
    (and inherit the bank's frozen state, so a frozen environment stays
    frozen even for newly added bands).
    """

    def __init__(self, cfg: BaselineConfig, n_bins: int) -> None:
        self.cfg = cfg
        self.n_bins = n_bins
        self.frozen = False
        self._models: dict[str, BaselineModel] = {}

    @staticmethod
    def _key(center_hz: float, sample_rate: float) -> str:
        return f"{int(round(center_hz))}_{int(round(sample_rate))}"

    def get(self, center_hz: float, sample_rate: float) -> BaselineModel:
        key = self._key(center_hz, sample_rate)
        model = self._models.get(key)
        if model is None:
            model = BaselineModel(self.n_bins, self.cfg, center_hz, sample_rate)
            model.frozen = self.frozen
            self._models[key] = model
        return model

    @property
    def models(self) -> dict[str, BaselineModel]:
        return dict(self._models)

    def freeze_all(self) -> None:
        self.frozen = True
        for model in self._models.values():
            model.frozen = True

    def any_ready(self) -> bool:
        return any(m.ready for m in self._models.values())

    def save(self, directory: str | Path, environment: str = "default") -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        saved = []
        for key, model in self._models.items():
            if model.frames_seen == 0:
                continue
            model.save(d / f"{key}.npz")
            saved.append(key)
        manifest = {
            "environment": environment,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "n_bins": self.n_bins,
            "steps": saved,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("environment '%s' saved: %d step baselines -> %s",
                 environment, len(saved), d)

    @classmethod
    def load(cls, directory: str | Path, cfg: BaselineConfig, n_bins: int) -> "BaselineBank":
        d = Path(directory)
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no environment manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["n_bins"]) != n_bins:
            raise BaselineMismatchError(
                f"environment was built with fft_size {manifest['n_bins']}, "
                f"current config uses {n_bins}"
            )
        bank = cls(cfg, n_bins)
        for key in manifest["steps"]:
            path = d / f"{key}.npz"
            with np.load(path) as data:
                center = float(data["center_hz"])
                fs = float(data["sample_rate"])
            bank._models[key] = BaselineModel.load(path, cfg, center, fs, n_bins)
        return bank
