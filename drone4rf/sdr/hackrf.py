"""HackRF One receive source via SoapySDR.

SoapySDR is imported lazily so the rest of the application (simulator,
file playback, tests) works on machines without the bindings installed.
Only the RX direction is ever configured; no TX API is touched.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from drone4rf.config import DeviceConfig
from drone4rf.sdr.base import IQChunk, SDRSource

log = logging.getLogger(__name__)

_SOAPY_IMPORT_HELP = (
    "SoapySDR Python bindings not found. They are not installable via pip:\n"
    "  Windows: install PothosSDR and add its python dir to PYTHONPATH\n"
    "  Debian/Ubuntu: sudo apt install python3-soapysdr soapysdr-module-hackrf\n"
    "The simulated (--source sim) and file (--source file) sources work "
    "without SoapySDR."
)


_pothos_wired = False


def _wire_pothos_sdr() -> bool:
    """On Windows, make a PothosSDR install importable without manual setup.

    PothosSDR ships _SoapySDR.pyd built against one specific CPython
    (currently 3.9) plus dependent DLLs in its bin directory. Python 3.8+
    does not resolve extension-module DLL dependencies via PATH, so the
    bin dir must be registered with os.add_dll_directory explicitly.
    Returns True if a matching install was wired up.
    """
    global _pothos_wired
    if _pothos_wired:
        return True
    if sys.platform != "win32":
        return False
    root = Path(os.environ.get("POTHOSSDR_ROOT", r"C:\Program Files\PothosSDR"))
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = root / "lib" / ver / "site-packages"
    if not (site / "_SoapySDR.pyd").exists():
        shipped = sorted(p.name for p in (root / "lib").glob("python*"))
        if shipped:
            log.warning(
                "PothosSDR at %s ships bindings for %s but this interpreter "
                "is %s - run drone4rf under a matching Python for live "
                "HackRF use",
                root,
                "/".join(shipped),
                ver,
            )
        return False
    # add_dll_directory covers _SoapySDR.pyd's own dependencies, but the
    # SoapySDR core loads driver modules (HackRFSupport.dll, ...) via
    # plain LoadLibrary, whose dependency search uses PATH - so bin must
    # be on the process PATH as well.
    os.add_dll_directory(str(root / "bin"))
    os.environ["PATH"] = str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
    if str(site) not in sys.path:
        sys.path.append(str(site))
    _pothos_wired = True
    log.debug("wired PothosSDR bindings from %s", site)
    return True


def _import_soapy():  # -> module
    # Wire BEFORE the first import attempt: the PothosSDR installer
    # registers its site-packages machine-wide via the PEP 514 registry
    # PythonPath key, so `import SoapySDR` can succeed even though the
    # DLL search path for the driver modules (HackRFSupport.dll ->
    # hackrf.dll etc.) has not been set up yet. Wiring is a no-op when
    # no matching PothosSDR install exists.
    _wire_pothos_sdr()
    try:
        import SoapySDR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(_SOAPY_IMPORT_HELP) from exc
    return SoapySDR


def list_devices() -> list[dict[str, str]]:
    """Enumerate SoapySDR devices; empty list if bindings are missing."""
    try:
        SoapySDR = _import_soapy()
    except RuntimeError as exc:
        log.warning(str(exc))
        return []
    return [dict(kw) for kw in SoapySDR.Device.enumerate()]


class HackRFSource(SDRSource):
    """Streams complex64 RX samples from a HackRF One (or any Soapy driver).

    Handles stream overflows (counted, flagged on the next chunk) and a
    bounded number of automatic reopen attempts after device errors.
    """

    name = "hackrf"

    def __init__(self, cfg: DeviceConfig, max_reconnects: int = 3) -> None:
        super().__init__()
        self._cfg = cfg
        self._max_reconnects = max_reconnects
        self._soapy = None
        self._dev = None
        self._stream = None
        self._seq = 0
        self._pending_overflow = False
        self._center_hz = cfg.center_freq_hz

    def open(self) -> None:
        self._soapy = _import_soapy()
        self._open_device()

    def _open_device(self) -> None:
        assert self._soapy is not None
        SoapySDR = self._soapy
        cfg = self._cfg
        self._dev = SoapySDR.Device({"driver": cfg.driver})
        rx = SoapySDR.SOAPY_SDR_RX
        self._dev.setSampleRate(rx, 0, cfg.sample_rate)
        # Apply crystal ppm correction directly to the tuned frequency:
        # not all Soapy drivers expose setFrequencyCorrection.
        corrected = self._center_hz * (1.0 + cfg.freq_correction_ppm * 1e-6)
        self._dev.setFrequency(rx, 0, corrected)
        if cfg.driver == "hackrf":
            self._dev.setGain(rx, 0, "LNA", cfg.lna_gain_db)
            self._dev.setGain(rx, 0, "VGA", cfg.vga_gain_db)
            self._dev.setGain(rx, 0, "AMP", 14 if cfg.amp_enabled else 0)
        else:
            # Generic drivers: single overall gain, conservative.
            self._dev.setGain(rx, 0, float(cfg.lna_gain_db + cfg.vga_gain_db) / 2)
        self._stream = self._dev.setupStream(rx, SoapySDR.SOAPY_SDR_CF32)
        self._dev.activateStream(self._stream)
        log.info(
            "opened %s: %.4f MHz @ %.1f MS/s, LNA %d dB, VGA %d dB, amp %s",
            cfg.driver,
            self._center_hz / 1e6,
            cfg.sample_rate / 1e6,
            cfg.lna_gain_db,
            cfg.vga_gain_db,
            "on" if cfg.amp_enabled else "off",
        )

    def read_chunk(self, num_samples: int) -> IQChunk | None:
        if self._dev is None or self._stream is None:
            raise RuntimeError("HackRFSource.read_chunk called before open()")
        assert self._soapy is not None
        SoapySDR = self._soapy

        buf = np.empty(num_samples, dtype=np.complex64)
        filled = 0
        t0 = time.time()
        while filled < num_samples:
            sr = self._dev.readStream(
                self._stream, [buf[filled:]], num_samples - filled, timeoutUs=500_000
            )
            if sr.ret > 0:
                filled += sr.ret
            elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                # Samples were lost inside the driver; note it and continue.
                self.health.overflows += 1
                self._pending_overflow = True
            elif sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                self.health.notes.append("read timeout")
                log.warning("readStream timeout; device stalled?")
            else:
                log.error("readStream error %d; attempting device reopen", sr.ret)
                if not self._reopen():
                    return None
        overflow = self._pending_overflow
        self._pending_overflow = False
        self._seq += 1
        self.health.chunks_produced += 1
        return IQChunk(
            samples=buf,
            center_hz=self._center_hz,
            sample_rate=self._cfg.sample_rate,
            timestamp=t0,
            seq=self._seq,
            overflow=overflow,
        )

    @property
    def supports_retune(self) -> bool:
        return True

    def retune(self, center_hz: float) -> None:
        """Retune the RX LO. Called from the acquisition thread only."""
        if self._dev is None:
            raise RuntimeError("HackRFSource.retune called before open()")
        assert self._soapy is not None
        corrected = center_hz * (1.0 + self._cfg.freq_correction_ppm * 1e-6)
        self._dev.setFrequency(self._soapy.SOAPY_SDR_RX, 0, corrected)
        self._center_hz = center_hz
        self.health.retunes += 1

    def _reopen(self) -> bool:
        """Try to recover from a device error (e.g. USB glitch)."""
        if self.health.reconnects >= self._max_reconnects:
            log.error("giving up after %d reconnect attempts", self.health.reconnects)
            return False
        self.health.reconnects += 1
        self._close_stream()
        time.sleep(1.0)
        try:
            self._open_device()
            return True
        except Exception:
            log.exception("device reopen failed")
            return False

    def _close_stream(self) -> None:
        if self._dev is not None and self._stream is not None:
            try:
                self._dev.deactivateStream(self._stream)
                self._dev.closeStream(self._stream)
            except Exception:
                log.exception("error closing stream")
        self._stream = None
        self._dev = None

    def close(self) -> None:
        self._close_stream()
