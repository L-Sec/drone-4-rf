"""Entry point for the installed Windows desktop application."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def bundled_root() -> Path:
    """Return the read-only application payload directory."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def user_root() -> Path:
    """Return the per-user writable application directory."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Drone4RF"
    return Path.home() / "AppData" / "Local" / "Drone4RF"


def prepare_user_files(payload: Path, root: Path) -> Path:
    """Create writable runtime directories without replacing local settings."""
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    shipped_config = payload / "config"
    shutil.copy2(shipped_config / "default.yaml", config_dir / "default.yaml")
    example = config_dir / "local.example.yaml"
    if not example.exists():
        shutil.copy2(shipped_config / "local.example.yaml", example)

    local = config_dir / "local.yaml"
    return local if local.is_file() else config_dir / "default.yaml"


def wire_bundled_sdr(payload: Path) -> None:
    """Expose the bundled receive-only SoapySDR/HackRF runtime."""
    pothos = payload / "pothos"
    bin_dir = pothos / "bin"
    if not bin_dir.is_dir():
        return
    os.environ["POTHOSSDR_ROOT"] = str(pothos)
    os.environ["SOAPY_SDR_PLUGIN_PATH"] = str(
        pothos / "lib" / "SoapySDR" / "modules0.8"
    )
    os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def main() -> int:
    payload = bundled_root()
    root = user_root()
    config_path = prepare_user_files(payload, root)
    os.chdir(root)
    wire_bundled_sdr(payload)

    from drone4rf.config import load_config

    cfg = load_config(config_path)
    if "--self-test" in sys.argv[1:]:
        from drone4rf.sdr.hackrf import _import_soapy

        _import_soapy()
        return 0

    from drone4rf.gui.app import run_gui

    return run_gui(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
