from pathlib import Path

import pytest

from drone4rf.config import AppConfig, ConfigError, load_config


def test_defaults_are_valid() -> None:
    cfg = load_config(None)
    assert cfg.device.driver == "hackrf"
    assert cfg.dsp.fft_size == 4096


def test_load_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "device:\n  sample_rate: 8000000\n  lna_gain_db: 24\n"
        "dsp:\n  fft_size: 2048\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.device.sample_rate == 8e6
    assert cfg.device.lna_gain_db == 24
    assert cfg.dsp.fft_size == 2048


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("no/such/file.yaml")


def test_hackrf_sample_rate_bounds(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("device:\n  sample_rate: 1000000\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="device.sample_rate"):
        load_config(p)


def test_hackrf_lna_gain_step(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("device:\n  lna_gain_db: 13\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="device.lna_gain_db"):
        load_config(p)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("device:\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bogus_key"):
        load_config(p)


def test_fft_size_power_of_two() -> None:
    from drone4rf.config import DSPConfig

    with pytest.raises(ConfigError, match="dsp.fft_size"):
        DSPConfig(fft_size=1000).validate()


def test_sweep_bands_parse_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "sweep:\n"
        "  bands:\n"
        "    - name: ism\n"
        "      start_hz: 2400000000\n"
        "      stop_hz: 2483500000\n"
        "      priority: 3\n"
        "    - name: video\n"
        "      start_hz: 5725000000\n"
        "      stop_hz: 5875000000\n"
        "      enabled: false\n"
        "  exclusions:\n"
        "    - start_hz: 2431000000\n"
        "      stop_hz: 2442000000\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert len(cfg.sweep.bands) == 2
    assert cfg.sweep.bands[0].priority == 3
    assert cfg.sweep.enabled_bands() == (cfg.sweep.bands[0],)
    assert cfg.sweep.is_excluded(2.4365e9)
    assert not cfg.sweep.is_excluded(2.45e9)


def test_sweep_band_start_after_stop_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "sweep:\n"
        "  bands:\n"
        "    - name: bad\n"
        "      start_hz: 2483500000\n"
        "      stop_hz: 2400000000\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="start_hz < stop_hz"):
        load_config(p)


def test_sweep_duplicate_band_names_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "sweep:\n"
        "  bands:\n"
        "    - {name: x, start_hz: 2400000000, stop_hz: 2405000000}\n"
        "    - {name: x, start_hz: 5725000000, stop_hz: 5730000000}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unique"):
        load_config(p)


def test_sweep_unknown_band_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "sweep:\n"
        "  bands:\n"
        "    - {name: x, start_hz: 2400000000, stop_hz: 2405000000, wat: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="wat"):
        load_config(p)


def test_repo_default_yaml_is_valid() -> None:
    repo_cfg = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(repo_cfg)
    assert isinstance(cfg, AppConfig)


def test_config_resolution_order(tmp_path: Path, monkeypatch) -> None:
    """A fresh clone must work with no --config flag."""
    from drone4rf.cli import resolve_config_path

    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None) is None  # nothing present -> built-ins

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "default.yaml").write_text("device:\n  lna_gain_db: 8\n")
    assert resolve_config_path(None) == str(Path("config/default.yaml"))

    # A user's own local.yaml takes precedence over the repo defaults.
    (cfg_dir / "local.yaml").write_text("device:\n  lna_gain_db: 24\n")
    assert resolve_config_path(None) == str(Path("config/local.yaml"))

    # An explicit path always wins.
    assert resolve_config_path("elsewhere.yaml") == "elsewhere.yaml"


def test_repo_example_yaml_is_valid() -> None:
    """The template users are told to copy must load without editing."""
    repo_cfg = Path(__file__).resolve().parents[1] / "config" / "local.example.yaml"
    cfg = load_config(repo_cfg)
    assert isinstance(cfg, AppConfig)
    assert cfg.dsp.overlap == 0.25  # the CPU-saving tweak the template exists for
