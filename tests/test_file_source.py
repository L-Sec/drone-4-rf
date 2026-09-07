import numpy as np
import pytest

from drone4rf.sdr.file_source import FileSource


def _write_cf32(path, samples: np.ndarray) -> None:
    inter = np.empty(2 * len(samples), dtype=np.float32)
    inter[0::2] = samples.real
    inter[1::2] = samples.imag
    inter.tofile(path)


def test_cf32_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    samples = (rng.normal(0, 0.1, 4096) + 1j * rng.normal(0, 0.1, 4096)).astype(
        np.complex64
    )
    p = tmp_path / "cap.cf32"
    _write_cf32(p, samples)
    src = FileSource(p, iq_format="cf32")
    with src:
        chunk = src.read_chunk(4096)
    assert chunk is not None
    assert np.allclose(chunk.samples, samples)


def test_cs8_hackrf_transfer_format(tmp_path) -> None:
    raw = np.array([127, 0, -127, 0, 0, 127, 0, -127], dtype=np.int8)
    p = tmp_path / "cap.cs8"
    raw.tofile(p)
    src = FileSource(p, iq_format="cs8")
    with src:
        chunk = src.read_chunk(4)
    assert chunk is not None
    expected = np.array([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex64)
    assert np.allclose(chunk.samples, expected)


def test_stream_ends_without_loop(tmp_path) -> None:
    samples = np.ones(1000, dtype=np.complex64)
    p = tmp_path / "cap.cf32"
    _write_cf32(p, samples)
    src = FileSource(p, iq_format="cf32")
    with src:
        first = src.read_chunk(2048)  # tail gets zero-padded to chunk size
        second = src.read_chunk(2048)
    assert first is not None and len(first.samples) == 2048
    assert np.all(first.samples[1000:] == 0)
    assert second is None


def test_loop_wraps(tmp_path) -> None:
    samples = np.ones(100, dtype=np.complex64)
    p = tmp_path / "cap.cf32"
    _write_cf32(p, samples)
    src = FileSource(p, iq_format="cf32", loop=True)
    with src:
        for _ in range(5):
            assert src.read_chunk(100) is not None


def test_missing_file_raises(tmp_path) -> None:
    src = FileSource(tmp_path / "nope.cf32")
    with pytest.raises(FileNotFoundError):
        src.open()


def test_bad_format_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="iq_format"):
        FileSource(tmp_path / "x", iq_format="wav")
