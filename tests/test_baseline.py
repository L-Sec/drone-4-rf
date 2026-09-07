import numpy as np
import pytest

from drone4rf.baseline import BaselineBank, BaselineMismatchError, BaselineModel
from drone4rf.config import BaselineConfig

N = 512
CENTER = 2_437e6
FS = 10e6


def _model() -> BaselineModel:
    return BaselineModel(N, BaselineConfig(), CENTER, FS)


def _train(model: BaselineModel, rng: np.random.Generator, frames: int = 30) -> None:
    for _ in range(frames):
        model.update(rng.normal(-80.0, 1.0, N))


def test_ready_after_enough_frames() -> None:
    model = _model()
    rng = np.random.default_rng(0)
    assert not model.ready
    _train(model, rng, frames=6)
    assert model.ready


def test_signal_exceeds_threshold_after_training() -> None:
    model = _model()
    rng = np.random.default_rng(1)
    _train(model, rng)
    thr = model.threshold_db()
    frame = rng.normal(-80.0, 1.0, N)
    frame[100:105] = -55.0  # +25 dB injected signal
    assert np.all(frame[100:105] > thr[100:105])
    # Noise-only bins stay (almost entirely) below threshold.
    noise_mask = np.ones(N, dtype=bool)
    noise_mask[100:105] = False
    assert np.mean(frame[noise_mask] > thr[noise_mask]) < 0.01


def test_poisoning_resistance() -> None:
    """A strong new persistent signal must be absorbed only very slowly."""
    model = _model()
    rng = np.random.default_rng(2)
    _train(model, rng)
    level_before = model.level_db()[200:210].mean()
    for _ in range(50):
        frame = rng.normal(-80.0, 1.0, N)
        frame[200:210] = -50.0  # persistent +30 dB emitter
        model.update(frame)
    rise = model.level_db()[200:210].mean() - level_before
    assert rise < 6.0  # far from the 30 dB step after 50 frames
    # And the signal still trips the threshold.
    frame = rng.normal(-80.0, 1.0, N)
    frame[200:210] = -50.0
    assert np.all(frame[200:210] > model.threshold_db()[200:210])


def test_frozen_baseline_does_not_adapt() -> None:
    model = _model()
    rng = np.random.default_rng(3)
    _train(model, rng)
    model.frozen = True
    before = model.level_db().copy()
    model.update(np.full(N, -20.0))
    assert np.array_equal(model.level_db(), before)


def test_save_load_roundtrip(tmp_path) -> None:
    model = _model()
    rng = np.random.default_rng(4)
    _train(model, rng)
    path = tmp_path / "baseline.npz"
    model.save(path)
    loaded = BaselineModel.load(path, BaselineConfig(), CENTER, FS, N)
    assert loaded.frames_seen == model.frames_seen
    assert np.allclose(loaded.level_db(), model.level_db())


def test_load_rejects_mismatched_tuning(tmp_path) -> None:
    model = _model()
    rng = np.random.default_rng(5)
    _train(model, rng)
    path = tmp_path / "baseline.npz"
    model.save(path)
    with pytest.raises(BaselineMismatchError):
        BaselineModel.load(path, BaselineConfig(), CENTER + 1e6, FS, N)


def test_bank_isolates_models_per_center() -> None:
    bank = BaselineBank(BaselineConfig(), N)
    rng = np.random.default_rng(6)
    m24 = bank.get(2.4e9, FS)
    m58 = bank.get(5.8e9, FS)
    assert m24 is not m58
    assert bank.get(2.4e9, FS) is m24  # stable identity per key
    m24.update(rng.normal(-80.0, 1.0, N))
    assert m58.frames_seen == 0


def test_bank_save_load_roundtrip(tmp_path) -> None:
    bank = BaselineBank(BaselineConfig(), N)
    rng = np.random.default_rng(7)
    for center in (2.4e9, 5.8e9):
        model = bank.get(center, FS)
        for _ in range(10):
            model.update(rng.normal(-80.0, 1.0, N))
    env = tmp_path / "envs" / "home"
    bank.save(env, environment="home")

    loaded = BaselineBank.load(env, BaselineConfig(), N)
    assert set(loaded.models) == set(bank.models)
    for key, model in bank.models.items():
        assert np.allclose(loaded.models[key].level_db(), model.level_db())


def test_frozen_bank_freezes_new_models(tmp_path) -> None:
    bank = BaselineBank(BaselineConfig(), N)
    rng = np.random.default_rng(8)
    model = bank.get(2.4e9, FS)
    model.update(rng.normal(-80.0, 1.0, N))
    bank.freeze_all()
    # A step visited for the first time AFTER freezing must not adapt.
    late = bank.get(5.8e9, FS)
    assert late.frozen


def test_bank_load_rejects_wrong_fft_size(tmp_path) -> None:
    bank = BaselineBank(BaselineConfig(), N)
    rng = np.random.default_rng(9)
    model = bank.get(2.4e9, FS)
    for _ in range(6):
        model.update(rng.normal(-80.0, 1.0, N))
    env = tmp_path / "env"
    bank.save(env)
    with pytest.raises(BaselineMismatchError):
        BaselineBank.load(env, BaselineConfig(), N * 2)
