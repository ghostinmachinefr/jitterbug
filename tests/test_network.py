import numpy as np
import pytest

from jitterbug.network import (
    PRESETS,
    NetworkProfile,
    conceal,
    gilbert_elliott_mask,
    loss_sweep,
    mulaw_decode,
    mulaw_encode,
    transmit,
)


def tone(seconds: float = 2.0, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_clean_profile_is_lossless():
    audio = tone()
    out, stats = transmit(audio, NetworkProfile())
    assert np.allclose(out, audio)
    assert stats.packets_lost == stats.packets_late == 0
    assert stats.delay_ms == 0


@pytest.mark.parametrize("loss, burst", [(5, 1), (10, 3), (20, 5)])
def test_gilbert_elliott_hits_target_loss_rate(loss, burst):
    rng = np.random.default_rng(1)
    mask = gilbert_elliott_mask(200_000, loss, burst, rng)
    assert mask.mean() * 100 == pytest.approx(loss, abs=0.6)


def test_bursty_loss_has_longer_runs_than_random_loss():
    def mean_run(mask):
        runs, run = [], 0
        for m in mask:
            if m:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        return np.mean(runs)

    rng = np.random.default_rng(2)
    random_loss = gilbert_elliott_mask(100_000, 10, 1, rng)
    bursty_loss = gilbert_elliott_mask(100_000, 10, 4, rng)
    assert mean_run(random_loss) == pytest.approx(1.0, abs=0.2)
    assert mean_run(bursty_loss) == pytest.approx(4.0, abs=0.4)


def test_same_seed_same_result():
    audio = tone()
    p = PRESETS["mobile_4g_weak"]
    a, sa = transmit(audio, p, seed=7)
    b, sb = transmit(audio, p, seed=7)
    assert np.array_equal(a, b) and sa == sb


def test_jitter_causes_late_packets_and_buffer_delay():
    p = NetworkProfile("jittery", jitter_ms=80, jitter_buffer_ms=60, latency_ms=50)
    _, stats = transmit(tone(5), p, seed=3)
    assert stats.packets_late > 0
    assert stats.packets_lost == 0
    assert stats.delay_ms == 110


def test_more_loss_means_more_damage():
    audio = tone(3)
    errors = []
    for profile in loss_sweep((0, 5, 20)):
        out, _ = transmit(audio, profile, seed=0)
        errors.append(float(np.mean((out - audio) ** 2)))
    assert errors[0] < errors[1] < errors[2]


def test_mulaw_roundtrip_is_close():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    y = mulaw_decode(mulaw_encode(x))
    # mu-law uses coarse steps for loud samples and fine steps for quiet ones
    assert np.max(np.abs(x - y)) < 0.03
    quiet = np.abs(x) < 0.05
    assert np.max(np.abs(x[quiet] - y[quiet])) < 0.002


def test_narrowband_keeps_length_and_removes_high_frequencies():
    sr = 16_000
    t = np.arange(sr) / sr
    high = (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)  # above 4 kHz
    out, _ = transmit(high, NetworkProfile("pstn", codec="narrowband"))
    assert len(out) == len(high)
    assert np.sqrt(np.mean(out**2)) < 0.05


def test_concealment_fades_then_goes_silent():
    frames = np.ones((6, 4), dtype=np.float32)
    missing = np.array([False, True, True, True, True, False])
    out = conceal(frames, missing)
    assert out[1, 0] == 0.5 and out[2, 0] == 0.25 and out[3, 0] == 0.125
    assert out[4, 0] == 0 and out[5, 0] == 1


def test_invalid_profiles_are_rejected():
    with pytest.raises(ValueError):
        NetworkProfile(loss_pct=120)
    with pytest.raises(ValueError):
        NetworkProfile(codec="opus")


def test_adaptive_buffer_keeps_late_loss_small():
    p = NetworkProfile("wifi", jitter_ms=30, latency_ms=40)
    assert p.buffer_ms == 110
    _, stats = transmit(tone(30), p, seed=4)
    assert stats.effective_loss_pct < 5
    assert stats.delay_ms == 150
