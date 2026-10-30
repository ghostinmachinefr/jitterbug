"""The streaming network emulator (used inside live Pipecat pipelines)."""

import numpy as np
import pytest

from jitterbug.network import NetworkProfile, StreamImpairer, transmit


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(seconds * 16_000))).astype(np.float32)


def feed(audio: np.ndarray, imp: StreamImpairer, chunk: int) -> np.ndarray:
    return np.concatenate([imp.process(audio[i : i + chunk]) for i in range(0, len(audio), chunk)])


def test_clean_stream_is_untouched():
    audio = noise(1)
    imp = StreamImpairer(NetworkProfile())
    assert np.array_equal(feed(audio, imp, 160), audio)
    assert imp.stats.packets_sent == 50 and imp.stats.effective_loss_pct == 0


@pytest.mark.parametrize("chunk", [1, 160, 333, 320, 16_000])
def test_result_does_not_depend_on_chunk_size(chunk):
    audio = noise(2)
    profile = NetworkProfile("bad", loss_pct=15, burst_len=3, jitter_ms=40)
    whole = StreamImpairer(profile, seed=5).process(audio)
    chunked = feed(audio, StreamImpairer(profile, seed=5), chunk)
    assert np.array_equal(whole, chunked)


def test_chunk_size_invariance_for_phone_line_codec():
    audio = noise(1)
    profile = NetworkProfile("pstn", codec="narrowband", loss_pct=5)
    a = StreamImpairer(profile, seed=1).process(audio)
    b = feed(audio, StreamImpairer(profile, seed=1), 160)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("loss, burst", [(5, 1), (10, 3), (20, 5)])
def test_stream_and_batch_both_match_theory(loss, burst):
    """Effective loss = lost packets + packets that arrived but too late.
    With exponential jitter of mean J and a buffer of B ms, a packet is late
    with probability exp(-B/J)."""
    audio = noise(400, seed=2)  # 20,000 packets
    profile = NetworkProfile("x", loss_pct=loss, burst_len=burst, jitter_ms=30)
    expected = loss + (100 - loss) * np.exp(-profile.buffer_ms / profile.jitter_ms)
    imp = StreamImpairer(profile, seed=3)
    feed(audio, imp, 320)
    _, batch = transmit(audio, profile, seed=3)
    assert imp.stats.packets_sent == batch.packets_sent == 20_000
    # bursty loss is "lumpy", so allow a couple of points either way
    assert imp.stats.effective_loss_pct == pytest.approx(expected, abs=2.0)
    assert batch.effective_loss_pct == pytest.approx(expected, abs=2.0)


def test_lost_packets_are_concealed_then_silenced():
    audio = np.ones(16_000, dtype=np.float32)
    imp = StreamImpairer(NetworkProfile("bursty", loss_pct=50, burst_len=10), seed=0)
    out = imp.process(audio)
    values = set(np.round(out[::320], 3))
    assert values <= {1.0, 0.5, 0.25, 0.125, 0.0}
    assert 0.0 in values and imp.stats.longest_gap > 3


def test_same_seed_same_stream():
    audio = noise(1)
    p = NetworkProfile("x", loss_pct=10, jitter_ms=50)
    assert np.array_equal(
        StreamImpairer(p, seed=9).process(audio), StreamImpairer(p, seed=9).process(audio)
    )
