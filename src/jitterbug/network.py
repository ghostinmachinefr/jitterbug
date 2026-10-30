"""Emulated network faults for real-time audio.

Audio travels over a call as small packets (here: 20 ms frames). This module
sends each frame through a simulated network and rebuilds what the other side
would hear:

* **Packet loss** uses a Gilbert-Elliott model, so losses can come in bursts
  (as they do on Wi-Fi and mobile links) instead of being spread evenly.
* **Jitter** gives every packet a random extra delay. A fixed jitter buffer
  waits a set time; packets that arrive after that are thrown away ("late
  loss"), exactly like a real VoIP receiver.
* **Packet loss concealment** fills each gap by repeating the last good frame
  at falling volume, then silence.
* **Codec**: "narrowband" squeezes audio through an 8 kHz, 8-bit mu-law
  channel, the same as a G.711 phone line.

Everything is seeded, so the same profile and seed always give the same result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from jitterbug.audio import SAMPLE_RATE, resample

FRAME_MS = 20


@dataclass(frozen=True)
class NetworkProfile:
    """One set of network conditions to test a call under."""

    name: str = "clean"
    loss_pct: float = 0.0  # average share of packets lost, 0-100
    burst_len: float = 1.0  # average lost packets in a row (1 = random loss)
    jitter_ms: float = 0.0  # typical extra delay per packet
    latency_ms: float = 0.0  # fixed one-way delay
    # How long the receiver waits for late packets. None = adaptive, like
    # WebRTC's NetEQ: sized to the jitter (3x jitter + 20 ms).
    jitter_buffer_ms: float | None = None
    codec: str = "wideband"  # "wideband" (16 kHz) or "narrowband" (8 kHz mu-law)

    def __post_init__(self) -> None:
        if not 0 <= self.loss_pct < 100:
            raise ValueError("loss_pct must be in [0, 100)")
        if self.burst_len < 1:
            raise ValueError("burst_len must be >= 1")
        if self.jitter_ms < 0 or self.latency_ms < 0 or (self.jitter_buffer_ms or 0) < 0:
            raise ValueError("delays must be >= 0")
        if self.codec not in ("wideband", "narrowband"):
            raise ValueError("codec must be 'wideband' or 'narrowband'")

    @property
    def buffer_ms(self) -> float:
        """Jitter buffer actually used (0 when there is no jitter to absorb)."""
        if self.jitter_ms == 0:
            return 0.0
        if self.jitter_buffer_ms is not None:
            return float(self.jitter_buffer_ms)
        return 3 * self.jitter_ms + 20

    @property
    def is_clean(self) -> bool:
        return (
            self.loss_pct == 0
            and self.jitter_ms == 0
            and self.latency_ms == 0
            and self.codec == "wideband"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TransmitStats:
    """What happened to the packets of one stream."""

    packets_sent: int
    packets_lost: int  # dropped by the network
    packets_late: int  # arrived after the jitter buffer gave up
    longest_gap: int  # longest run of missing packets
    delay_ms: float  # one-way delay the listener experiences

    @property
    def effective_loss_pct(self) -> float:
        if self.packets_sent == 0:
            return 0.0
        return 100.0 * (self.packets_lost + self.packets_late) / self.packets_sent

    def to_dict(self) -> dict:
        d = asdict(self)
        d["effective_loss_pct"] = round(self.effective_loss_pct, 2)
        return d


def gilbert_elliott_mask(
    n: int, loss_pct: float, burst_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a boolean array, True where a packet is lost.

    Two states: Good (packet arrives) and Bad (packet lost). With average burst
    length B and loss rate L, the chance of leaving Bad is r = 1/B and the
    chance of entering Bad is p = L*r / (1-L), which makes the long-run share
    of Bad packets equal to L.
    """
    if n <= 0:
        return np.zeros(0, dtype=bool)
    loss = loss_pct / 100.0
    if loss == 0:
        return np.zeros(n, dtype=bool)
    r = 1.0 / burst_len
    p = loss * r / (1.0 - loss)
    p = min(p, 1.0)
    lost = np.empty(n, dtype=bool)
    bad = rng.random() < loss  # start in the long-run state mix
    draws = rng.random(n)
    for i in range(n):
        lost[i] = bad
        bad = draws[i] >= r if bad else draws[i] < p
    return lost


def jitter_late_mask(
    n: int, jitter_ms: float, buffer_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """True where a packet arrives too late for the jitter buffer.

    Extra delay per packet is drawn from an exponential distribution with mean
    `jitter_ms` (most packets are a little late, a few are very late).
    """
    if n <= 0 or jitter_ms == 0:
        return np.zeros(max(n, 0), dtype=bool)
    extra = rng.exponential(jitter_ms, size=n)
    return extra > buffer_ms


def mulaw_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.round((y + 1) / 2 * mu).astype(np.uint8)


def mulaw_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = 2 * (q.astype(np.float32) / mu) - 1
    return (np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu).astype(np.float32)


def narrowband(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pass audio through an 8 kHz mu-law phone channel and back."""
    low = resample(audio, sr, 8000)
    low = mulaw_decode(mulaw_encode(low))
    out = resample(low, 8000, sr)
    return _fit_length(out, len(audio))


def conceal(frames: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill missing frames: repeat the last good frame at half volume each
    time, and go silent after 3 missing frames in a row."""
    out = frames.copy()
    last_good = np.zeros(frames.shape[1], dtype=frames.dtype)
    run = 0
    for i in range(len(frames)):
        if missing[i]:
            run += 1
            out[i] = last_good * (0.5**run) if run <= 3 else 0.0
        else:
            run = 0
            last_good = frames[i]
    return out


def transmit(
    audio: np.ndarray,
    profile: NetworkProfile,
    seed: int = 0,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, TransmitStats]:
    """Send mono float32 audio through the network and return what the
    listener receives, plus packet statistics."""
    audio = np.asarray(audio, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if profile.codec == "narrowband":
        audio = narrowband(audio, sr)

    frame_len = sr * FRAME_MS // 1000
    n_frames = -(-len(audio) // frame_len)  # ceiling division
    padded = np.zeros(n_frames * frame_len, dtype=np.float32)
    padded[: len(audio)] = audio
    frames = padded.reshape(n_frames, frame_len)

    lost = gilbert_elliott_mask(n_frames, profile.loss_pct, profile.burst_len, rng)
    late = jitter_late_mask(n_frames, profile.jitter_ms, profile.buffer_ms, rng) & ~lost
    missing = lost | late
    received = conceal(frames, missing).reshape(-1)[: len(audio)]

    # The listener hears audio after the fixed delay plus the buffer wait.
    delay = profile.latency_ms + profile.buffer_ms
    stats = TransmitStats(
        packets_sent=n_frames,
        packets_lost=int(lost.sum()),
        packets_late=int(late.sum()),
        longest_gap=_longest_run(missing),
        delay_ms=float(delay),
    )
    return received.astype(np.float32), stats


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


# Ready-made conditions for sweeps. Numbers are typical of real links.
PRESETS: dict[str, NetworkProfile] = {
    "clean": NetworkProfile("clean"),
    "good_wifi": NetworkProfile(
        "good_wifi", loss_pct=0.5, burst_len=1.5, jitter_ms=10, latency_ms=30
    ),
    "busy_wifi": NetworkProfile(
        "busy_wifi", loss_pct=3, burst_len=2.5, jitter_ms=30, latency_ms=60
    ),
    "mobile_4g_weak": NetworkProfile(
        "mobile_4g_weak", loss_pct=6, burst_len=3, jitter_ms=50, latency_ms=120
    ),
    "phone_line": NetworkProfile(
        "phone_line", loss_pct=1, burst_len=1.5, jitter_ms=15, latency_ms=80, codec="narrowband"
    ),
    "congested": NetworkProfile(
        "congested", loss_pct=12, burst_len=4, jitter_ms=80, latency_ms=200
    ),
}


def loss_sweep(
    losses: tuple[float, ...] = (0, 1, 2, 5, 10, 15, 20),
    burst_len: float = 2.0,
    jitter_ms: float = 20.0,
) -> list[NetworkProfile]:
    """Profiles that differ only in packet loss, for 'where does it break?' charts."""
    return [
        NetworkProfile(
            f"loss_{loss:g}pct",
            loss_pct=loss,
            burst_len=burst_len,
            jitter_ms=jitter_ms if loss else 0.0,
        )
        for loss in losses
    ]
