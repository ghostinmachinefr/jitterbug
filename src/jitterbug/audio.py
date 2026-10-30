"""Small audio helpers. All audio inside Jitterbug is mono float32 at 16 kHz."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(audio, dtype=np.float32)
    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def load(path: str | Path) -> np.ndarray:
    """Read any WAV/AIFF/FLAC file as mono 16 kHz float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), sr, SAMPLE_RATE)


def save(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")


def duration_ms(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    return 1000.0 * len(audio) / sr


def silence(ms: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)
