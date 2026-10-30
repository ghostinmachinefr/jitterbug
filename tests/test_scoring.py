import importlib.util

import numpy as np
import pytest

from jitterbug.network import NetworkProfile, transmit
from jitterbug.scoring.quality import DNSMOS
from jitterbug.scoring.wer import normalize, number_to_words, wer


@pytest.mark.parametrize(
    "n, words",
    [
        (0, "zero"),
        (4, "four"),
        (21, "twenty one"),
        (130, "one hundred thirty"),
        (2026, "two thousand twenty six"),
    ],
)
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_normalize_handles_times_and_punctuation():
    assert normalize("A table for 4 at 8:30 P.M., please!") == (
        "a table for four at eight thirty pm please"
    )
    assert normalize("8:00 pm") == "eight pm"


def test_wer_basics():
    assert wer("table for four", "Table for 4.") == 0.0
    assert wer("table for four", "table for") == pytest.approx(1 / 3)
    assert wer("table for four", "") == 1.0
    assert wer("", "") == 0.0


dnsmos = DNSMOS()
needs_model = pytest.mark.skipif(
    not dnsmos.available or importlib.util.find_spec("onnxruntime") is None,
    reason="DNSMOS model or onnxruntime not installed",
)


@needs_model
def test_dnsmos_scores_damaged_audio_lower():
    rng = np.random.default_rng(0)
    sr = 16_000
    t = np.arange(4 * sr) / sr
    # speech-like signal: a few harmonics with a syllable-rate envelope
    voice = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([180, 360, 540, 720], 1))
    voice = (0.3 * voice * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    damaged, _ = transmit(voice, NetworkProfile("bad", loss_pct=25, burst_len=4), seed=1)
    noisy = damaged + 0.05 * rng.standard_normal(len(damaged)).astype(np.float32)
    clean_score = dnsmos.score(voice)
    bad_score = dnsmos.score(noisy)
    assert 1.0 <= bad_score.ovrl < clean_score.ovrl <= 5.0


def test_dnsmos_silence_is_worst_score():
    assert DNSMOS(model_path="/nonexistent").score(np.zeros(16_000)).ovrl == 1.0
