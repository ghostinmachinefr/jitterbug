"""Lightweight stand-ins for real speech engines, so tests run in milliseconds.

FakeTTS turns each word into a short tone. FakeSTT checks each word's slot:
if packet loss wiped out most of its energy, the word is "misheard" (dropped).
That keeps the cause-and-effect of a real call (lost audio -> lost words)
without downloading any models.
"""

from __future__ import annotations

import itertools

import numpy as np

WORD_SAMPLES = 3200  # 200 ms per word at 16 kHz


class FakeSpeech:
    def __init__(self) -> None:
        self._registry: dict[int, list[str]] = {}
        self._counter = itertools.count(1)

    def tts(self, voice: str = "fake") -> FakeTTS:
        return FakeTTS(self, voice)

    def stt(self) -> FakeSTT:
        return FakeSTT(self)


class FakeTTS:
    def __init__(self, speech: FakeSpeech, voice: str) -> None:
        self.speech = speech
        self.name = f"fake:{voice}"
        self.last_ms = 5.0

    def synthesize(self, text: str) -> np.ndarray:
        words = text.split()
        # a unique tail length lets FakeSTT look the utterance up again
        tail = 16 * next(self.speech._counter)
        n = len(words) * WORD_SAMPLES + tail
        t = np.arange(n) / 16_000
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        audio[len(words) * WORD_SAMPLES :] = 0
        self.speech._registry[n] = words
        return audio


class FakeSTT:
    name = "fake-stt"

    def __init__(self, speech: FakeSpeech) -> None:
        self.speech = speech

    def transcribe(self, audio: np.ndarray) -> str:
        pieces = self._find_pieces(len(audio))
        expected_rms = 0.5 / np.sqrt(2)
        kept, offset = [], 0
        for n in pieces:
            words = self.speech._registry[n]
            for i, w in enumerate(words):
                seg = audio[offset + i * WORD_SAMPLES : offset + (i + 1) * WORD_SAMPLES]
                if np.sqrt(np.mean(seg**2)) >= 0.6 * expected_rms:
                    kept.append(w)
            offset += n
        return " ".join(kept)

    def _find_pieces(self, total: int) -> list[int]:
        """The audio may be several clips joined (agents speak sentence by
        sentence). Clips made one after another are adjacent in the registry,
        so look for a run of them whose lengths add up."""
        lengths = list(self.speech._registry)
        for start in range(len(lengths)):
            acc = 0
            for end in range(start, len(lengths)):
                acc += lengths[end]
                if acc == total:
                    return lengths[start : end + 1]
                if acc > total:
                    break
        return []
