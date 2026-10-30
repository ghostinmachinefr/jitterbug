"""Text-to-speech and speech-to-text engines.

Both are small interfaces so any provider can be plugged in (Deepgram,
ElevenLabs, a Pipecat pipeline...). Built in:

* `MacSayTTS`  - the voices that ship with macOS (free, offline).
* `WhisperSTT` - OpenAI Whisper running locally through faster-whisper.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from jitterbug.audio import SAMPLE_RATE, load, save


class TTS(Protocol):
    name: str

    def synthesize(self, text: str) -> np.ndarray: ...


class STT(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...


class MacSayTTS:
    """macOS `say` command. Results are cached on disk, so sweeps that repeat
    the same lines don't re-synthesise them."""

    def __init__(
        self, voice: str = "Samantha", rate_wpm: int = 180, cache_dir: str | Path | None = None
    ) -> None:
        if shutil.which("say") is None:
            raise RuntimeError("MacSayTTS needs macOS (the `say` command)")
        self.voice = voice
        self.rate_wpm = rate_wpm
        self.name = f"macos-say:{voice}"
        self.last_ms: float | None = None
        self.cache_dir = Path(cache_dir or Path(tempfile.gettempdir()) / "jitterbug-tts")

    def synthesize(self, text: str) -> np.ndarray:
        """Return the audio. `self.last_ms` holds how long synthesis took; on a
        cache hit it is the time measured when the clip was first made, so
        latency numbers stay honest."""
        key = hashlib.sha1(f"{self.voice}|{self.rate_wpm}|{text}".encode()).hexdigest()[:20]
        path = self.cache_dir / f"{key}.wav"
        timing = path.with_suffix(".ms")
        if not path.exists() or not timing.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.wav")
            t0 = time.perf_counter()
            subprocess.run(
                [
                    "say",
                    "-v",
                    self.voice,
                    "-r",
                    str(self.rate_wpm),
                    "-o",
                    str(tmp),
                    f"--data-format=LEI16@{SAMPLE_RATE}",
                    text,
                ],
                check=True,
                capture_output=True,
            )
            elapsed = 1000 * (time.perf_counter() - t0)
            tmp.rename(path)
            timing.write_text(f"{elapsed:.1f}")
        self.last_ms = float(timing.read_text())
        return load(path)


class WhisperSTT:
    """Local Whisper via faster-whisper. The model downloads on first use
    (base.en is about 140 MB). One model instance is shared across threads,
    guarded by a lock."""

    _models: dict[tuple[str, str], object] = {}
    _load_lock = threading.Lock()

    def __init__(self, model: str = "base.en", compute_type: str = "int8") -> None:
        self.model_name = model
        self.compute_type = compute_type
        self.name = f"whisper:{model}"
        self._run_lock = threading.Lock()

    def _model(self):
        key = (self.model_name, self.compute_type)
        with self._load_lock:
            if key not in self._models:
                from faster_whisper import WhisperModel

                self._models[key] = WhisperModel(
                    self.model_name, device="cpu", compute_type=self.compute_type
                )
            return self._models[key]

    def warmup(self) -> None:
        noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.01
        self.transcribe(noise)

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if not np.any(audio):
            return ""
        with self._run_lock:
            segments, _ = self._model().transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()


def save_clip(path: str | Path, audio: np.ndarray) -> str:
    save(path, audio)
    return str(path)
