"""The voice agent under test.

A voice agent is a loop: speech-to-text -> "brain" -> text-to-speech.
`CascadeAgent` wires any STT, brain and TTS together and times each step,
which is what lets Jitterbug say *where* a slow or wrong reply came from.

The built-in brain is a small restaurant-booking assistant with clear rules.
Rules (rather than an LLM) keep the test agent fast, free and repeatable: if
a call fails, it failed because of what the agent *heard*, which is exactly
what network testing needs to isolate. An LLM brain can be plugged in through
the same `Brain` interface.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from jitterbug.scoring.wer import normalize
from jitterbug.speech import STT, TTS


class Brain(Protocol):
    slots: dict[str, str]

    def greeting(self) -> str: ...

    def reply(self, heard: str) -> str: ...

    @property
    def done(self) -> bool: ...


_WORD_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_MINUTES = {"fifteen": 15, "thirty": 30, "forty five": 45, "half": 30}


class BookingBrain:
    """Collects party size, time and name for a table booking."""

    REQUIRED = ("party_size", "time", "name")
    QUESTIONS = {
        "party_size": "How many people is the table for?",
        "time": "What time would you like?",
        "name": "And what name should I put the booking under?",
    }

    def __init__(self) -> None:
        self.slots: dict[str, str] = {}

    def greeting(self) -> str:
        return "Hi, thanks for calling Spice Garden. How can I help you today?"

    @property
    def done(self) -> bool:
        return all(k in self.slots for k in self.REQUIRED)

    def reply(self, heard: str) -> str:
        text = normalize(heard)
        if not text:
            return "Sorry, I didn't catch that. Could you say it again?"
        for slot, value in self._extract(text).items():
            self.slots.setdefault(slot, value)
        if self.done:
            return (
                f"Great, a table for {self.slots['party_size']} at {self.slots['time']} "
                f"under {self.slots['name'].title()}. See you then!"
            )
        missing = next(k for k in self.REQUIRED if k not in self.slots)
        return self.QUESTIONS[missing]

    @staticmethod
    def _extract(text: str) -> dict[str, str]:
        found: dict[str, str] = {}
        nums = "|".join(_WORD_NUM)
        m = re.search(rf"\b(?:for|of)\s+({nums})\b(?:\s+(?:people|persons|guests|of us))?", text)
        if m and not re.match(rf"(?:{nums})\s*(?:am|pm|o'?clock)", text[m.start(1) :]):
            found["party_size"] = str(_WORD_NUM[m.group(1)])
        m = re.search(rf"\bat\s+({nums})(?:\s+(fifteen|thirty|forty five))?\s*(am|pm)?\b", text)
        if m:
            hour = _WORD_NUM[m.group(1)]
            minute = _MINUTES.get(m.group(2) or "", 0)
            found["time"] = f"{hour}:{minute:02d}" + (f" {m.group(3)}" if m.group(3) else "")
        m = re.search(r"\b(?:name is|under the name|under|it's|it is)\s+([a-z]+)\b", text)
        if m and m.group(1) not in {"a", "the", "for", "at"}:
            found["name"] = m.group(1)
        return found


@dataclass
class AgentTurn:
    heard: str  # what speech-to-text produced
    reply_text: str
    reply_audio: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)


class CascadeAgent:
    """STT -> brain -> TTS, with each step timed."""

    def __init__(self, stt: STT, tts: TTS, brain: Brain | None = None) -> None:
        self.stt = stt
        self.tts = tts
        self.brain = brain or BookingBrain()

    def greet(self) -> AgentTurn:
        text = self.brain.greeting()
        audio, first_ms = self._speak(text)
        return AgentTurn("", text, audio, {"tts_first_audio_ms": first_ms})

    def handle(self, audio: np.ndarray) -> AgentTurn:
        t0 = time.perf_counter()
        heard = self.stt.transcribe(audio)
        t1 = time.perf_counter()
        reply = self.brain.reply(heard)
        t2 = time.perf_counter()
        reply_audio, first_ms = self._speak(reply)
        return AgentTurn(
            heard,
            reply,
            reply_audio,
            {"stt_ms": _ms(t0, t1), "brain_ms": _ms(t1, t2), "tts_first_audio_ms": first_ms},
        )

    def _speak(self, text: str) -> tuple[np.ndarray, float]:
        """Synthesise sentence by sentence, like streaming voice frameworks
        (Pipecat sends each finished sentence to TTS). The caller starts
        hearing the reply once the first sentence is ready, so only that
        sentence counts towards response latency."""
        parts, first_ms = [], 0.0
        for i, sentence in enumerate(split_sentences(text)):
            t0 = time.perf_counter()
            parts.append(self.tts.synthesize(sentence))
            if i == 0:
                first_ms = round(getattr(self.tts, "last_ms", None) or _ms(t0), 1)
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), first_ms


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _ms(start: float, end: float | None = None) -> float:
    return round(1000 * ((end if end is not None else time.perf_counter()) - start), 1)
