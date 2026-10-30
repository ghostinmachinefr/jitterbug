"""Pipecat integration.

Pipecat (https://github.com/pipecat-ai/pipecat) is an open-source Python
framework for voice agents. A Pipecat agent is a *pipeline*: a chain of
processors that pass *frames* (chunks of audio, text, control signals) along,
for example  transport -> speech-to-text -> LLM -> text-to-speech -> transport.

This module gives two things:

1. `NetworkImpairmentProcessor` - a drop-in processor that damages audio the
   way a bad network would. Put it right after `transport.input()` in any
   Pipecat app to hear (and measure) how your bot copes with packet loss.

2. `PipecatAgent` - lets Jitterbug's call runner test a real Pipecat pipeline.
   It runs one pipeline per call in a background thread, feeds it the
   caller's (already network-damaged) audio as 20 ms frames, and watches every
   frame with a Pipecat *observer* to collect the transcript, the reply text,
   the reply audio and timings.

Plus the parts for a ready-made example bot (`booking_bot`) that needs no API
keys: Jitterbug's local Whisper and macOS voices wrapped as Pipecat services,
and the booking brain wrapped as an LLM-like processor.

Needs the optional extra:  uv sync --extra pipecat
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field

import numpy as np

try:
    from pipecat.frames.frames import (
        EndFrame,
        Frame,
        InputAudioRawFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        OutputAudioRawFrame,
        TranscriptionFrame,
        TTSAudioRawFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.settings import STTSettings, TTSSettings
    from pipecat.services.stt_service import SegmentedSTTService
    from pipecat.services.tts_service import TTSService
    from pipecat.utils.time import time_now_iso8601
    from pipecat.workers.runner import WorkerRunner
except ImportError as exc:  # pragma: no cover - only without the extra
    raise ImportError(
        "The Pipecat integration needs Pipecat. Install it with: uv sync --extra pipecat"
    ) from exc

from jitterbug.agent import AgentTurn, BookingBrain, Brain
from jitterbug.audio import SAMPLE_RATE, resample
from jitterbug.network import FRAME_MS, NetworkProfile, StreamImpairer, TransmitStats
from jitterbug.speech import STT, TTS


# --------------------------------------------------------------------------
# Audio helpers: Pipecat carries audio as 16-bit PCM bytes.
# --------------------------------------------------------------------------
def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    # Same scale as pcm16_to_float, so audio that wasn't changed comes back
    # byte for byte.
    samples = np.round(np.asarray(audio, dtype=np.float32) * 32768.0)
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


# --------------------------------------------------------------------------
# 1. Drop-in network emulator for any Pipecat pipeline
# --------------------------------------------------------------------------
class NetworkImpairmentProcessor(FrameProcessor):
    """Damages audio frames the way a bad network would.

        pipeline = Pipeline([
            transport.input(),
            NetworkImpairmentProcessor(PRESETS["mobile_4g_weak"]),  # <- add this
            stt, user_aggregator, llm, tts,
            transport.output(), assistant_aggregator,
        ])

    By default it damages what the *user* says (input audio, the uplink). Set
    `impair_output=True` and place it before `transport.output()` to damage
    what the bot says (the downlink) as well. All other frames pass through
    untouched. Frames are changed in place, so their ids, timestamps and
    metadata survive.

    Delay is reported in `stats` but not added in real time: real input
    frames are system frames that Pipecat handles immediately, and holding
    them back would stall the pipeline. Late packets are still dropped, which
    is how jitter actually hurts the audio.
    """

    def __init__(
        self,
        profile: NetworkProfile,
        *,
        seed: int = 0,
        impair_input: bool = True,
        impair_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.profile = profile
        self.seed = seed
        self.impair_input = impair_input
        self.impair_output = impair_output
        self._streams: dict[str, StreamImpairer] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if self.impair_input and isinstance(frame, InputAudioRawFrame):
                self._impair(frame, "uplink")
            elif self.impair_output and isinstance(frame, OutputAudioRawFrame):
                self._impair(frame, "downlink")
        await self.push_frame(frame, direction)

    def _impair(self, frame: InputAudioRawFrame | OutputAudioRawFrame, stream: str) -> None:
        if frame.num_channels != 1 or not frame.audio:
            return  # the emulator models mono voice audio
        imp = self._streams.get(stream)
        if imp is None or imp.sample_rate != frame.sample_rate:
            seed = self.seed * 2 + (stream == "downlink")
            imp = StreamImpairer(self.profile, seed=seed, sample_rate=frame.sample_rate)
            self._streams[stream] = imp
        frame.audio = float_to_pcm16(imp.process(pcm16_to_float(frame.audio)))

    @property
    def uplink_stats(self) -> TransmitStats | None:
        imp = self._streams.get("uplink")
        return imp.stats if imp else None

    @property
    def downlink_stats(self) -> TransmitStats | None:
        imp = self._streams.get("downlink")
        return imp.stats if imp else None


# --------------------------------------------------------------------------
# 2. Jitterbug engines as Pipecat services (no API keys needed)
# --------------------------------------------------------------------------
class EngineSTTService(SegmentedSTTService):
    """Runs any Jitterbug speech-to-text engine (e.g. local Whisper) as a
    Pipecat STT service. Like Pipecat's own Whisper service it is
    *segmented*: it collects audio between the "user started / stopped
    speaking" signals and transcribes each segment.

    It always sends a transcription, even an empty one, so the bot can say
    "sorry, I didn't catch that" instead of waiting forever.
    """

    def __init__(self, engine: STT, **kwargs) -> None:
        # Typical worst-case time from end of speech to transcript, used by
        # Pipecat's turn-taking; local Whisper on a laptop is around a second.
        kwargs.setdefault("ttfs_p99_latency", 1.5)
        super().__init__(settings=STTSettings(model=engine.name, language=None), **kwargs)
        self._engine = engine

    @property
    def wants_wav_segments(self) -> bool:
        return False  # we read raw 16-bit PCM, not a WAV file

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        samples = resample(pcm16_to_float(audio), self.sample_rate, SAMPLE_RATE)
        # Models are slow and synchronous: run them off the event loop.
        text = await asyncio.to_thread(self._engine.transcribe, samples)
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601())


class EngineTTSService(TTSService):
    """Runs any Jitterbug text-to-speech engine (e.g. macOS voices) as a
    Pipecat TTS service. Pipecat's TTS base class splits LLM text into
    sentences and calls `run_tts` once per sentence, which is why the caller
    starts hearing the reply after the first sentence."""

    def __init__(self, engine: TTS, **kwargs) -> None:
        # Pipecat's default: a reply's audio is "over" after 3 s without new
        # audio. It must be longer than one sentence takes to synthesise, or
        # replies get cut after the first sentence.
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=engine.name, language=None),
            **kwargs,
        )
        self._engine = engine

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        started = time.perf_counter()
        audio = await asyncio.to_thread(self._engine.synthesize, text)
        # Cached clips come back instantly. Wait out the time the clip really
        # took to make (engines report it as `last_ms`), so latency stays
        # honest and matches the built-in agent's numbers.
        real_ms = getattr(self._engine, "last_ms", None)
        if real_ms:
            await asyncio.sleep(max(0.0, real_ms / 1000 - (time.perf_counter() - started)))
        audio = resample(audio, SAMPLE_RATE, self.sample_rate)
        yield TTSAudioRawFrame(
            audio=float_to_pcm16(audio),
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )


class BrainProcessor(FrameProcessor):
    """Puts a Jitterbug brain where an LLM would sit in a Pipecat pipeline:
    a final transcription comes in, a reply goes out as LLM text frames,
    wrapped in the same start/end frames an LLM service sends."""

    def __init__(self, brain: Brain, **kwargs) -> None:
        super().__init__(**kwargs)
        self.brain = brain

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            reply = self.brain.reply(frame.text)
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(reply))
            await self.push_frame(LLMFullResponseEndFrame())


def booking_bot(stt: STT, tts: TTS, brain: BookingBrain) -> list[FrameProcessor]:
    """The example Pipecat bot: speech-to-text -> booking brain -> text-to-speech."""
    return [EngineSTTService(stt), BrainProcessor(brain), EngineTTSService(tts)]


# --------------------------------------------------------------------------
# 3. Adapter: test a Pipecat pipeline with Jitterbug's call runner
# --------------------------------------------------------------------------
@dataclass
class _Turn:
    """What the observer saw during one turn of the call."""

    stopped_at: float = 0.0  # when the caller finished speaking
    transcript: str | None = None
    transcript_at: float | None = None
    reply_started_at: float | None = None
    first_audio_at: float | None = None
    reply_text: list[str] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class _TurnObserver(BaseObserver):
    """Watches every frame pushed between processors (an observer sees them
    even when a processor consumes them, so this works with any bot)."""

    def __init__(self) -> None:
        super().__init__()
        self.turn: _Turn | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        turn, frame = self.turn, data.frame
        if turn is None:
            return
        # The turn is over when the TTS service passes on "LLM response
        # ended": Pipecat's TTS holds that frame back until all of the reply's
        # audio has gone out, so it marks the true end of the reply.
        if isinstance(frame, LLMFullResponseEndFrame) and isinstance(data.source, TTSService):
            if turn.reply_started_at is not None:
                turn.done.set()
            return
        if not data.first_push:  # count each frame once
            return
        now = time.perf_counter()
        if isinstance(frame, TranscriptionFrame) and turn.transcript is None:
            turn.transcript, turn.transcript_at = frame.text, now
        elif isinstance(frame, LLMFullResponseStartFrame) and turn.reply_started_at is None:
            turn.reply_started_at = now
        elif isinstance(frame, LLMTextFrame):
            turn.reply_text.append(frame.text)
        elif isinstance(frame, TTSAudioRawFrame):
            if turn.first_audio_at is None:
                turn.first_audio_at = now
            turn.audio.append(resample(pcm16_to_float(frame.audio), frame.sample_rate, SAMPLE_RATE))


class _NoState:
    """Used when a bot doesn't expose its state: nothing to check yet."""

    slots: dict[str, str] = {}
    done = False


class PipecatAgent:
    """Lets Jitterbug's `CallRunner` call a Pipecat bot.

    `build` returns the bot's processors (without a transport: this adapter
    plays the transport's role). It is called once per call, so each call
    starts fresh, but turns within a call share one running pipeline, so a
    bot keeps its conversation context exactly as it would on a real call.

    The adapter also plays the voice activity detector: it sends "user
    started speaking", the caller's audio in 20 ms frames, then "user stopped
    speaking". A scripted caller knows exactly when it stops talking, which
    keeps runs repeatable.

    `brain` is the bot's state (slots and whether it is done), used to check
    task success. Bots that don't expose state can leave it out; they get
    WER, quality and latency, and task checks come with the LLM judge later.
    """

    def __init__(
        self,
        build: Callable[[], list[FrameProcessor]],
        brain: Brain | None = None,
        stt: STT | None = None,
        greeting: str | None = None,
        turn_timeout_s: float = 60.0,
    ) -> None:
        self.build = build
        self.brain = brain or _NoState()
        self.stt = stt  # only used to warm the model up before timing starts
        self.greeting = greeting if greeting is not None else getattr(brain, "greeting", None)
        self.turn_timeout_s = turn_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._worker: PipelineWorker | None = None
        self._observer = _TurnObserver()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        ready = threading.Event()
        errors: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            try:
                loop.run_until_complete(self._run_pipeline(ready))
            except BaseException as exc:  # surfaced to the caller below
                errors.append(exc)
                ready.set()
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name="pipecat-agent", daemon=True)
        self._thread.start()
        if not ready.wait(timeout=30) or errors:
            raise RuntimeError(f"Pipecat pipeline failed to start: {errors[:1]}")

    async def _run_pipeline(self, ready: threading.Event) -> None:
        params = PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE)
        worker = PipelineWorker(
            Pipeline(self.build()),
            params=params,
            observers=[self._observer],
            cancel_on_idle_timeout=False,
        )
        self._worker = worker

        @worker.event_handler("on_pipeline_started")
        async def _started(worker, frame):
            ready.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()

    def close(self) -> None:
        """End the call: send Pipecat's EndFrame and wait for the pipeline to stop."""
        if self._thread is None or self._loop is None or self._worker is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._worker.queue_frame(EndFrame()), self._loop)
            fut.result(timeout=10)
        except Exception:
            pass
        self._thread.join(timeout=30)
        self._thread = None

    # -- the CascadeAgent interface used by CallRunner -------------------------
    def greet(self) -> AgentTurn:
        self.start()
        if not self.greeting:
            return AgentTurn("", "", np.zeros(0, np.float32), {})
        text = self.greeting() if callable(self.greeting) else str(self.greeting)
        return self._run(self._speak(text))

    def handle(self, audio: np.ndarray) -> AgentTurn:
        self.start()
        return self._run(self._listen(audio))

    def _run(self, coro) -> AgentTurn:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=self.turn_timeout_s + 5
        )

    # -- inside the pipeline's event loop -------------------------------------
    async def _listen(self, audio: np.ndarray) -> AgentTurn:
        turn = self._observer.turn = _Turn()
        worker = self._worker
        await worker.queue_frame(VADUserStartedSpeakingFrame())
        step = SAMPLE_RATE * FRAME_MS // 1000
        for i in range(0, len(audio), step):
            chunk = float_to_pcm16(audio[i : i + step])
            await worker.queue_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=SAMPLE_RATE, num_channels=1)
            )
        turn.stopped_at = time.perf_counter()
        await worker.queue_frame(VADUserStoppedSpeakingFrame())
        return await self._finish(turn)

    async def _speak(self, text: str) -> AgentTurn:
        """Make the bot say a line (the greeting), sent the way an LLM reply is."""
        turn = self._observer.turn = _Turn()
        turn.stopped_at = turn.transcript_at = time.perf_counter()
        turn.transcript = ""
        for frame in (LLMFullResponseStartFrame(), LLMTextFrame(text), LLMFullResponseEndFrame()):
            await self._worker.queue_frame(frame)
        return await self._finish(turn)

    async def _finish(self, turn: _Turn) -> AgentTurn:
        try:
            await asyncio.wait_for(turn.done.wait(), timeout=self.turn_timeout_s)
        except TimeoutError:
            pass  # report what arrived; a silent bot shows up as a failed turn
        self._observer.turn = None
        audio = np.concatenate(turn.audio) if turn.audio else np.zeros(0, np.float32)
        return AgentTurn(
            heard=turn.transcript or "",
            reply_text="".join(turn.reply_text).strip(),
            reply_audio=audio,
            timings_ms=_timings(turn),
        )


def _timings(t: _Turn) -> dict[str, float]:
    """Split the time from 'caller stopped' to 'first reply audio' into
    speech-to-text, brain (LLM) and text-to-speech."""

    def ms(a: float | None, b: float | None) -> float:
        return round(1000 * (b - a), 1) if a is not None and b is not None else 0.0

    reply_start = t.reply_started_at or t.transcript_at
    return {
        "stt_ms": ms(t.stopped_at, t.transcript_at),
        "brain_ms": ms(t.transcript_at, reply_start),
        "tts_first_audio_ms": ms(reply_start, t.first_audio_at),
    }


def booking_agent_factory(stt: STT, tts: TTS) -> Callable[[], PipecatAgent]:
    """CallRunner agent factory for the example Pipecat booking bot."""

    def make() -> PipecatAgent:
        brain = BookingBrain()
        return PipecatAgent(lambda: booking_bot(stt, tts, brain), brain=brain, stt=stt)

    return make
