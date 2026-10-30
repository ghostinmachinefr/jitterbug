"""Pipecat integration: the drop-in network processor and the agent adapter.

Uses Pipecat's own test helper (`run_test`) for the processor, and the fake
speech engines for the adapter, so nothing is downloaded.
"""

import asyncio

import numpy as np
import pytest

pytest.importorskip("pipecat", reason="needs the pipecat extra")

from pipecat.frames.frames import (  # noqa: E402
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TextFrame,
)
from pipecat.tests.utils import run_test  # noqa: E402

from jitterbug.diagnosis import diagnose  # noqa: E402
from jitterbug.integrations.pipecat import (  # noqa: E402
    NetworkImpairmentProcessor,
    PipecatAgent,
    booking_agent_factory,
    float_to_pcm16,
    pcm16_to_float,
)
from jitterbug.network import NetworkProfile, loss_sweep  # noqa: E402
from jitterbug.runner import CallRunner, run_sweep, sweep_table  # noqa: E402


def audio_frames(n: int = 50, cls=InputAudioRawFrame):
    """n x 20 ms frames of a steady tone."""
    t = np.arange(320) / 16_000
    pcm = float_to_pcm16(0.5 * np.sin(2 * np.pi * 300 * t))
    return [cls(audio=pcm, sample_rate=16_000, num_channels=1) for _ in range(n)]


def run(processor, frames):
    down, _ = asyncio.run(run_test(processor, frames_to_send=frames))
    return down


def test_pcm_round_trip():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert np.max(np.abs(pcm16_to_float(float_to_pcm16(x)) - x)) < 1e-4
    pcm = np.arange(-32768, 32768, 7, dtype=np.int16).tobytes()
    assert float_to_pcm16(pcm16_to_float(pcm)) == pcm  # untouched audio is exact


def test_processor_damages_input_audio_and_passes_other_frames():
    frames = audio_frames(50)
    original = frames[0].audio
    text = TextFrame("hello")
    proc = NetworkImpairmentProcessor(NetworkProfile("bad", loss_pct=40, burst_len=5), seed=1)
    down = run(proc, [*frames, text])

    assert [type(f) for f in down] == [InputAudioRawFrame] * 50 + [TextFrame]
    assert down[-1] is text  # untouched
    assert all(len(f.audio) == len(original) for f in down[:-1])  # same size, in place
    damaged = sum(f.audio != original for f in down[:-1])
    stats = proc.uplink_stats
    assert stats.packets_sent == 50
    assert damaged == stats.packets_lost + stats.packets_late > 0


def test_clean_profile_changes_nothing():
    frames = audio_frames(10)
    original = [f.audio for f in frames]
    down = run(NetworkImpairmentProcessor(NetworkProfile()), frames)
    assert [f.audio for f in down] == original


def test_output_audio_is_only_damaged_when_asked():
    profile = NetworkProfile("bad", loss_pct=50, burst_len=5)
    frames = audio_frames(30, OutputAudioRawFrame)
    original = frames[0].audio
    down = run(NetworkImpairmentProcessor(profile, seed=2), frames)
    assert all(f.audio == original for f in down)

    proc = NetworkImpairmentProcessor(profile, seed=2, impair_input=False, impair_output=True)
    down = run(proc, audio_frames(30, OutputAudioRawFrame))
    assert any(f.audio != original for f in down)
    assert proc.downlink_stats.packets_lost > 0 and proc.uplink_stats is None


@pytest.fixture
def pipecat_runner(speech):
    stt = speech.stt()
    return CallRunner(
        booking_agent_factory(stt, speech.tts("agent")),
        caller_tts=speech.tts,
        caller_stt=stt,
    )


def test_pipecat_bot_completes_a_booking(pipecat_runner, booking):
    call = diagnose(pipecat_runner.run(booking, NetworkProfile()), None)
    assert call.verdict == "pass", call.diagnosis
    assert call.slots == {"party_size": "4", "time": "8:30 pm", "name": "kapoor"}
    first = call.turns[0]
    assert first.heard_by_agent == booking.turns[0]
    assert first.agent_reply == "What time would you like?"
    assert first.downlink_wer == 0  # the caller heard the reply audio
    assert set(first.timings_ms) == {"stt_ms", "brain_ms", "tts_first_audio_ms"}


def test_pipecat_bot_failures_are_blamed_on_the_network(pipecat_runner, booking):
    baseline = diagnose(pipecat_runner.run(booking, NetworkProfile()), None)
    bad = pipecat_runner.run(booking, NetworkProfile("awful", loss_pct=60, burst_len=8), seed=3)
    diagnose(bad, baseline)
    assert bad.verdict == "network" and "Uplink lost" in bad.diagnosis


def test_pipecat_sweep(pipecat_runner, booking):
    results = run_sweep(pipecat_runner, booking, loss_sweep((0, 40)), seeds=3)
    table = {row["profile"]: row for row in sweep_table(results)}
    assert table["clean"]["task_success_pct"] == 100
    assert table["loss_40pct"]["mean_wer"] > 0


def test_silence_gets_a_polite_retry(speech):
    agent = booking_agent_factory(speech.stt(), speech.tts("agent"))()
    try:
        agent.greet()
        turn = agent.handle(np.zeros(16_000, dtype=np.float32))
    finally:
        agent.close()
    assert turn.heard == ""
    assert "didn't catch" in turn.reply_text
    assert len(turn.reply_audio) > 0


def test_bot_without_state_still_runs(speech):
    """Any Pipecat bot can be tested; without exposed state there's nothing
    to check task success against, but transcript, reply and timing work."""
    from jitterbug.agent import BookingBrain
    from jitterbug.integrations.pipecat import booking_bot

    stt, tts = speech.stt(), speech.tts("agent")
    agent = PipecatAgent(lambda: booking_bot(stt, tts, BookingBrain()))
    try:
        agent.greet()
        turn = agent.handle(tts.synthesize("a table for two please"))
    finally:
        agent.close()
    assert agent.brain.slots == {} and not agent.brain.done
    assert turn.reply_text == "What time would you like?"


class SlowTTS:
    """A TTS that takes 0.4 s per sentence, like a real local voice."""

    def __init__(self, inner):
        self.inner, self.name, self.last_ms = inner, "slow", None

    def synthesize(self, text):
        import time

        time.sleep(0.4)
        return self.inner.synthesize(text)


def test_slow_tts_multi_sentence_reply_is_heard_in_full(speech, booking):
    """Regression: replies used to be cut after the first sentence when the
    TTS needed longer than Pipecat's idle timeout to make the next one."""
    stt = speech.stt()
    runner = CallRunner(
        booking_agent_factory(stt, SlowTTS(speech.tts("agent"))),
        caller_tts=speech.tts,
        caller_stt=stt,
    )
    call = diagnose(runner.run(booking, NetworkProfile()), None)
    last = call.turns[-1]
    assert last.agent_reply.endswith("See you then!")  # two sentences
    assert last.heard_by_caller.endswith("See you then!")
    assert last.downlink_wer == 0
