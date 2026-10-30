"""Run scripted test calls through emulated networks and score them.

One call, turn by turn:

    caller TTS --uplink network--> agent STT -> brain -> agent TTS
                                                            |
    caller STT <--downlink network-----------------------------

For every turn we record what was said, what the agent heard (and the WER),
what the agent replied, what the caller heard back, packet statistics in both
directions, DNSMOS quality of the audio each side received, and the response
latency the caller experienced.
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jitterbug.agent import CascadeAgent
from jitterbug.audio import duration_ms, save
from jitterbug.network import NetworkProfile, transmit
from jitterbug.scenarios import Scenario
from jitterbug.scoring.quality import DNSMOS
from jitterbug.scoring.wer import wer
from jitterbug.speech import STT, TTS

# Time a real agent waits after the caller stops talking before it decides the
# turn is over (voice activity detection). Counted in response latency.
ENDPOINTING_MS = 300.0
# Replies slower than this feel broken to callers. Voice-agent testing guides
# put a healthy end-to-end median around 1.5 s; we flag a p95 above 2 s.
LATENCY_BUDGET_MS = 2000.0


@dataclass
class TurnResult:
    index: int
    said: str
    heard_by_agent: str
    uplink_wer: float
    agent_reply: str
    heard_by_caller: str | None
    downlink_wer: float | None
    uplink: dict
    downlink: dict
    uplink_mos: float | None
    downlink_mos: float | None
    latency_ms: float
    timings_ms: dict[str, float]


@dataclass
class CallResult:
    id: str
    scenario: str
    profile: dict
    seed: int
    turns: list[TurnResult]
    slots: dict[str, str]
    expected: dict[str, str]
    task_success: bool
    verdict: str = "pending"  # pass | network | latency | agent
    diagnosis: str = ""
    audio_dir: str | None = None

    # summary numbers -------------------------------------------------------
    @property
    def mean_uplink_wer(self) -> float:
        return round(statistics.fmean(t.uplink_wer for t in self.turns), 3)

    @property
    def mean_uplink_mos(self) -> float | None:
        vals = [t.uplink_mos for t in self.turns if t.uplink_mos is not None]
        return round(statistics.fmean(vals), 2) if vals else None

    @property
    def p95_latency_ms(self) -> float:
        lat = sorted(t.latency_ms for t in self.turns)
        return round(lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))], 1)

    @property
    def effective_loss_pct(self) -> float:
        sent = sum(t.uplink["packets_sent"] for t in self.turns)
        bad = sum(t.uplink["packets_lost"] + t.uplink["packets_late"] for t in self.turns)
        return round(100 * bad / sent, 2) if sent else 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def summary(self) -> dict:
        return {
            "id": self.id,
            "scenario": self.scenario,
            "profile": self.profile["name"],
            "task_success": self.task_success,
            "verdict": self.verdict,
            "mean_uplink_wer": self.mean_uplink_wer,
            "mean_uplink_mos": self.mean_uplink_mos,
            "p95_latency_ms": self.p95_latency_ms,
            "effective_loss_pct": self.effective_loss_pct,
            "diagnosis": self.diagnosis,
        }

    def to_dict(self) -> dict:
        """Everything about the call. `profile` is the profile name (as in the
        summary); the full network settings are under `network`."""
        d = asdict(self)
        d["network"] = d.pop("profile")
        d.update(self.summary())
        return d


AgentFactory = Callable[[], CascadeAgent]


class CallRunner:
    def __init__(
        self,
        agent_factory: AgentFactory,
        caller_tts: Callable[[str], TTS],
        caller_stt: STT | None = None,
        quality: DNSMOS | None = None,
        audio_root: str | Path | None = None,
    ) -> None:
        """
        agent_factory: makes a fresh agent per call (agents keep conversation state).
        caller_tts:    voice name -> TTS engine for the caller.
        caller_stt:    if given, also measures how well the caller understood the agent.
        quality:       DNSMOS scorer; skipped when None or the model isn't downloaded.
        audio_root:    if given, saves what each side heard, for playback.
        """
        self.agent_factory = agent_factory
        self.caller_tts = caller_tts
        self.caller_stt = caller_stt
        self.quality = quality if quality is not None and quality.available else None
        self.audio_root = Path(audio_root) if audio_root else None

    def warmup(self) -> None:
        """Load speech models before timing anything, so the first call isn't
        charged for model loading."""
        engines = [self.agent_factory().stt, self.caller_stt]
        for engine in engines:
            if engine is not None and hasattr(engine, "warmup"):
                engine.warmup()

    def run(self, scenario: Scenario, profile: NetworkProfile, seed: int = 0) -> CallResult:
        call_id = uuid.uuid4().hex[:12]
        agent = self.agent_factory()
        tts = self.caller_tts(scenario.caller_voice)
        audio_dir = self.audio_root / call_id if self.audio_root else None

        agent.greet()  # the greeting is not scored; it just starts the conversation
        turns: list[TurnResult] = []
        for i, line in enumerate(scenario.turns):
            turn_seed = seed * 1000 + i * 2
            said_audio = tts.synthesize(line)
            up_audio, up_stats = transmit(said_audio, profile, seed=turn_seed)
            reply = agent.handle(up_audio)
            down_audio, down_stats = transmit(reply.reply_audio, profile, seed=turn_seed + 1)

            heard_by_caller = self.caller_stt.transcribe(down_audio) if self.caller_stt else None
            processing = sum(reply.timings_ms.values())
            latency = ENDPOINTING_MS + up_stats.delay_ms + processing + down_stats.delay_ms

            turns.append(
                TurnResult(
                    index=i,
                    said=line,
                    heard_by_agent=reply.heard,
                    uplink_wer=round(wer(line, reply.heard), 3),
                    agent_reply=reply.reply_text,
                    heard_by_caller=heard_by_caller,
                    downlink_wer=(
                        round(wer(reply.reply_text, heard_by_caller), 3)
                        if heard_by_caller is not None
                        else None
                    ),
                    uplink=up_stats.to_dict(),
                    downlink=down_stats.to_dict(),
                    uplink_mos=self._mos(up_audio),
                    downlink_mos=self._mos(down_audio),
                    latency_ms=round(latency, 1),
                    timings_ms=reply.timings_ms,
                )
            )
            if audio_dir:
                save(audio_dir / f"turn{i}_agent_heard.wav", up_audio)
                save(audio_dir / f"turn{i}_caller_heard.wav", down_audio)
            if agent.brain.done:
                break

        slots = {k: v.lower() for k, v in agent.brain.slots.items()}
        success = all(slots.get(k) == v for k, v in scenario.expected.items())
        return CallResult(
            id=call_id,
            scenario=scenario.name,
            profile=profile.to_dict(),
            seed=seed,
            turns=turns,
            slots=slots,
            expected=dict(scenario.expected),
            task_success=success,
            audio_dir=str(audio_dir) if audio_dir else None,
        )

    def _mos(self, audio: np.ndarray) -> float | None:
        if self.quality is None or duration_ms(audio) < 200:
            return None
        return round(self.quality.score(audio).ovrl, 2)


def run_sweep(
    runner: CallRunner,
    scenario: Scenario,
    profiles: list[NetworkProfile],
    seeds: int = 1,
    on_call: Callable[[CallResult], None] | None = None,
) -> list[CallResult]:
    """Run a clean baseline first, then every profile, and diagnose each call
    against the baseline so network faults and agent faults can be told apart."""
    from jitterbug.diagnosis import diagnose

    runner.warmup()
    baseline = runner.run(scenario, NetworkProfile("clean"), seed=0)
    diagnose(baseline, None)
    results = [baseline]
    if on_call:
        on_call(baseline)
    for profile in profiles:
        if profile.is_clean:
            continue
        for s in range(seeds):
            call = runner.run(scenario, profile, seed=s)
            diagnose(call, baseline)
            results.append(call)
            if on_call:
                on_call(call)
    return results


def sweep_table(results: list[CallResult]) -> list[dict]:
    """One row per profile: share of calls passed plus mean scores."""
    by_profile: dict[str, list[CallResult]] = {}
    for r in results:
        by_profile.setdefault(r.profile["name"], []).append(r)
    rows = []
    for name, calls in by_profile.items():
        mos = [c.mean_uplink_mos for c in calls if c.mean_uplink_mos is not None]
        rows.append(
            {
                "profile": name,
                "loss_pct": calls[0].profile["loss_pct"],
                "jitter_ms": calls[0].profile["jitter_ms"],
                "calls": len(calls),
                "task_success_pct": round(100 * sum(c.task_success for c in calls) / len(calls), 1),
                "pass_pct": round(100 * sum(c.passed for c in calls) / len(calls), 1),
                "mean_wer": round(statistics.fmean(c.mean_uplink_wer for c in calls), 3),
                "mean_mos": round(statistics.fmean(mos), 2) if mos else None,
                "p95_latency_ms": max(c.p95_latency_ms for c in calls),
                "verdicts": {
                    v: sum(c.verdict == v for c in calls)
                    for v in sorted({c.verdict for c in calls})
                },
            }
        )
    return rows


__all__ = [
    "CallResult",
    "CallRunner",
    "TurnResult",
    "run_sweep",
    "sweep_table",
    "ENDPOINTING_MS",
    "LATENCY_BUDGET_MS",
]
