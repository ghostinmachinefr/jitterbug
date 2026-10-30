"""Root cause: was a bad call the network's fault or the agent's?

The key idea is a **counterfactual baseline**: the same scripted call is run
on a clean network first. Then for a failed call:

* If the agent also fails on the clean network, the network is not the cause.
  -> verdict "agent" (speech-to-text, logic or TTS problem in the agent itself).
* If it passes on clean but fails here, the network made the difference. We
  then point at the turn where the agent misheard the caller and report the
  packet loss, the longest audio gap and the words that were lost.
  -> verdict "network".
* If the task succeeded but replies were too slow -> verdict "latency", split
  into network delay vs. agent processing time.
"""

from __future__ import annotations

from jitterbug.network import FRAME_MS
from jitterbug.runner import LATENCY_BUDGET_MS, CallResult

MISHEARD_WER = 0.15  # a turn counts as misheard above this WER


def diagnose(call: CallResult, baseline: CallResult | None) -> CallResult:
    if call.task_success and call.p95_latency_ms <= LATENCY_BUDGET_MS:
        call.verdict, call.diagnosis = "pass", _pass_note(call)
        return call

    if call.task_success:  # right answer, too slow
        worst = max(call.turns, key=lambda t: t.latency_ms)
        net = worst.uplink["delay_ms"] + worst.downlink["delay_ms"]
        agent_ms = sum(worst.timings_ms.values())
        slow = max(worst.timings_ms, key=worst.timings_ms.get)
        call.verdict = "latency"
        call.diagnosis = (
            f"Task done, but turn {worst.index + 1} took {worst.latency_ms:.0f} ms "
            f"(budget {LATENCY_BUDGET_MS:.0f} ms): {net:.0f} ms network delay, "
            f"{agent_ms:.0f} ms agent processing (slowest step: {slow.replace('_ms', '')})."
        )
        return call

    wrong = _wrong_slots(call)
    if baseline is not None and not baseline.task_success:
        call.verdict = "agent"
        call.diagnosis = (
            f"Agent also fails on a clean network, so the network is not the cause. "
            f"Wrong or missing: {wrong}."
        )
        return call

    misheard = [t for t in call.turns if t.uplink_wer > MISHEARD_WER]
    if misheard:
        t = max(misheard, key=lambda x: x.uplink_wer)
        up = t.uplink
        call.verdict = "network"
        call.diagnosis = (
            f'Turn {t.index + 1}: agent heard "{t.heard_by_agent}" instead of "{t.said}" '
            f"(WER {t.uplink_wer:.0%}). Uplink lost {up['effective_loss_pct']:.1f}% of packets "
            f"({up['packets_lost']} dropped, {up['packets_late']} late), longest gap "
            f"{up['longest_gap'] * FRAME_MS} ms. Same call passes on a clean network. "
            f"Wrong or missing: {wrong}."
        )
        return call

    call.verdict = "agent" if baseline is not None else "network"
    call.diagnosis = (
        f"Agent heard every turn correctly but still got: {wrong}. Look at the agent's logic."
        if baseline is not None
        else f"No clean baseline to compare against. Wrong or missing: {wrong}."
    )
    return call


def _wrong_slots(call: CallResult) -> str:
    parts = []
    for k, v in call.expected.items():
        got = call.slots.get(k)
        if got != v:
            parts.append(f"{k} (expected '{v}', got '{got}')" if got else f"{k} (missing)")
    return ", ".join(parts) or "nothing"


def _pass_note(call: CallResult) -> str:
    notes = [f"p95 latency {call.p95_latency_ms:.0f} ms"]
    if call.mean_uplink_mos is not None:
        notes.append(f"MOS {call.mean_uplink_mos:.2f}")
    if call.mean_uplink_wer > 0:
        notes.append(f"WER {call.mean_uplink_wer:.0%} but task still done")
    return "Passed: " + ", ".join(notes) + "."
