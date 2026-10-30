import pytest

from jitterbug.agent import BookingBrain, split_sentences
from jitterbug.diagnosis import diagnose
from jitterbug.network import NetworkProfile, loss_sweep
from jitterbug.runner import LATENCY_BUDGET_MS, run_sweep, sweep_table
from jitterbug.scenarios import list_scenarios


@pytest.mark.parametrize(
    "heard, slots",
    [
        (
            "A table for 4 at 8:30 p.m., the name is Kapoor.",
            {"party_size": "4", "time": "8:30 pm", "name": "kapoor"},
        ),
        ("table for two at 7pm", {"party_size": "2", "time": "7:00 pm"}),
        ("at 830 PM tonight", {"time": "8:30 pm"}),
        ("party of six", {"party_size": "6"}),
        ("it's Sharma", {"name": "sharma"}),
    ],
)
def test_brain_extracts_slots(heard, slots):
    brain = BookingBrain()
    brain.reply(heard)
    assert brain.slots == slots


def test_brain_asks_for_missing_details_then_confirms():
    brain = BookingBrain()
    assert "time" in brain.reply("a table for four please")
    assert "name" in brain.reply("at eight pm")
    final = brain.reply("the name is Rao")
    assert brain.done and "Rao" in final
    assert "didn't catch" in BookingBrain().reply("")


def test_split_sentences():
    assert split_sentences("Great. See you then!  Bye?") == ["Great.", "See you then!", "Bye?"]


def test_clean_call_passes(make_runner, booking):
    call = make_runner().run(booking, NetworkProfile())
    diagnose(call, None)
    assert call.task_success and call.verdict == "pass"
    assert call.mean_uplink_wer == 0
    assert len(call.turns) == 3
    assert all(t.downlink_wer == 0 for t in call.turns)


def test_heavy_loss_is_blamed_on_the_network(make_runner, booking):
    runner = make_runner()
    baseline = diagnose(runner.run(booking, NetworkProfile()), None)
    bad = runner.run(booking, NetworkProfile("awful", loss_pct=60, burst_len=8), seed=3)
    diagnose(bad, baseline)
    assert not bad.task_success
    assert bad.verdict == "network"
    assert "clean network" in bad.diagnosis and "Uplink lost" in bad.diagnosis


def test_agent_fault_when_it_fails_even_on_clean(make_runner, booking):
    broken = type(booking)(
        name="x", turns=booking.turns, expected={**booking.expected, "name": "someone_else"}
    )
    runner = make_runner()
    baseline = diagnose(runner.run(broken, NetworkProfile()), None)
    assert baseline.verdict == "agent" or baseline.verdict == "network"
    call = diagnose(runner.run(broken, NetworkProfile("meh", loss_pct=2)), baseline)
    assert call.verdict == "agent"
    assert "clean network" in call.diagnosis


def test_slow_replies_get_latency_verdict(make_runner, booking):
    far = NetworkProfile("satellite", latency_ms=LATENCY_BUDGET_MS)
    call = diagnose(make_runner().run(booking, far), None)
    assert call.task_success and call.verdict == "latency"
    assert "network delay" in call.diagnosis


def test_sweep_success_falls_as_loss_rises(make_runner, booking):
    results = run_sweep(make_runner(), booking, loss_sweep((0, 5, 40)), seeds=6)
    table = {row["profile"]: row for row in sweep_table(results)}
    assert table["clean"]["task_success_pct"] == 100
    assert table["loss_40pct"]["task_success_pct"] < table["loss_5pct"]["task_success_pct"]
    assert table["loss_40pct"]["mean_wer"] > table["loss_5pct"]["mean_wer"]


def test_audio_is_saved_when_asked(make_runner, booking, tmp_path):
    call = make_runner(audio_root=tmp_path).run(booking, NetworkProfile())
    files = sorted(p.name for p in (tmp_path / call.id).iterdir())
    assert files[:2] == ["turn0_agent_heard.wav", "turn0_caller_heard.wav"]


def test_bundled_scenarios_load():
    scenarios = list_scenarios()
    assert {"restaurant_booking", "booking_one_breath"} <= set(scenarios)
    assert all(s.turns and s.expected for s in scenarios.values())
