"""The caller says a line again when the agent didn't get it, like a person would."""

from jitterbug.diagnosis import diagnose
from jitterbug.network import NetworkProfile
from jitterbug.runner import run_sweep, sweep_table
from jitterbug.scoring.wer import normalize

LOSSY = NetworkProfile("lossy", loss_pct=30, burst_len=4)


def clean_baseline(runner, scenario):
    return diagnose(runner.run(scenario, NetworkProfile()), None)


def test_clean_call_never_repeats(make_runner, booking):
    runner = make_runner()
    baseline = clean_baseline(runner, booking)
    call = runner.run(booking, NetworkProfile(), baseline=baseline)
    assert call.repeats == 0 and call.task_success


def test_caller_repeats_only_when_the_agent_answered_differently(make_runner, booking):
    runner = make_runner()
    baseline = clean_baseline(runner, booking)
    clean = {t.index: normalize(t.agent_reply) for t in baseline.turns}
    calls = [runner.run(booking, LOSSY, seed=s, baseline=baseline) for s in range(20)]
    assert sum(c.repeats for c in calls) > 0  # 30% loss does make the agent miss lines

    for call in calls:
        for prev, turn in zip(call.turns, call.turns[1:], strict=False):
            if turn.attempt:
                # a repeat follows the same line, and only after a different answer
                assert (turn.index, turn.said) == (prev.index, prev.said)
                assert turn.attempt == prev.attempt + 1 <= runner.max_repeats
                assert normalize(prev.agent_reply) != clean[prev.index]
            else:
                # moved on: the agent answered as on a clean line, or repeats ran out
                missed = normalize(prev.agent_reply) != clean[prev.index]
                assert not missed or prev.attempt == runner.max_repeats


def test_first_attempts_match_a_call_without_repeats(make_runner, booking):
    """Repeats use their own random losses, so everything up to the first
    repeat is exactly what the caller-never-repeats version saw."""
    runner = make_runner()
    baseline = clean_baseline(runner, booking)
    for s in range(20):
        runner.max_repeats = 2
        with_repeats = runner.run(booking, LOSSY, seed=s, baseline=baseline)
        runner.max_repeats = 0
        without = runner.run(booking, LOSSY, seed=s, baseline=baseline)
        assert without.repeats == 0
        # (the fake voice makes each clip a little longer, so compare what was
        # lost and heard rather than the packet count)
        a, b = with_repeats.turns[0], without.turns[0]
        assert a.heard_by_agent == b.heard_by_agent
        assert (a.uplink["packets_lost"], a.uplink["longest_gap"]) == (
            b.uplink["packets_lost"],
            b.uplink["longest_gap"],
        )


def test_no_baseline_means_no_repeats(make_runner, booking):
    runner = make_runner()
    calls = [runner.run(booking, LOSSY, seed=s) for s in range(10)]
    assert all(c.repeats == 0 for c in calls)


def test_repeating_rescues_bookings(make_runner, booking):
    profiles = [LOSSY]
    with_repeats = sweep_table(run_sweep(make_runner(), booking, profiles, seeds=20))
    without = sweep_table(run_sweep(make_runner(max_repeats=0), booking, profiles, seeds=20))
    lossy_with, lossy_without = with_repeats[1], without[1]
    assert lossy_with["task_success_pct"] > lossy_without["task_success_pct"]
    assert lossy_with["mean_repeats"] > 0 == lossy_without["mean_repeats"]


def test_diagnosis_mentions_repeats(make_runner, booking):
    runner = make_runner()
    baseline = clean_baseline(runner, booking)
    calls = [
        diagnose(runner.run(booking, LOSSY, seed=s, baseline=baseline), baseline) for s in range(20)
    ]
    for call in calls:
        if call.repeats and call.verdict in ("pass", "network"):
            assert "repeated" in call.diagnosis, call.diagnosis
