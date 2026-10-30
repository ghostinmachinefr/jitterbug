"""Command line.

jitterbug sweep restaurant_booking --preset loss_sweep --seeds 3
jitterbug sweep scenarios/my_test.yaml --preset presets
jitterbug sweep restaurant_booking --agent pipecat
jitterbug serve
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from jitterbug.network import PRESETS, loss_sweep
from jitterbug.scenarios import Scenario, list_scenarios


def _scenario(arg: str) -> Scenario:
    if arg.endswith((".yaml", ".yml")):
        return Scenario.load(arg)
    known = list_scenarios()
    if arg not in known:
        raise SystemExit(f"unknown scenario '{arg}'. Known: {', '.join(known)}")
    return known[arg]


def cmd_sweep(args: argparse.Namespace) -> None:
    from jitterbug.api.main import default_runner
    from jitterbug.report import loss_chart, print_table, save_results
    from jitterbug.runner import run_sweep, sweep_table

    scenario = _scenario(args.scenario)
    profiles = loss_sweep() if args.preset == "loss_sweep" else list(PRESETS.values())
    out_dir = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}-{scenario.name}"

    if args.agent == "pipecat":
        _quiet_pipecat_logs()
    runner = default_runner(args.agent)
    runner.audio_root = out_dir / "audio" if args.save_audio else None
    total = 1 + sum(not p.is_clean for p in profiles) * args.seeds
    done = 0
    started = time.time()

    def progress(call):
        nonlocal done
        done += 1
        print(
            f"[{done:>3}/{total}] {call.profile['name']:16} {call.verdict:8} "
            f"WER {call.mean_uplink_wer:.2f}  {call.p95_latency_ms:.0f} ms",
            flush=True,
        )

    print(
        f"Agent: {args.agent}. Running '{scenario.name}' under {len(profiles)} "
        f"network profiles x {args.seeds} seeds"
    )
    results = run_sweep(runner, scenario, profiles, seeds=args.seeds, on_call=progress)
    rows = sweep_table(results)
    print()
    print_table(rows)
    path = save_results(results, out_dir)
    print(f"\nSaved {path} ({time.time() - started:.0f} s)")
    if args.preset == "loss_sweep":
        chart = loss_chart(
            rows, out_dir / "loss_chart.png", f"{scenario.name}: effect of packet loss"
        )
        print(f"Chart: {chart}")


def _quiet_pipecat_logs() -> None:
    """Pipecat logs every frame at DEBUG level; keep warnings and errors only."""
    import sys

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("jitterbug.api.main:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jitterbug", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(required=True)

    sweep = sub.add_parser("sweep", help="run a scenario under many network conditions")
    sweep.add_argument("scenario", help="scenario name or path to a .yaml file")
    sweep.add_argument("--preset", choices=["presets", "loss_sweep"], default="presets")
    sweep.add_argument("--seeds", type=int, default=1, help="repeats per profile")
    sweep.add_argument("--out", default="runs")
    sweep.add_argument("--save-audio", action="store_true")
    sweep.add_argument(
        "--agent",
        choices=["cascade", "pipecat"],
        default="cascade",
        help="agent under test: built-in cascade, or the same bot as a Pipecat pipeline",
    )
    sweep.set_defaults(func=cmd_sweep)

    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
