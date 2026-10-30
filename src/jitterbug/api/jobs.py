"""Background execution of test runs.

Speech models are CPU-heavy, so runs execute one at a time on a worker
thread and the API stays responsive. Calls are written to the database as
they finish, so progress is visible while a run is going.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from sqlalchemy.orm import sessionmaker

from jitterbug.api.db import Call, Run, now
from jitterbug.network import PRESETS, NetworkProfile, loss_sweep
from jitterbug.runner import CallResult, CallRunner, run_sweep, sweep_table
from jitterbug.scenarios import Scenario

log = logging.getLogger("jitterbug.jobs")


def resolve_profiles(preset: str | None, profiles: list | None) -> list[NetworkProfile]:
    out: list[NetworkProfile] = []
    if preset == "presets":
        out += list(PRESETS.values())
    elif preset == "loss_sweep":
        out += loss_sweep()
    for p in profiles or []:
        if isinstance(p, str):
            if p not in PRESETS:
                raise KeyError(f"unknown profile '{p}'. Known: {', '.join(PRESETS)}")
            out.append(PRESETS[p])
        else:
            out.append(NetworkProfile(**(p if isinstance(p, dict) else p.model_dump())))
    return out


class JobQueue:
    def __init__(
        self, sessions: sessionmaker, runner_factory: Callable[[], CallRunner], workers: int = 1
    ) -> None:
        self.sessions = sessions
        self.runner_factory = runner_factory
        self._runner: CallRunner | None = None
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jitterbug")

    @property
    def runner(self) -> CallRunner:
        if self._runner is None:
            self._runner = self.runner_factory()
        return self._runner

    def submit(
        self, run_id: str, scenario: Scenario, profiles: list[NetworkProfile], seeds: int
    ) -> Future:
        return self._pool.submit(self._execute, run_id, scenario, profiles, seeds)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _execute(
        self, run_id: str, scenario: Scenario, profiles: list[NetworkProfile], seeds: int
    ) -> None:
        with self.sessions() as db:
            run = db.get(Run, run_id)
            run.status, run.started_at = "running", now()
            db.commit()

        def save_call(result: CallResult) -> None:
            s = result.summary()
            with self.sessions() as db:
                db.add(
                    Call(
                        id=result.id,
                        run_id=run_id,
                        profile=s["profile"],
                        verdict=s["verdict"],
                        task_success=s["task_success"],
                        mean_wer=s["mean_uplink_wer"],
                        mean_mos=s["mean_uplink_mos"],
                        p95_latency_ms=s["p95_latency_ms"],
                        effective_loss_pct=s["effective_loss_pct"],
                        diagnosis=s["diagnosis"],
                        detail=result.to_dict(),
                    )
                )
                db.commit()

        try:
            results = run_sweep(self.runner, scenario, profiles, seeds=seeds, on_call=save_call)
            status, error, summary = "done", None, sweep_table(results)
        except Exception as exc:  # a failed run must not kill the worker
            log.exception("run %s failed", run_id)
            status, error, summary = "failed", f"{exc}\n{traceback.format_exc(limit=3)}", None
        with self.sessions() as db:
            run = db.get(Run, run_id)
            run.status, run.error, run.summary, run.finished_at = status, error, summary, now()
            db.commit()
