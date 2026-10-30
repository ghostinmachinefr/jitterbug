"""Jitterbug HTTP API.

uvicorn jitterbug.api.main:app --reload      # then open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from jitterbug import __version__
from jitterbug.api.db import Call, Run, make_sessionmaker
from jitterbug.api.jobs import JobQueue, resolve_profiles
from jitterbug.api.schemas import CallDetail, CallOut, RunCreate, RunDetail, RunOut
from jitterbug.network import PRESETS
from jitterbug.runner import CallRunner
from jitterbug.scenarios import list_scenarios

AUDIO_ROOT = Path("runs/audio")


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


DB = Annotated[Session, Depends(get_db)]


def default_runner() -> CallRunner:
    """Real engines: Whisper speech-to-text, macOS voices, DNSMOS scoring."""
    from jitterbug.agent import CascadeAgent
    from jitterbug.scoring.quality import DNSMOS
    from jitterbug.speech import MacSayTTS, WhisperSTT

    stt = WhisperSTT("base.en")
    agent_tts = MacSayTTS("Samantha")
    return CallRunner(
        agent_factory=lambda: CascadeAgent(stt, agent_tts),
        caller_tts=lambda voice: MacSayTTS(voice),
        caller_stt=stt,
        quality=DNSMOS(),
        audio_root=AUDIO_ROOT,
    )


def create_app(
    runner_factory: Callable[[], CallRunner] = default_runner,
    database_url: str | None = None,
    scenario_dir: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.sessions = make_sessionmaker(database_url)
        app.state.jobs = JobQueue(app.state.sessions, runner_factory)
        app.state.scenarios = list_scenarios(scenario_dir) if scenario_dir else list_scenarios()
        yield
        app.state.jobs.shutdown()

    app = FastAPI(
        title="Jitterbug",
        version=__version__,
        description="Call-quality testing for voice AI agents: scripted calls through "
        "emulated bad networks, scored for speech quality, word error rate, "
        "latency and task success, with a network-vs-agent root cause.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/scenarios")
    def scenarios(request: Request) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "turns": s.turns,
                "expected": s.expected,
                "caller_voice": s.caller_voice,
            }
            for s in request.app.state.scenarios.values()
        ]

    @app.get("/profiles")
    def profiles() -> list[dict]:
        return [p.to_dict() for p in PRESETS.values()]

    @app.post("/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
    def create_run(body: RunCreate, request: Request, session: DB) -> Run:
        scenario = request.app.state.scenarios.get(body.scenario)
        if scenario is None:
            raise HTTPException(404, f"unknown scenario '{body.scenario}'")
        try:
            profiles = resolve_profiles(body.preset, body.profiles)
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc).strip("'\"")) from exc
        run = Run(
            id=uuid.uuid4().hex[:12],
            scenario=scenario.name,
            seeds=body.seeds,
            profiles=[p.to_dict() for p in profiles],
            status="queued",
        )
        session.add(run)
        session.commit()
        request.app.state.jobs.submit(run.id, scenario, profiles, body.seeds)
        return run

    @app.get("/runs", response_model=list[RunOut])
    def list_runs(session: DB, limit: int = 20) -> list[Run]:
        limit = max(1, min(limit, 100))
        return list(session.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)))

    @app.get("/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str, session: DB) -> Run:
        run = session.scalar(select(Run).where(Run.id == run_id).options(selectinload(Run.calls)))
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    @app.get("/runs/{run_id}/calls", response_model=list[CallOut])
    def run_calls(run_id: str, session: DB, verdict: str | None = None) -> list[Call]:
        q = select(Call).where(Call.run_id == run_id).order_by(Call.created_at)
        if verdict:
            q = q.where(Call.verdict == verdict)
        return list(session.scalars(q))

    @app.get("/calls/{call_id}", response_model=CallDetail)
    def get_call(call_id: str, session: DB) -> Call:
        call = session.get(Call, call_id)
        if call is None:
            raise HTTPException(404, "call not found")
        return call

    @app.get("/calls/{call_id}/audio/{filename}")
    def call_audio(call_id: str, filename: str, session: DB) -> FileResponse:
        """Play back what the agent or the caller heard on a given turn."""
        call = session.get(Call, call_id)
        audio_dir = call.detail.get("audio_dir") if call else None
        if not audio_dir:
            raise HTTPException(404, "no audio saved for this call")
        base = Path(audio_dir).resolve()
        path = (base / filename).resolve()
        if path.parent != base or path.suffix != ".wav" or not path.exists():
            raise HTTPException(404, "audio file not found")
        return FileResponse(path, media_type="audio/wav")

    return app


app = create_app()
