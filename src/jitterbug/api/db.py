"""Database tables. SQLite by default; set JITTERBUG_DATABASE_URL to a
PostgreSQL URL (postgresql+psycopg://user:pass@host/db) for production."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DEFAULT_URL = "sqlite:///./jitterbug.db"


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    scenario: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    profiles: Mapped[list] = mapped_column(JSON)
    seeds: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[list | None] = mapped_column(JSON)

    calls: Mapped[list[Call]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Call.created_at"
    )


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    profile: Mapped[str] = mapped_column(String(100), index=True)
    verdict: Mapped[str] = mapped_column(String(20), index=True)
    task_success: Mapped[bool] = mapped_column(Boolean)
    mean_wer: Mapped[float] = mapped_column(Float)
    mean_mos: Mapped[float | None] = mapped_column(Float)
    p95_latency_ms: Mapped[float] = mapped_column(Float)
    effective_loss_pct: Mapped[float] = mapped_column(Float)
    diagnosis: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    run: Mapped[Run] = relationship(back_populates="calls")


def make_sessionmaker(url: str | None = None) -> sessionmaker:
    url = url or os.environ.get("JITTERBUG_DATABASE_URL", DEFAULT_URL)
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
