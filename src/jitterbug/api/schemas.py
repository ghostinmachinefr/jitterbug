"""Request and response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProfileIn(BaseModel):
    name: str = Field(..., max_length=100)
    loss_pct: float = Field(0, ge=0, lt=100)
    burst_len: float = Field(1, ge=1, le=50)
    jitter_ms: float = Field(0, ge=0, le=1000)
    latency_ms: float = Field(0, ge=0, le=5000)
    jitter_buffer_ms: float | None = Field(None, ge=0, le=2000)
    codec: Literal["wideband", "narrowband"] = "wideband"


class RunCreate(BaseModel):
    scenario: str = Field(..., examples=["restaurant_booking"])
    preset: Literal["presets", "loss_sweep"] | None = Field(
        None, description="Use a ready-made set of network profiles."
    )
    profiles: list[str | ProfileIn] | None = Field(
        None, description="Preset names and/or custom profiles."
    )
    seeds: int = Field(1, ge=1, le=20, description="Repeats per profile (different random losses).")

    @model_validator(mode="after")
    def need_profiles(self) -> RunCreate:
        if not self.preset and not self.profiles:
            raise ValueError("give either 'preset' or 'profiles'")
        return self


class CallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    profile: str
    verdict: str
    task_success: bool
    mean_wer: float
    mean_mos: float | None
    p95_latency_ms: float
    effective_loss_pct: float
    diagnosis: str


class CallDetail(CallOut):
    run_id: str
    detail: dict


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    scenario: str
    status: str
    seeds: int
    profiles: list
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None
    summary: list | None


class RunDetail(RunOut):
    calls: list[CallOut]
