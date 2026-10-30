"""Test scenarios: what the caller says and what the agent should end up with."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: list[str]
    expected: dict[str, str]
    description: str = ""
    caller_voice: str = "Rishi"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        if not d.get("turns"):
            raise ValueError("scenario needs at least one turn")
        return cls(
            name=d["name"],
            turns=[str(t) for t in d["turns"]],
            expected={k: str(v).lower() for k, v in (d.get("expected") or {}).items()},
            description=d.get("description", ""),
            caller_voice=d.get("caller_voice", "Rishi"),
            tags=list(d.get("tags", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def list_scenarios(directory: str | Path = SCENARIO_DIR) -> dict[str, Scenario]:
    out = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        s = Scenario.load(p)
        out[s.name] = s
    return out
