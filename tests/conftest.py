import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import FakeSpeech  # noqa: E402

from jitterbug.agent import CascadeAgent  # noqa: E402
from jitterbug.runner import CallRunner  # noqa: E402
from jitterbug.scenarios import Scenario  # noqa: E402


@pytest.fixture
def speech():
    return FakeSpeech()


@pytest.fixture
def make_runner(speech):
    def _make(**kwargs):
        stt = speech.stt()
        return CallRunner(
            agent_factory=lambda: CascadeAgent(stt, speech.tts("agent")),
            caller_tts=speech.tts,
            caller_stt=stt,
            quality=None,
            **kwargs,
        )

    return _make


@pytest.fixture
def booking():
    return Scenario.from_dict(
        {
            "name": "booking",
            "turns": [
                "I would like a table for four people please",
                "at eight thirty pm",
                "the name is kapoor",
            ],
            "expected": {"party_size": "4", "time": "8:30 pm", "name": "kapoor"},
        }
    )
