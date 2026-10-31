import time

import pytest
from fastapi.testclient import TestClient

from jitterbug.api.main import create_app


@pytest.fixture
def client(make_runner, tmp_path):
    app = create_app(
        runner_factory=lambda: make_runner(audio_root=tmp_path / "audio"),
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
    )
    with TestClient(app) as c:
        yield c


def wait_for(client, run_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] in ("done", "failed"):
            return run
        time.sleep(0.05)
    raise TimeoutError(run)


def test_home_opens_the_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307 and resp.headers["location"] == "/docs"


def test_health_and_catalogue(client):
    assert client.get("/health").json()["status"] == "ok"
    names = {s["name"] for s in client.get("/scenarios").json()}
    assert "restaurant_booking" in names
    assert any(p["name"] == "phone_line" for p in client.get("/profiles").json())


def test_full_run_lifecycle(client):
    body = {
        "scenario": "restaurant_booking",
        "profiles": ["phone_line", {"name": "awful", "loss_pct": 60, "burst_len": 8}],
        "seeds": 2,
    }
    resp = client.post("/runs", json=body)
    assert resp.status_code == 202
    run_id = resp.json()["id"]

    run = wait_for(client, run_id)
    assert run["status"] == "done", run["error"]
    assert len(run["calls"]) == 1 + 2 * 2  # clean baseline + 2 profiles x 2 seeds
    profiles = {row["profile"] for row in run["summary"]}
    assert profiles == {"clean", "phone_line", "awful"}

    failed = client.get(f"/runs/{run_id}/calls", params={"verdict": "network"}).json()
    assert failed, "60% bursty loss should break at least one call"

    call = client.get(f"/calls/{failed[0]['id']}").json()
    assert call["detail"]["turns"][0]["uplink"]["packets_sent"] > 0
    assert call["detail"]["network"]["loss_pct"] == 60

    audio = client.get(f"/calls/{call['id']}/audio/turn0_agent_heard.wav")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert client.get(f"/calls/{call['id']}/audio/..%2F..%2Fsecret.wav").status_code == 404

    assert client.get("/runs").json()[0]["id"] == run_id


@pytest.mark.parametrize(
    "body, code",
    [
        ({"scenario": "nope", "preset": "presets"}, 404),
        ({"scenario": "restaurant_booking"}, 422),
        ({"scenario": "restaurant_booking", "profiles": ["mars"]}, 422),
        ({"scenario": "restaurant_booking", "profiles": [{"name": "x", "loss_pct": 150}]}, 422),
    ],
)
def test_bad_requests(client, body, code):
    assert client.post("/runs", json=body).status_code == code


def test_unknown_ids_404(client):
    assert client.get("/runs/missing").status_code == 404
    assert client.get("/calls/missing").status_code == 404
