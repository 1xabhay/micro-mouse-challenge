"""Dashboard HTTP API."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from micromouse.server.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


class TestHealth:
    def test_health_is_ok(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_serves_the_dashboard_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "canvas" in r.text.lower()


class TestBots:
    def test_lists_the_pool(self, client):
        bots = client.get("/api/bots").json()["bots"]
        names = {b["name"] for b in bots}
        assert {"floodfill", "dynaq", "ppo"} <= names
        assert all({"name", "style", "description"} <= set(b) for b in bots)


class TestMaze:
    def test_returns_wall_data_and_geometry(self, client):
        maze = client.get("/api/maze", params={"seed": 3}).json()
        assert maze["size"] == 16
        assert len(maze["walls"]) == 16 and len(maze["walls"][0]) == 16
        assert maze["start"] == [0, 0]
        assert len(maze["goal"]) == 4
        assert maze["optimal_cells"] > 16

    def test_wall_bitmask_uses_the_maz_encoding(self, client):
        maze = client.get("/api/maze", params={"seed": 3}).json()
        # start cell is a dead end open to the north: E|S|W = 2+4+8 = 14
        assert maze["walls"][0][0] == 14

    def test_is_reproducible(self, client):
        a = client.get("/api/maze", params={"seed": 5}).json()
        b = client.get("/api/maze", params={"seed": 5}).json()
        assert a == b

    def test_rejects_a_silly_size(self, client):
        assert client.get("/api/maze", params={"seed": 1, "size": 1}).status_code == 422


class TestRun:
    def test_runs_a_bot_and_returns_a_trace(self, client):
        r = client.post("/api/run", json={"bot": "floodfill", "seed": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["result"]["solved"] is True
        assert len(body["result"]["trace"]) > 10
        assert body["maze"]["size"] == 16

    def test_unknown_bot_is_a_client_error(self, client):
        r = client.post("/api/run", json={"bot": "nope", "seed": 1})
        assert r.status_code == 400
        assert "nope" in r.json()["detail"]

    def test_caps_the_step_budget(self, client):
        r = client.post("/api/run", json={"bot": "ppo", "seed": 1, "max_steps": 40})
        assert r.json()["result"]["steps"] <= 40


class TestRace:
    def test_races_bots_and_returns_a_leaderboard(self, client):
        r = client.post("/api/race", json={"bots": ["floodfill"], "mazes": 2})
        body = r.json()
        assert len(body["leaderboard"]) == 1
        assert body["leaderboard"][0]["attempts"] == 2
        assert len(body["attempts"]) == 2

    def test_race_rejects_too_many_mazes(self, client):
        r = client.post("/api/race", json={"bots": ["floodfill"], "mazes": 999})
        assert r.status_code == 422

    def test_race_rejects_an_empty_pool(self, client):
        r = client.post("/api/race", json={"bots": [], "mazes": 2})
        assert r.status_code == 422


class TestInfiniteScoreContract:
    def test_an_unsolved_bot_reports_a_null_score_not_a_number(self, client):
        """JSON has no infinity, so the API must not emit a misleading number."""
        r = client.post("/api/race", json={"bots": ["ppo"], "mazes": 6})
        raw = r.text
        assert "Infinity" not in raw, "invalid JSON would break the dashboard"
        rows = {row["bot"]: row for row in r.json()["leaderboard"]}
        ppo = rows["ppo"]
        if ppo["solved"] < ppo["attempts"]:
            assert ppo["mean_score"] is None
        assert ppo["solve_rate"] <= 1.0
