import os
from pathlib import Path

os.environ["CONNECT4_RUNS_DIR"] = "/tmp/connect4-zero-api-tests"

from fastapi.testclient import TestClient

from connect4_zero.api import app


def test_system_and_empty_run_list() -> None:
    client = TestClient(app)
    response = client.get("/api/system")
    assert response.status_code == 200
    assert response.json()["model_parameters"] > 400_000
    assert client.get("/api/runs").status_code == 200


def test_human_game_rejects_illegal_move() -> None:
    client = TestClient(app)
    game = client.post("/api/games", json={"difficulty": "beginner", "human_player": 1, "simulations": 2})
    assert game.status_code == 201
    game_id = game.json()["id"]
    response = client.post(f"/api/games/{game_id}/moves", json={"action": 9})
    assert response.status_code == 422

