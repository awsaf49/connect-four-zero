import os
from pathlib import Path

os.environ["CONNECT4_RUNS_DIR"] = "/tmp/connect4-zero-api-tests"

import pytest
from pydantic import ValidationError
from httpx import ASGITransport, AsyncClient

from connect4_zero.api import app
from connect4_zero.schemas import RunConfig


@pytest.mark.asyncio
async def test_system_and_empty_run_list() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/system")
        assert response.status_code == 200
        assert response.json()["model_parameters"] > 400_000
        assert (await client.get("/api/runs")).status_code == 200


@pytest.mark.asyncio
async def test_human_game_rejects_illegal_move() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        game = await client.post("/api/games", json={"difficulty": "beginner", "human_player": 1, "simulations": 2})
        assert game.status_code == 201
        game_id = game.json()["id"]
        response = await client.post(f"/api/games/{game_id}/moves", json={"action": 9})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_human_game_analysis_contains_bounded_solver() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        game = (await client.post("/api/games", json={"difficulty": "beginner", "human_player": 1, "simulations": 2})).json()
        analysis = await client.get(f"/api/games/{game['id']}/analysis")
        assert analysis.status_code == 200
        assert "solver" in analysis.json()
        assert len(analysis.json()["solver"]["grades"]) == 7


@pytest.mark.asyncio
async def test_invalid_compute_counts_are_rejected_before_worker_start() -> None:
    with pytest.raises(ValidationError):
        RunConfig(simulations=0)
    with pytest.raises(ValidationError):
        RunConfig(evaluation_games=0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/games", json={"simulations": 0})
        assert response.status_code == 422
