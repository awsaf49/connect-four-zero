from __future__ import annotations

import asyncio
import os
import queue
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .network import PolicyValueNet
from .schemas import RunConfig, SystemStatus
from .service import ArenaService, EventBroker, TrainingManager
from .store import RunRepository

RUNS_ROOT = Path(os.getenv("CONNECT4_RUNS_DIR", "runs"))
repository = RunRepository(RUNS_ROOT)
broker = EventBroker()
training = TrainingManager(repository, broker)
arena = ArenaService(repository)

app = FastAPI(title="Connect Four Zero", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateRunRequest(BaseModel):
    name: str | None = None
    config: RunConfig = Field(default_factory=RunConfig)


class RunActionRequest(BaseModel):
    action: str


class CreateGameRequest(BaseModel):
    difficulty: str = "champion"
    human_player: int = 1
    checkpoint_id: str | None = None
    simulations: int | None = None


class MoveRequest(BaseModel):
    action: int


@app.get("/api/system", response_model=SystemStatus)
def system_status() -> SystemStatus:
    mps = torch.backends.mps.is_available()
    return SystemStatus(
        mps_available=mps,
        mps_built=torch.backends.mps.is_built(),
        device="mps" if mps else "unavailable",
        active_run_id=training.active_run_id,
        wandb_available=_wandb_available(),
        model_parameters=PolicyValueNet().parameter_count,
    )


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return [run.model_dump(mode="json") for run in repository.list_runs()]


@app.post("/api/runs", status_code=201)
def create_run(request: CreateRunRequest) -> dict[str, Any]:
    if training.active_run_id:
        raise HTTPException(409, "A training run is already active")
    if not request.config.allow_cpu and not torch.backends.mps.is_available():
        raise HTTPException(503, "MPS is required for production training")
    return training.create(request.config, request.name).model_dump(mode="json")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = repository.open(run_id).get_run()
    if run is None:
        raise HTTPException(404, "Run not found")
    return run.model_dump(mode="json")


@app.post("/api/runs/{run_id}/action")
def run_action(run_id: str, request: RunActionRequest) -> dict[str, Any]:
    try:
        return training.action(run_id, request.action).model_dump(mode="json")
    except KeyError:
        raise HTTPException(404, "Active run not found") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str, limit: int = 2_000) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in repository.open(run_id).list_metrics(limit)]


@app.get("/api/runs/{run_id}/checkpoints")
def run_checkpoints(run_id: str) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in repository.open(run_id).list_checkpoints()]


@app.get("/api/runs/{run_id}/games")
def run_games(run_id: str, source: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in repository.open(run_id).list_games(source, limit)]


@app.get("/api/runs/{run_id}/games/{game_id}")
def get_replay(run_id: str, game_id: str) -> dict[str, Any]:
    game = repository.open(run_id).get_game(game_id)
    if game is None:
        raise HTTPException(404, "Replay not found")
    return game.model_dump(mode="json")


@app.post("/api/games", status_code=201)
def create_game(request: CreateGameRequest) -> dict[str, Any]:
    try:
        return arena.create_game(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/games/{game_id}/moves")
def human_move(game_id: str, request: MoveRequest) -> dict[str, Any]:
    try:
        return arena.move(game_id, request.action)
    except KeyError:
        raise HTTPException(404, "Game not found") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/games/{game_id}/analysis")
def game_analysis(game_id: str) -> dict[str, Any]:
    try:
        return arena.analyze(game_id)
    except KeyError:
        raise HTTPException(404, "Game not found") from None


@app.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    subscriber = broker.subscribe(run_id)
    try:
        while True:
            try:
                event = await asyncio.to_thread(subscriber.get, True, 1.0)
                await websocket.send_json(event)
            except queue.Empty:
                await websocket.send_json({"version": 1, "type": "heartbeat", "run_id": run_id})
    except WebSocketDisconnect:
        pass
    finally:
        broker.unsubscribe(run_id, subscriber)


def _wandb_available() -> bool:
    try:
        import wandb

        return bool(wandb.api.api_key or os.getenv("WANDB_API_KEY"))
    except Exception:
        return False


WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        candidate = WEB_DIST / path
        return FileResponse(candidate if candidate.is_file() else WEB_DIST / "index.html")

