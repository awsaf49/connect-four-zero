from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class RunConfig(BaseModel):
    version: int = 1
    minutes: float | None = 60.0
    simulations: int = 256
    evaluation_simulations: int = 256
    self_play_games: int = 64
    evaluation_games: int = 80
    training_steps: int = 200
    batch_size: int = 512
    replay_capacity: int = 100_000
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 10
    promotion_threshold: float = 0.55
    seed: int = 7
    allow_cpu: bool = False
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    @field_validator("minutes")
    @classmethod
    def validate_minutes(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("minutes must be positive or null for unlimited")
        return value


class RunRecord(BaseModel):
    version: int = 1
    id: str
    name: str
    status: RunStatus
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    config: RunConfig
    phase: str = "ready"
    generation: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


class MetricSample(BaseModel):
    version: int = 1
    step: int
    wall_time: float
    values: dict[str, float | int | str | bool]


class CheckpointRecord(BaseModel):
    version: int = 1
    id: str
    run_id: str
    generation: int
    training_step: int
    kind: Literal["initial", "candidate", "champion", "rejected", "final", "demo"]
    path: str
    created_at: datetime = Field(default_factory=utc_now)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchTrace(BaseModel):
    version: int = 1
    simulations: int
    priors: list[float]
    visits: list[int]
    q_values: list[float]
    root_value: float
    principal_variations: list[list[int]] = Field(default_factory=list)
    snapshots: list[dict[str, Any]] = Field(default_factory=list)


class MoveTrace(BaseModel):
    version: int = 1
    ply: int
    player: int
    action: int
    board_before: list[list[int]]
    board_after: list[list[int]]
    search: SearchTrace | None = None
    solver_grade: str | None = None


class GameReplay(BaseModel):
    version: int = 1
    id: str
    run_id: str
    source: Literal["self_play", "evaluation", "human", "demo"]
    player_one: str
    player_two: str
    winner: int
    moves: list[MoveTrace]
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    version: int = 1
    mps_available: bool
    mps_built: bool
    device: str
    active_run_id: str | None
    wandb_available: bool
    model_parameters: int

