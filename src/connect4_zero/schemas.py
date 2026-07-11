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
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class RunConfig(BaseModel):
    version: int = 1
    minutes: float | None = 60.0
    simulations: int = Field(default=256, ge=1, le=10_000)
    evaluation_simulations: int = Field(default=256, ge=1, le=10_000)
    self_play_games: int = Field(default=64, ge=1, le=1_024)
    evaluation_games: int = Field(default=80, ge=2, le=10_000)
    training_steps: int = Field(default=200, ge=1, le=100_000)
    batch_size: int = Field(default=512, ge=1, le=65_536)
    replay_capacity: int = Field(default=100_000, ge=1)
    c_puct: float = Field(default=1.5, gt=0)
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1)
    temperature_moves: int = Field(default=10, ge=0, le=42)
    promotion_threshold: float = Field(default=0.55, ge=0, le=1)
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
    worker_pid: int | None = None


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
