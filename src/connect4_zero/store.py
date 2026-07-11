from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .schemas import CheckpointRecord, GameReplay, MetricSample, RunConfig, RunRecord, RunStatus, utc_now


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.root / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.export_dir = self.root / "exports"
        self.export_dir.mkdir(exist_ok=True)
        self.replay_path = self.root / "replay_buffer.npz"
        self.database_path = self.root / "run.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS run (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    wall_time REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS metrics_step_idx ON metrics(step);
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS games_source_idx ON games(source);
                """
            )

    def create_run(self, run: RunRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO run(id, payload) VALUES (?, ?)",
                (run.id, run.model_dump_json()),
            )

    def get_run(self) -> RunRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM run LIMIT 1").fetchone()
        return RunRecord.model_validate_json(row["payload"]) if row else None

    def update_run(self, **changes: object) -> RunRecord:
        run = self.get_run()
        if run is None:
            raise RuntimeError("run has not been initialized")
        updated = run.model_copy(update={**changes, "updated_at": utc_now()})
        self.create_run(updated)
        return updated

    def add_metric(self, metric: MetricSample) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO metrics(step, wall_time, payload) VALUES (?, ?, ?)",
                (metric.step, metric.wall_time, metric.model_dump_json()),
            )

    def list_metrics(self, limit: int = 2_000) -> list[MetricSample]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [MetricSample.model_validate_json(row["payload"]) for row in reversed(rows)]

    def add_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO checkpoints(id, generation, created_at, payload) VALUES (?, ?, ?, ?)",
                (
                    checkpoint.id,
                    checkpoint.generation,
                    checkpoint.created_at.isoformat(),
                    checkpoint.model_dump_json(),
                ),
            )

    def list_checkpoints(self) -> list[CheckpointRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT payload FROM checkpoints ORDER BY generation, created_at").fetchall()
        return [CheckpointRecord.model_validate_json(row["payload"]) for row in rows]

    def add_game(self, game: GameReplay) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO games(id, source, created_at, payload) VALUES (?, ?, ?, ?)",
                (game.id, game.source, game.created_at.isoformat(), game.model_dump_json()),
            )

    def list_games(self, source: str | None = None, limit: int = 200) -> list[GameReplay]:
        query = "SELECT payload FROM games"
        parameters: list[object] = []
        if source:
            query += " WHERE source = ?"
            parameters.append(source)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [GameReplay.model_validate_json(row["payload"]) for row in rows]

    def get_game(self, game_id: str) -> GameReplay | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT payload FROM games WHERE id = ?", (game_id,)).fetchone()
        return GameReplay.model_validate_json(row["payload"]) if row else None


class RunRepository:
    def __init__(self, root: Path = Path("runs")) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def open(self, run_id: str) -> RunStore:
        return RunStore(self.root, run_id)

    def list_runs(self) -> list[RunRecord]:
        runs: list[RunRecord] = []
        for database in self.root.glob("*/run.sqlite3"):
            run = RunStore(self.root, database.parent.name).get_run()
            if run:
                runs.append(run)
        return sorted(runs, key=lambda run: run.created_at, reverse=True)

