from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .auditor import NegamaxSolver, Position
from .game import ConnectFour
from .mcts import MCTS, Node, SearchResult
from .network import PolicyValueEvaluator, PolicyValueNet
from .schemas import CheckpointRecord, GameReplay, MoveTrace, RunConfig, RunRecord, RunStatus, SearchTrace
from .store import RunRepository, RunStore
from .trainer import TrainingController


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[queue.Queue[dict[str, object]]]] = {}
        self._lock = threading.RLock()

    def publish(self, event: dict[str, object]) -> None:
        run_id = str(event.get("run_id", "system"))
        with self._lock:
            subscribers = list(self._subscribers.get(run_id, ()))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(event)
                except queue.Empty:
                    pass

    def subscribe(self, run_id: str) -> queue.Queue[dict[str, object]]:
        subscriber: queue.Queue[dict[str, object]] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.setdefault(run_id, set()).add(subscriber)
        return subscriber

    def unsubscribe(self, run_id: str, subscriber: queue.Queue[dict[str, object]]) -> None:
        with self._lock:
            self._subscribers.get(run_id, set()).discard(subscriber)


class TrainingManager:
    def __init__(self, repository: RunRepository, broker: EventBroker) -> None:
        self.repository = repository
        self.broker = broker
        self.controllers: dict[str, TrainingController] = {}
        self._lock = threading.RLock()

    @property
    def active_run_id(self) -> str | None:
        with self._lock:
            for run_id, controller in self.controllers.items():
                if controller.thread and controller.thread.is_alive():
                    return run_id
        return None

    def create(self, config: RunConfig, name: str | None = None) -> RunRecord:
        with self._lock:
            active_statuses = {
                RunStatus.CREATED,
                RunStatus.RUNNING,
                RunStatus.PAUSED,
                RunStatus.STOPPING,
                RunStatus.INTERRUPTED,
            }
            if self.active_run_id or any(run.status in active_statuses for run in self.repository.list_runs()):
                raise RuntimeError("A training run is already active")
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            run = RunRecord(
                id=run_id,
                name=name or f"Zero run {run_id[-4:]}",
                status=RunStatus.CREATED,
                config=config,
            )
            store = self.repository.open(run_id)
            store.create_run(run)
            controller = TrainingController(store, config, self.broker.publish)
            self.controllers[run_id] = controller
            controller.start()
        return run

    def action(self, run_id: str, action: str) -> RunRecord:
        with self._lock:
            controller = self.controllers.get(run_id)
            store = self.repository.open(run_id)
            run = store.get_run()
            if run is None:
                raise KeyError(run_id)
            if action == "resume" and controller is None:
                if run.status not in {RunStatus.PAUSED, RunStatus.INTERRUPTED}:
                    raise ValueError(f"run in status {run.status.value} cannot be resumed")
                if run.id == "demo-bundled":
                    raise ValueError("the bundled demo is read-only")
                if self.active_run_id:
                    raise RuntimeError("A training run is already active")
                controller = TrainingController(store, run.config, self.broker.publish)
                self.controllers[run_id] = controller
                controller.start()
                return store.get_run() or run
            if controller is None:
                raise KeyError(run_id)
        if action == "pause":
            if run.status != RunStatus.RUNNING:
                raise ValueError(f"run in status {run.status.value} cannot be paused")
            controller.pause()
        elif action == "resume":
            if run.status != RunStatus.PAUSED or not (controller.thread and controller.thread.is_alive()):
                raise ValueError(f"run in status {run.status.value} cannot be resumed")
            controller.resume()
        elif action == "stop":
            if run.status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
                raise ValueError(f"run in status {run.status.value} cannot be stopped")
            controller.stop()
        else:
            raise ValueError(action)
        run = store.get_run()
        if run is None:
            raise KeyError(run_id)
        return run


DIFFICULTY_SIMULATIONS = {
    "beginner": 32,
    "intermediate": 96,
    "hard": 256,
    "champion": 800,
}


@dataclass(slots=True)
class HumanGame:
    id: str
    state: ConnectFour
    human_player: int
    difficulty: str
    checkpoint: CheckpointRecord | None
    evaluator: PolicyValueEvaluator
    search: MCTS
    root: Node | None = None
    moves: list[MoveTrace] = field(default_factory=list)
    persisted: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class ArenaService:
    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository
        self.games: dict[str, HumanGame] = {}
        self._evaluators: dict[str, PolicyValueEvaluator] = {}
        self._lock = threading.RLock()
        self.auditor = NegamaxSolver(max_nodes=250_000, max_seconds=0.35)
        self._auditor_lock = threading.RLock()

    def available_checkpoints(self) -> list[CheckpointRecord]:
        records: list[CheckpointRecord] = []
        for run in self.repository.list_runs():
            records.extend(self.repository.open(run.id).list_checkpoints())
        return records

    def _pick_checkpoint(self, difficulty: str, checkpoint_id: str | None) -> CheckpointRecord | None:
        checkpoints = self.available_checkpoints()
        deployable = [item for item in checkpoints if item.kind in {"champion", "final", "demo"}]
        if checkpoint_id:
            matches = [item for item in deployable if item.id == checkpoint_id]
            if len(matches) > 1:
                raise ValueError("checkpoint id is ambiguous; use a run-qualified checkpoint")
            if not matches:
                raise ValueError("checkpoint is not deployable")
            return matches[0]
        champions = deployable
        if not champions:
            return None
        champions.sort(key=lambda item: (item.generation, item.training_step, item.created_at))
        if difficulty == "beginner":
            return champions[0]
        if difficulty == "intermediate":
            return champions[len(champions) // 2]
        return champions[-1]

    def _load_evaluator(self, checkpoint: CheckpointRecord | None) -> PolicyValueEvaluator:
        key = str(Path(checkpoint.path).resolve()) if checkpoint else "untrained"
        with self._lock:
            if key in self._evaluators:
                return self._evaluators[key]
            if checkpoint and Path(checkpoint.path).exists():
                evaluator, _ = PolicyValueEvaluator.load(Path(checkpoint.path))
            else:
                evaluator = PolicyValueEvaluator(PolicyValueNet())
            self._evaluators[key] = evaluator
            return evaluator

    def create_game(
        self,
        difficulty: str = "champion",
        human_player: int = 1,
        checkpoint_id: str | None = None,
        simulations: int | None = None,
    ) -> dict[str, Any]:
        if difficulty not in DIFFICULTY_SIMULATIONS and difficulty != "advanced":
            raise ValueError("unknown difficulty")
        if human_player not in (1, 2):
            raise ValueError("human_player must be 1 or 2")
        checkpoint = self._pick_checkpoint(difficulty, checkpoint_id)
        evaluator = self._load_evaluator(checkpoint)
        count = simulations if simulations is not None else DIFFICULTY_SIMULATIONS.get(difficulty, 256)
        game = HumanGame(
            id=f"game-{uuid.uuid4().hex[:12]}",
            state=ConnectFour.new(),
            human_player=human_player,
            difficulty=difficulty,
            checkpoint=checkpoint,
            evaluator=evaluator,
            search=MCTS(evaluator, simulations=count, seed=int(uuid.uuid4().int & 0xFFFFFFFF)),
        )
        self.games[game.id] = game
        ai_trace = self._ai_move(game) if human_player == 2 else None
        return self._game_payload(game, ai_trace=ai_trace)

    def move(self, game_id: str, action: int) -> dict[str, Any]:
        game = self.games.get(game_id)
        if game is None:
            raise KeyError(game_id)
        with game.lock:
            if game.state.is_terminal:
                raise ValueError("game is already complete")
            if game.state.current_player != game.human_player:
                raise ValueError("it is not the human player's turn")
            analysis, _ = game.search.search(game.state, record_snapshots=True)
            before = game.state
            audit = self._grade_position(before)
            game.state = before.play(action)
            human_trace = self._trace(before, game.state, action, analysis)
            grade = audit.grades.get(action)
            human_trace.solver_grade = self._grade_label(grade.outcome.value, grade.exact) if grade else None
            game.moves.append(human_trace)
            game.root = None
            ai_trace = None if game.state.is_terminal else self._ai_move(game)
            self._persist_completed_game(game)
            return self._game_payload(game, human_trace=human_trace, ai_trace=ai_trace)

    def _persist_completed_game(self, game: HumanGame) -> None:
        if game.persisted or not game.state.is_terminal:
            return
        checkpoint = game.checkpoint.id if game.checkpoint else "untrained"
        human = "human"
        self.repository.open("human-games").add_game(
            GameReplay(
                id=game.id,
                run_id="human-games",
                source="human",
                player_one=human if game.human_player == 1 else checkpoint,
                player_two=checkpoint if game.human_player == 1 else human,
                winner=game.state.winner,
                moves=list(game.moves),
                metadata={
                    "difficulty": game.difficulty,
                    "checkpoint": checkpoint,
                    "simulations": game.search.simulations,
                    "human_player": game.human_player,
                },
            )
        )
        game.persisted = True

    def analyze(self, game_id: str) -> dict[str, Any]:
        game = self.games.get(game_id)
        if game is None:
            raise KeyError(game_id)
        with game.lock:
            if game.state.is_terminal:
                return {"terminal": True, "winner": game.state.winner}
            result, _ = game.search.search(game.state, record_snapshots=True)
            audit = self._grade_position(game.state)
            return {**self._search_payload(result), "solver": self._audit_payload(audit)}

    def analyze_position(
        self,
        board: list[list[int]],
        current_player: int,
        simulations: int = 256,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        state = ConnectFour(np.asarray(board, dtype=np.int8), current_player=current_player)
        if state.is_terminal:
            return {"terminal": True, "winner": state.winner}
        checkpoint = self._pick_checkpoint("champion", checkpoint_id)
        evaluator = self._load_evaluator(checkpoint)
        result, _ = MCTS(evaluator, simulations=simulations, seed=17).search(state, record_snapshots=True)
        audit = self._grade_position(state)
        return {
            **self._search_payload(result),
            "solver": self._audit_payload(audit),
            "checkpoint": checkpoint.model_dump(mode="json") if checkpoint else None,
        }

    def _ai_move(self, game: HumanGame) -> MoveTrace:
        before = game.state
        audit = self._grade_position(before)
        result, root = game.search.search(before, root=game.root, record_snapshots=True)
        action = game.search.select_action(result, 0.0)
        game.state = before.play(action)
        trace = self._trace(before, game.state, action, result)
        grade = audit.grades.get(action)
        trace.solver_grade = self._grade_label(grade.outcome.value, grade.exact) if grade else None
        game.moves.append(trace)
        game.root = MCTS.advance_root(root, action)
        return trace

    def _game_payload(
        self,
        game: HumanGame,
        human_trace: MoveTrace | None = None,
        ai_trace: MoveTrace | None = None,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "id": game.id,
            "board": game.state.as_list(),
            "current_player": game.state.current_player,
            "human_player": game.human_player,
            "winner": game.state.winner,
            "terminal": game.state.is_terminal,
            "difficulty": game.difficulty,
            "checkpoint": game.checkpoint.model_dump(mode="json") if game.checkpoint else None,
            "model_label": game.checkpoint.id if game.checkpoint else "Untrained model",
            "human_trace": human_trace.model_dump(mode="json") if human_trace else None,
            "ai_trace": ai_trace.model_dump(mode="json") if ai_trace else None,
            "moves": [trace.model_dump(mode="json") for trace in game.moves],
        }

    @staticmethod
    def _trace(before: ConnectFour, after: ConnectFour, action: int, result: SearchResult) -> MoveTrace:
        return MoveTrace(
            ply=before.move_count,
            player=before.current_player,
            action=action,
            board_before=before.as_list(),
            board_after=after.as_list(),
            search=SearchTrace(**ArenaService._search_payload(result)),
        )

    @staticmethod
    def _search_payload(result: SearchResult) -> dict[str, Any]:
        return {
            "simulations": result.simulations,
            "tree_size": result.tree_size,
            "priors": [float(value) for value in result.priors],
            "visits": result.visits.tolist(),
            "q_values": [float(value) for value in result.q_values],
            "root_value": result.root_value,
            "principal_variations": result.principal_variations,
            "snapshots": [
                {
                    "simulation": item.simulation,
                    "visits": item.visits,
                    "q_values": item.q_values,
                    "priors": item.priors,
                    "root_value": item.root_value,
                }
                for item in result.snapshots
            ],
        }

    @staticmethod
    def _grade_label(outcome: str, exact: bool) -> str:
        return f"{outcome}:{'exact' if exact else 'bounded'}"

    def _grade_position(self, state: ConnectFour) -> Any:
        with self._auditor_lock:
            return self.auditor.grade_moves(Position.from_game(state))

    @staticmethod
    def _audit_payload(audit: Any) -> dict[str, Any]:
        return {
            "complete": audit.complete,
            "outcome": audit.outcome.value,
            "best_moves": list(audit.best_moves),
            "nodes": audit.nodes,
            "aborted_reason": audit.aborted_reason,
            "grades": {
                str(action): {
                    "outcome": grade.outcome.value,
                    "value": grade.value,
                    "exact": grade.exact,
                }
                for action, grade in audit.grades.items()
            },
        }


def reconcile_interrupted_runs(repository: RunRepository) -> None:
    for run in repository.list_runs():
        if run.status != RunStatus.RUNNING:
            continue
        if run.worker_pid is None:
            continue
        alive = False
        try:
            import os

            os.kill(run.worker_pid, 0)
            alive = True
        except OSError:
            alive = False
        if not alive:
            repository.open(run.id).update_run(
                status=RunStatus.INTERRUPTED,
                phase="interrupted",
                worker_pid=None,
            )
