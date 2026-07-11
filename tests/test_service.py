from pathlib import Path

import pytest

from connect4_zero.game import ConnectFour
from connect4_zero.schemas import CheckpointRecord, RunConfig, RunRecord, RunStatus
from connect4_zero.service import ArenaService, EventBroker, HumanGame, TrainingManager, reconcile_interrupted_runs
from connect4_zero.store import RunRepository


def test_completed_and_demo_runs_cannot_resume(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path)
    store = repository.open("demo-bundled")
    store.create_run(
        RunRecord(
            id="demo-bundled",
            name="Demo",
            status=RunStatus.COMPLETED,
            config=RunConfig(allow_cpu=True, wandb_mode="disabled"),
        )
    )
    manager = TrainingManager(repository, EventBroker())
    with pytest.raises(ValueError, match="cannot be resumed"):
        manager.action("demo-bundled", "resume")


def test_reconcile_marks_dead_worker_interrupted(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path)
    store = repository.open("stale")
    store.create_run(
        RunRecord(
            id="stale",
            name="Stale",
            status=RunStatus.RUNNING,
            config=RunConfig(allow_cpu=True, wandb_mode="disabled"),
            worker_pid=999_999_999,
        )
    )
    reconcile_interrupted_runs(repository)
    assert store.get_run().status == RunStatus.INTERRUPTED  # type: ignore[union-attr]


def test_candidate_checkpoint_is_not_publicly_deployable(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path)
    store = repository.open("candidate-run")
    store.create_run(
        RunRecord(
            id="candidate-run",
            name="Candidate",
            status=RunStatus.COMPLETED,
            config=RunConfig(allow_cpu=True, wandb_mode="disabled"),
        )
    )
    store.add_checkpoint(
        CheckpointRecord(
            id="same-id",
            run_id="candidate-run",
            generation=0,
            training_step=1,
            kind="candidate",
            path=str(tmp_path / "candidate.pt"),
        )
    )
    with pytest.raises(ValueError, match="not deployable"):
        ArenaService(repository).create_game(checkpoint_id="same-id")


def test_completed_human_game_is_persisted_once(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path)
    arena = ArenaService(repository)
    game = HumanGame(
        id="human-win",
        state=ConnectFour.from_moves([0, 6, 1, 6, 2, 5, 3]),
        human_player=1,
        difficulty="champion",
        checkpoint=None,
        evaluator=None,  # type: ignore[arg-type]
        search=type("Search", (), {"simulations": 800})(),  # type: ignore[arg-type]
    )

    arena._persist_completed_game(game)
    arena._persist_completed_game(game)

    saved = repository.open("human-games").list_games()
    assert len(saved) == 1
    assert saved[0].winner == 1
    assert saved[0].metadata["simulations"] == 800
