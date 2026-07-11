import inspect
import time
from pathlib import Path

from connect4_zero.schemas import RunConfig, RunRecord, RunStatus
from connect4_zero.store import RunStore
from connect4_zero.trainer import TrainingController
from connect4_zero.trainer import ReplayBuffer, TrainingExample
from connect4_zero.game import ConnectFour
import numpy as np


def test_small_self_play_and_train_cycle(tmp_path: Path) -> None:
    config = RunConfig(
        minutes=1,
        simulations=2,
        evaluation_simulations=2,
        self_play_games=2,
        evaluation_games=2,
        training_steps=1,
        batch_size=8,
        allow_cpu=True,
        wandb_mode="disabled",
    )
    store = RunStore(tmp_path, "trainer-test")
    store.create_run(RunRecord(id="trainer-test", name="Trainer", status=RunStatus.CREATED, config=config))
    trainer = TrainingController(store, config)
    trainer.started_at = time.monotonic()
    examples, games = trainer._self_play(2)
    assert examples
    assert len(games) == 2
    trainer.replay.extend(examples)
    losses = trainer._train_candidate()
    assert losses["loss/total"] > 0
    assert 0 <= trainer._evaluate_candidate() <= 1


def test_checkpoint_restore_keeps_progress(tmp_path: Path) -> None:
    config = RunConfig(minutes=1, allow_cpu=True, wandb_mode="disabled")
    store = RunStore(tmp_path, "resume-test")
    store.create_run(RunRecord(id="resume-test", name="Resume", status=RunStatus.CREATED, config=config))
    trainer = TrainingController(store, config)
    trainer.step = 9
    trainer.generation = 2
    trainer._save_checkpoint("champion", 0.7)
    restored = TrainingController(store, config)
    assert restored.step == 9
    assert restored.generation == 2


def test_training_module_cannot_import_auditor() -> None:
    import connect4_zero.trainer as trainer_module

    source = inspect.getsource(trainer_module)
    assert "connect4_zero.auditor" not in source
    assert "from .auditor" not in source


def test_explicit_stop_has_distinct_terminal_status(tmp_path: Path) -> None:
    config = RunConfig(
        minutes=1,
        simulations=1,
        self_play_games=1,
        evaluation_games=2,
        training_steps=1,
        batch_size=2,
        allow_cpu=True,
        wandb_mode="disabled",
    )
    store = RunStore(tmp_path, "stop-test")
    store.create_run(RunRecord(id="stop-test", name="Stop", status=RunStatus.CREATED, config=config))
    trainer = TrainingController(store, config)
    trainer.stop()
    trainer.start()
    trainer.join(10)
    assert store.get_run().status == RunStatus.STOPPED  # type: ignore[union-attr]


def test_replay_augmentation_preserves_value_and_policy_geometry() -> None:
    state = ConnectFour.from_moves([0, 1, 0])
    policy = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    replay = ReplayBuffer(capacity=10, seed=3)
    replay.extend([TrainingExample(state.encode(), policy, -1.0)])
    boards, policies, values = replay.sample(64)
    assert set(np.argmax(policies, axis=1)).issubset({0, 6})
    assert np.all(values == -1.0)
    assert all(np.array_equal(board[0] + board[1], state.encode()[0] + state.encode()[1]) or np.array_equal(board[0] + board[1], np.flip(state.encode()[0] + state.encode()[1], axis=1)) for board in boards)
