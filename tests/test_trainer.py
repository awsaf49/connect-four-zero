import inspect
import time
from pathlib import Path

from connect4_zero.schemas import RunConfig, RunRecord, RunStatus
from connect4_zero.store import RunStore
from connect4_zero.trainer import TrainingController


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

