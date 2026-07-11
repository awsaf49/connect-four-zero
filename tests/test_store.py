from pathlib import Path

from connect4_zero.schemas import CheckpointRecord, GameReplay, MetricSample, RunConfig, RunRecord, RunStatus
from connect4_zero.store import RunRepository, RunStore


def test_store_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run-test")
    run = RunRecord(id="run-test", name="Test", status=RunStatus.CREATED, config=RunConfig(wandb_mode="disabled"))
    store.create_run(run)
    assert store.get_run() == run
    assert store.update_run(status=RunStatus.RUNNING).status == RunStatus.RUNNING

    store.add_metric(MetricSample(step=1, wall_time=0.5, values={"loss": 1.2}))
    assert store.list_metrics()[0].values["loss"] == 1.2

    checkpoint = CheckpointRecord(
        id="cp-1", run_id="run-test", generation=0, training_step=1, kind="initial", path="model.pt"
    )
    store.add_checkpoint(checkpoint)
    assert store.list_checkpoints()[0] == checkpoint

    game = GameReplay(id="g-1", run_id="run-test", source="self_play", player_one="a", player_two="a", winner=0, moves=[])
    store.add_game(game)
    assert store.get_game("g-1") == game
    assert RunRepository(tmp_path).list_runs()[0].id == "run-test"


def test_store_instances_share_atomic_run_updates(tmp_path: Path) -> None:
    first = RunStore(tmp_path, "shared")
    second = RunStore(tmp_path, "shared")
    first.create_run(RunRecord(id="shared", name="Shared", status=RunStatus.CREATED, config=RunConfig()))
    first.update_run(status=RunStatus.RUNNING)
    second.update_run(phase="self_play")
    current = first.get_run()
    assert current is not None
    assert current.status == RunStatus.RUNNING
    assert current.phase == "self_play"
