from __future__ import annotations

import json
from pathlib import Path

from .schemas import CheckpointRecord, GameReplay, MetricSample, RunConfig, RunRecord, RunStatus
from .store import RunRepository


DEMO_RUN_ID = "demo-bundled"


def install_bundled_demo(repository: RunRepository, demo_dir: Path | None = None) -> None:
    demo_dir = demo_dir or Path(__file__).resolve().parents[2] / "demo"
    manifest_path = demo_dir / "manifest.json"
    checkpoint_path = demo_dir / "champion.pt"
    if not manifest_path.exists() or not checkpoint_path.exists():
        return
    store = repository.open(DEMO_RUN_ID)
    if store.get_run() is not None:
        return
    manifest = json.loads(manifest_path.read_text())
    config = RunConfig.model_validate(manifest["config"])
    store.create_run(
        RunRecord(
            id=DEMO_RUN_ID,
            name=manifest.get("name", "Bundled self-trained champion"),
            status=RunStatus.COMPLETED,
            config=config,
            phase="complete",
            generation=int(manifest["generation"]),
            elapsed_seconds=float(manifest["elapsed_seconds"]),
        )
    )
    store.add_checkpoint(
        CheckpointRecord(
            id="demo-champion",
            run_id=DEMO_RUN_ID,
            generation=int(manifest["generation"]),
            training_step=int(manifest["training_step"]),
            kind="demo",
            path=str(checkpoint_path.resolve()),
            metadata={**manifest.get("checkpoint_metadata", {}), "bundled": True},
        )
    )
    for item in manifest.get("metrics", []):
        store.add_metric(MetricSample.model_validate(item))
    for item in manifest.get("games", []):
        game = GameReplay.model_validate(item).model_copy(update={"run_id": DEMO_RUN_ID})
        store.add_game(game)

